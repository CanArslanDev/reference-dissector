import numpy as np
import pytest

from refdissect.audio import SR
from refdissect.synth808 import Params, compare, fit, laid_out, one_shot, voice


def _zero_crossing_hz(x):
    s = np.signbit(x)
    crossings = np.flatnonzero(s[1:] != s[:-1])
    return (len(crossings) - 1) / 2 / ((crossings[-1] - crossings[0]) / SR)


TRUE = Params(f_end_hz=41.2, f_start_hz=70.0, glide_ms=40, decay_s=0.8, attack_ms=2, drive=2.0, asym=0.05,
              phase=0.3)


def test_voice_pitch_and_drop():
    y = voice(TRUE, 1.0)
    assert abs(_zero_crossing_hz(y[int(0.4 * SR):]) - 41.2) < 0.5
    assert abs(TRUE.glide_semitones - 12 * np.log2(70 / 41.2)) < 1e-9
    moved = voice(TRUE, 1.0, pitch_hz=55.0)
    assert abs(_zero_crossing_hz(moved[int(0.4 * SR):]) - 55.0) < 0.6


def test_one_shot_level_and_fade():
    y = one_shot(TRUE, 41.2, seconds=1.5)
    assert abs(20 * np.log10(np.max(np.abs(y))) + 1.0) < 0.05
    assert abs(y[-1]) < 1e-3


def test_fit_gets_close():
    reference = laid_out(TRUE, int(0.6 * SR))
    p, info = fit(reference, 41.2, maxiter=25, popsize=10, seed=0)
    m = compare(reference, laid_out(p, len(reference)), 41.2)
    assert m["ncc"] > 0.9
    assert abs(p.f_end_hz - 41.2) < 1.5


def test_envelope_and_harmonics_match_their_direct_forms():
    """The O(n) moving mean and the sliced harmonic bands give what the
    convolution and the boolean masks gave (the 808 fit got twice as fast)."""
    import numpy as np
    from scipy.signal import hilbert

    from refdissect.audio import SR, samples
    from refdissect.describe import envelope_db, harmonics
    from refdissect.synth808 import Params, laid_out

    y = laid_out(Params(34, 55, 30, 1.2, 3, 2.5, 0.05, 0.3), int(0.65 * SR))
    env = np.abs(hilbert(y))
    w = samples(25)
    direct = np.sqrt(np.convolve(env ** 2, np.ones(w) / w, mode="same"))
    direct = 20 * np.log10(direct / direct.max() + 1e-9)
    assert np.max(np.abs(np.maximum(envelope_db(y, 25), -40) - np.maximum(direct, -40))) < 1e-4
    seg = y[int(0.15 * SR):int(0.6 * SR)] * np.hanning(int(0.6 * SR) - int(0.15 * SR))
    p = np.abs(np.fft.rfft(seg, 1 << 16)) ** 2
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    lv = [p[(f > h * 33.3 * 0.97) & (f < h * 33.3 * 1.03)].max() for h in range(1, 13)]
    masks = np.maximum(10 * np.log10(np.array(lv[1:]) / lv[0] + 1e-12), -60)
    assert np.allclose(harmonics(y, 33.3), masks)


def test_a_fit_is_read_back_while_nothing_it_depends_on_changed(tmp_path, monkeypatch):
    """A cached fit is the one the search would find again (it is seeded);
    another template, f1 or setting, or no cache folder, searches."""
    from refdissect import synth808

    searches = []
    search = synth808.differential_evolution
    monkeypatch.setattr(synth808, "differential_evolution", lambda *a, **k: searches.append(1) or search(*a, **k))
    reference = laid_out(TRUE, int(0.6 * SR))
    p, info = fit(reference, 41.2, maxiter=3, popsize=4, cache=tmp_path)
    again, kept = fit(reference, 41.2, maxiter=3, popsize=4, cache=tmp_path)
    assert again == p and kept["loss"] == info["loss"] and "cached" not in info
    assert len(searches) == 1 and [f.name for f in tmp_path.glob("fit-808-*.json")] == [kept["cached"]]
    fit(reference, 41.2, maxiter=4, popsize=4, cache=tmp_path)
    fit(reference, 41.3, maxiter=3, popsize=4, cache=tmp_path)
    fit(reference * 0.5, 41.2, maxiter=3, popsize=4, cache=tmp_path)
    fit(reference, 41.2, maxiter=3, popsize=4)
    assert len(searches) == 5
    assert fit(reference, 41.2, maxiter=3, popsize=4, cache=tmp_path)[0] == p


def test_the_fit_designs_its_filter_window_and_times_once(monkeypatch):
    """Each loss reads the 808 band, the harmonics' window and the voice's
    sample times made once (butter alone was 0.13 ms of a 2.6 ms
    candidate), with the same numbers as before."""
    from refdissect import audio, describe, synth808

    made = []
    butter, hanning = audio.butter, np.hanning
    monkeypatch.setattr(audio, "butter", lambda *a, **k: made.append("butter") or butter(*a, **k))
    monkeypatch.setattr(describe.np, "hanning", lambda n: made.append("hanning") or hanning(n))
    audio._design.cache_clear(), describe._hann.cache_clear(), synth808._times.cache_clear()
    reference = laid_out(TRUE, int(0.6 * SR))
    fit(reference, 41.2, maxiter=2, popsize=4)
    assert made.count("butter") == 1 and made.count("hanning") == 1 and synth808._times.cache_info().misses <= 2
    x = np.random.default_rng(0).standard_normal(4410).astype(np.float32)
    assert np.array_equal(audio.band(x, "808"), audio.sosfiltfilt(butter(4, 1500.0, fs=SR, output="sos"), x)
                          .astype(np.float32))


HELD = Params(f_end_hz=32.70, f_start_hz=32.70 * 2 ** (5 / 12), glide_ms=25, decay_s=1.5, attack_ms=3, drive=2.0,
              asym=0.05, phase=0.3)


def _held(seconds: float = 1.5, tail: float = 1.5):
    """HELD's drop ending in a slow tail (`tail` semitones more, falling
    with a 150 ms time constant: the suite's 808s hold about 100 cents over
    their settled pitch until 150-200 ms), its settled pitch and pitch curve
    (scenarios.tuning), and the curve as bass808.json keeps a template's
    (every 10 ms, to a tenth of a cent)."""
    from refdissect.scenarios import tuning
    x = voice(HELD, seconds)
    t = np.arange(len(x)) / SR
    x = np.interp(np.cumsum(2 ** (tail * np.exp(-t / 0.15) / 12)), np.arange(len(x)), x).astype(np.float32)
    f0, curve = tuning(x)
    return x, f0, curve, [[s, round(c, 1)] for s, c in curve[::2]]


def test_the_crossings_a_pitch_curve_was_read_off_are_found_back():
    from refdissect.scenarios import crossings
    from refdissect.synth808 import periods
    x, f0, _, kept = _held()
    z, rms = periods(kept, f0)
    assert rms < 0.1
    assert np.max(np.abs(z - crossings(x, f0)[1:len(z) + 1])) < 1e-4


def test_a_one_shot_on_the_template_s_periods_sounds_as_it_at_any_length():
    """bass_low-1's failure in the plan: the fitted drop (one exponential)
    has settled where the 808 still holds its tail, so notes of 0.11, 0.2
    and 0.4 s, read as the truth reads a note, sound 86, 65 and 35 cents
    under the template's, and no one Detune fits them all. On the
    template's own periods each sounds within a cent of it, read off the
    one-shot as estimate_from_plan reads it."""
    from refdissect.scenarios import drop, sounding, tuning
    from refdissect.synth808 import tracked_shot
    _, f0, curve, kept = _held()
    y, info = tracked_shot(HELD, f0, kept, f0)
    assert info["read_as_template"] and info["settled"] and info["followed"] < info["periods"]
    assert abs(20 * np.log10(np.max(np.abs(y))) + 1.0) < 0.05 and abs(y[-1]) < 1e-3

    def under(shot):  # cents each note length sounds under the template's (the settled pitches lined up)
        g0, c = tuning(shot)
        return np.array([100 * sounding(lambda q: drop(curve, q) / 100, d) - 1200 * np.log2(g0 / f0)
                         - 100 * sounding(lambda q: drop(c, q) / 100, d) for d in (0.11, 0.2, 0.4)])
    assert np.all(np.abs(under(y)) < 1.0)
    plain = under(one_shot(HELD, f0))
    assert plain.min() > 30 and plain.max() - plain.min() > 40


def test_fit_808_writes_its_one_shot_on_the_template_s_periods(tmp_path):
    """fit_808 reads bass808.json's template curve: its one-shot follows the
    template's periods (synth808.json pitch_track) and reads as the curve,
    within SETTLED_CENTS where it eases to the settled pitch."""
    import json

    import soundfile as sf

    from refdissect import pipeline
    from refdissect.scenarios import drop, tuning
    from refdissect.synth808 import SETTLED_CENTS
    x, f0, curve, kept = _held(0.9)
    onsets = [0.5, 1.5, 2.5, 3.5]
    bass = np.zeros(int(4.5 * SR), np.float32)
    for s in onsets:
        bass[int(s * SR):int(s * SR) + len(x)] += x
    stems = tmp_path / "stems"
    stems.mkdir()
    sf.write(stems / "bass.wav", bass, SR)
    sf.write(stems / "drums.wav", np.zeros_like(bass), SR)
    work = tmp_path / "work"
    (work / "templates").mkdir(parents=True)
    (work / "analysis.json").write_text(json.dumps({"source": {"stems": str(stems)}, "tuning": {"808": 0.0},
                                                    "notes_808": []}))
    (work / "bass808.json").write_text(json.dumps({"template": {
        "onsets_s": onsets, "root_midi": 24, "hz": f0, "method": "qifft_300_600ms", "length_s": 0.6, "curve": kept}}))
    out = pipeline.fit_808(work, log=lambda *a: None, maxiter=2, popsize=4)
    assert out["pitch_track"]["read_as_template"] and out["pitch_track"]["periods"] >= 20
    g0, c = tuning(sf.read(out["file"], dtype="float32")[0])
    t = np.arange(0.06, 0.45, 0.005)
    assert np.max(np.abs(drop(c, t) + 1200 * np.log2(g0 / f0) - drop(curve, t))) < SETTLED_CENTS


def test_the_one_shot_leaves_the_template_s_periods_once_its_drop_has_settled():
    """A template's curve past its settled point holds its ripple and, near
    its end, the next note coming in (-100 cents over its last 100 ms
    here): the one-shot follows the periods to the first one past the point
    from which the curve stays within 5 cents for 100 ms. A curve that never
    holds that long is followed until it first reads 5 cents under its
    settled pitch after coming that close."""
    from refdissect.synth808 import _settled_at
    t = np.arange(0.0, 0.8, 0.01)
    z = 0.03 * np.arange(1, 27)
    settles = [[s, 300 * np.exp(-s / 0.05)] for s in t]
    dips = [[s, c if s < 0.7 else -100.0] for s, c in settles]
    never = [[s, c if s < 0.25 else -30.0] for s, c in settles]
    assert 0.2 < z[_settled_at(z, settles)] < 0.27 and _settled_at(z, dips) == _settled_at(z, settles)
    assert z[_settled_at(z, never)] < 0.25 <= z[_settled_at(z, never) + 1]
    assert _settled_at(z, [[s, 300 * np.exp(-s / 0.5)] for s in t]) == len(z) - 1


def test_the_rise_is_found_where_the_curve_reads_100_cents_over_its_lowest_point_since():
    """One 808's templates (htdemucs) read -228 cents at 50 ms and +228 at
    80 ms, where a kick plays; its truth falls from 995 to 67 cents over
    0-80 ms. The rise ends at the first crossing past which the curve never
    reads 100 cents over its lowest point since; a drop that only falls has
    none."""
    from refdissect.synth808 import _falls_from
    t = np.arange(0.0, 0.5, 0.01)
    z = 0.015 + 0.03 * np.arange(16)
    falls = [[s, 1200 * np.exp(-s / 0.04)] for s in t]
    dips = [[s, -228.0 if round(s, 2) == 0.05 else 228.0 if round(s, 2) == 0.08 else c] for s, c in falls]
    assert _falls_from(z, falls, 15) == 0
    assert _falls_from(z, dips, 15) == 2


def test_the_periods_across_a_rise_are_pooled_into_ones_that_never_shorten():
    """Pool adjacent violators: each run of periods that shortens becomes
    its mean, so the first and last crossings, and the cycles between,
    stay where they were."""
    from refdissect.synth808 import _pooled
    z = np.cumsum([0.03, 0.0175, 0.030, 0.0278, 0.0287, 0.0294, 0.0300])
    got = _pooled(z, 4)
    assert np.all(np.diff(np.diff(got[:5])) >= -1e-12)
    assert got[0] == z[0] and got[4] == pytest.approx(z[4]) and np.array_equal(got[5:], z[5:])
    assert np.diff(got)[1:4] == pytest.approx([0.028833] * 3, abs=1e-6)
    assert np.array_equal(_pooled(z[:3], 2), z[:3])


def test_the_one_shot_only_falls_where_its_template_rises():
    """A kick in the template: its curve reads 228 cents under the drop at
    60-80 ms and 228 over it at 90-110 ms. Followed, every note of the
    one-shot swooped 456 cents up; with the fitted drop played up to the
    rise instead (081b962), it left the template's periods before it and
    read up to 577 cents apart. Pooled, the periods never shorten across
    the rise, so the one-shot's pitch never rises there, and the crossings
    before and after it stay the template's."""
    from refdissect.scenarios import crossings, drop, tuning
    from refdissect.synth808 import periods, tracked_shot
    _, f0, curve, kept = _held()
    dipped = [[s, c - 228.0 if 0.059 < s < 0.081 else c + 228.0 if 0.089 < s < 0.111 else c] for s, c in kept]
    y, info = tracked_shot(HELD, f0, dipped, f0)
    assert info["pooled"] > 0 and info["read_as_template"]
    g0, c = tuning(y)
    tt = np.arange(0.05, info["until_s"], 0.001)
    heard = drop(c, tt) + 1200 * np.log2(g0 / f0)
    assert np.max(heard - np.minimum.accumulate(heard)) < 15
    z = periods(dipped, f0)[0]
    read = crossings(y, g0)[1:]
    near = [read[np.argmin(np.abs(read - q))] - q for q in (z[0], z[info["pooled"] + 1])]
    assert np.max(np.abs(near)) < 1e-4


def _late(x: np.ndarray, ms: float) -> np.ndarray:
    """x at 15 % of its level until ms, then up to it over 2 ms: a template
    that attacks ms after its onset (bass_low-1's: 7 ms)."""
    t = np.arange(len(x)) / SR
    w = np.clip((t - ms / 1000) / 0.002 + 1, 0.0, 1.0)
    return (x * (0.15 + 0.85 * w * w * (3 - 2 * w))).astype(np.float32)


def test_the_one_shot_attacks_where_its_template_does_after_its_onset(tmp_path):
    """bass_low-1's template attacks 7 ms after its onset, its fitted voice
    (a 0.3 ms attack) at once: plan_808, which puts each note's attack on
    the analysis' t_attack, started the one-shot 7 ms late, its drop with
    it, and the notes under 120 ms sounded 38 cents over the analysis'.
    Laid out as the template is (a fast voice quiet until the template's
    attack, a slow one with its head left out), the one-shot attacks where
    the template does: plan_808 starts each note at its onset, and each
    length sounds as the analysis heard it."""
    import soundfile as sf

    from refdissect.audio import attack_time
    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import plan_808
    from refdissect.scenarios import drop, sounding, tuning
    from refdissect.synth808 import tracked_shot
    x = _late(_held()[0], 8.0)
    f0, curve = tuning(x)
    kept = [[s, round(c, 1)] for s, c in curve[::2]]
    onset_attack = attack_time(x, "808")
    own = 69 + 12 * np.log2(f0 / 440)
    lengths = (0.07, 0.11, 0.2, 0.4)
    notes = [{"t": 0.25 + 0.5 * i, "t_attack": 0.25 + 0.5 * i + onset_attack, "end": 0.25 + 0.5 * i + d,
              "pitch": 24, "peak": 1.0, "midi_settled": own,
              "midi_float": own + sounding(lambda q: drop(curve, q) / 100, d)} for i, d in enumerate(lengths)]
    for k, attack_ms in enumerate((0.3, 90.0)):
        y, info = tracked_shot(Params(32.70, 32.70 * 2 ** (6 / 12), 20, 1.5, attack_ms, 4.0, 0.05, 1.9), f0, kept, f0,
                               attack_s=onset_attack)
        assert abs(attack_time(y, "808") - onset_attack) < 0.0005
        shot = tmp_path / f"808 {k}.wav"
        sf.write(shot, y, SR, subtype="FLOAT")
        synth = {"root_midi": 24, "pitch_hz": f0, "file": str(shot), "file_name": shot.name}
        track = plan_808({"tuning": {"808": own - 24}, "notes_808": notes}, synth, tmp_path / "stems", 2.0)
        assert [p["start_time"] / 2.0 for p in track["notes"]] == pytest.approx([n["t"] for n in notes], abs=0.0005)
        plan = {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": {}, "notes": []}, "808": track},
                "levels": {}}
        heard = estimate_from_plan(plan)["bass808"]
        assert [h["midi"] for h in heard] == pytest.approx([n["midi_float"] for n in notes], abs=0.015)


def test_a_drop_done_by_its_second_crossing_keeps_the_fitted_one_shot():
    """An 808 whose drop is over by its second crossing (a 3.5 ms glide) has
    a flat curve: periods() put its first crossing where the curve first
    moves (0.93 s in; the template's own second crossing is at 45 ms), and
    the one-shot played the scaled fitted drop until there, 10 cents off at
    every length. One_shot's voice sounds within 2 cents."""
    from refdissect.scenarios import drop, sounding, tuning
    from refdissect.synth808 import tracked_shot
    P = Params(f_end_hz=39.95, f_start_hz=85.36, glide_ms=3.54, decay_s=1.19, attack_ms=1.38, drive=1.31, asym=0.033,
               phase=0.3)
    x = voice(P, 1.0)
    f0, curve = tuning(x)
    kept = [[s, round(c, 1)] for s, c in curve[::2]]
    fitted = Params(39.95, 85.36 * 1.04, 3.54 * 0.95, 1.19, 1.38, 1.31, 0.033, 2.0)
    assert tracked_shot(fitted, f0, kept, f0) is None
    g0, c = tuning(one_shot(fitted, f0))
    off = [100 * sounding(lambda q: drop(c, q) / 100, d) + 1200 * np.log2(g0 / f0)
           - 100 * sounding(lambda q: drop(curve, q) / 100, d) for d in (0.11, 0.2, 0.4)]
    assert np.max(np.abs(off)) < 2.0


def test_the_one_shot_eases_on_from_where_the_reader_reads_its_last_period():
    """Past the last period followed, the reader's crossings keep landing
    on the template's: the crossings made past it move with the last one,
    so the ease starts from the slope the reader reads there (a strong
    drive and asymmetry move the crossings 0.3-0.7 ms; left where they
    were, the first periods past it were 0.15-0.65 ms off, a blip of up to
    12 cents on the suite's one-shots)."""
    from refdissect.scenarios import crossings, tuning
    from refdissect.synth808 import tracked_shot
    P = Params(f_end_hz=45.0, f_start_hz=90.0, glide_ms=20, decay_s=1.5, attack_ms=3, drive=5.9, asym=-0.39, phase=0.3)
    x = voice(P, 1.5)
    t = np.arange(len(x)) / SR
    x = np.interp(np.cumsum(2 ** (1.5 * np.exp(-t / 0.15) / 12)), np.arange(len(x)), x).astype(np.float32)
    f0, curve = tuning(x)
    kept = [[s, round(c, 1)] for s, c in curve[::2]]
    y, info = tracked_shot(Params(45.0, 90.0 * 1.05, 18.0, 1.5, 3, 5.9, -0.39, 2.0), f0, kept, f0)
    own, read = crossings(x, f0), crossings(y, tuning(y)[0])
    after = own[own > info["until_s"] + 1e-4][:3]
    assert len(after) == 3 and max(abs(read[np.argmin(np.abs(read - q))] - q) for q in after) < 1e-4


def test_a_curve_that_falls_through_the_one_shot_s_settled_pitch_is_left_there():
    """One htdemucs_ft template's settled pitch was read on its unsettled
    tail: its curve falls on through it, -260 cents by its end, and the
    fit's settled pitch, which the one-shot takes, sits 49 cents under it.
    Followed to the curve's end, the one-shot dipped 147 cents under its
    own settled pitch and rose back; it is left before the curve passes
    that pitch (between two of its points), so it never reads under it
    (the reader's cubic overshoots 13 cents where the ease begins)."""
    from refdissect.scenarios import drop, tuning
    from refdissect.synth808 import tracked_shot
    hz = 23.43
    t = np.round(np.arange(0.0, 0.211, 0.01), 2)
    falls = np.where(t <= 0.04, 958.0, -260.0 + 1218.0 * np.exp(-((t - 0.04) / 0.09) ** 2))
    curve = [[float(a), float(round(b, 1))] for a, b in zip(t, falls)]
    pitch = hz * 2 ** (-49.1 / 1200)
    y, info = tracked_shot(Params(pitch, 90.0, 60.0, 1.2, 0.3, 3.0, 0.05, 1.0), pitch, curve, hz)
    g0, c = tuning(y)
    heard = drop(c, np.arange(0.05, 0.6, 0.005)) + 1200 * np.log2(g0 / pitch)
    assert info["followed"] < info["periods"] and heard.min() > -5
    assert np.max(heard - np.minimum.accumulate(heard)) < 20


def test_attack_cycles_that_cannot_reach_the_first_period_are_none():
    """A first period more than 2.4 times the next gives the PCHIP no
    slope at its start: no cycle reaches z[0] (the scale was infinite and
    the voice NaN; the rounds kept the best of what was left)."""
    from refdissect.synth808 import _cycles, _times
    z = np.array([0.03, 0.10, 0.125, 0.15, 0.175, 0.2])
    assert _cycles(HELD, _times(int(0.5 * SR)), 32.7, z, 4, None) is None
