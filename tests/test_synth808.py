import numpy as np

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
