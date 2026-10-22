"""render_plan and estimate_from_plan on synthetic samples: what Live would
play, offline, where the answer is known."""

import json

import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import SR, attack_time, samples
from refdissect.evaluate import SCHEMA, estimate_from_plan, score
from refdissect.render import render_plan, velocity_db


def wav(folder, name, x):
    path = folder / name
    sf.write(path, np.asarray(x, np.float32), SR, subtype="FLOAT")
    return str(path)


def noise(seconds, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, int(seconds * SR)).astype(np.float32)


def note(pitch, beat, velocity=127, duration=0.25):
    return {"pitch": pitch, "start_time": beat, "duration": duration, "velocity": velocity}


def plan(pads, notes, t808=None):
    """120 BPM: a beat is 0.5 s. Faders at 0 dB."""
    return {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": pads, "notes": notes},
                                                            "808": t808 or {"notes": []}},
            "levels": {"headroom_db": 4.0, "drums_track_db": 0.0, "808_track_db": 0.0}}


def rendered(folder, manifest, key):
    return sf.read(folder / manifest["tracks"][key], dtype="float32")[0]


def test_a_sample_with_10_ms_of_lead_in_at_1_s_attacks_at_1_010_s(kit, tmp_path):
    path = wav(tmp_path, "snare.wav", np.r_[np.zeros(samples(10), np.float32), kit["snare"]])
    p = plan({"38": {"path": path, "transpose": 0, "role": "snare"}}, [note(38, 2.0)])
    m = render_plan(p, tmp_path / "out")
    x = rendered(tmp_path / "out", m, "38")
    assert 0.9 + attack_time(x[int(0.9 * SR):], "snare") == pytest.approx(1.010, abs=0.001)
    hit = estimate_from_plan(p)["drums"][0]
    assert hit["t"] == 1.0 and hit["t_attack"] == pytest.approx(1.010, abs=0.001)
    # start_s skips the lead-in, as Simpler's start marker would
    p["tracks"]["drums"]["pads"]["38"]["start_s"] = 0.010
    assert estimate_from_plan(p)["drums"][0]["t_attack"] == pytest.approx(1.000, abs=0.001)


def test_transpose_up_an_octave_halves_the_length(tmp_path):
    path = wav(tmp_path, "perc.wav", noise(0.2))
    for semitones, seconds in ((0, 0.2), (12, 0.1), (-12, 0.4)):
        p = plan({"43": {"path": path, "transpose": semitones, "role": "perc"}}, [note(43, 0.0)])
        x = rendered(tmp_path / str(semitones), render_plan(p, tmp_path / str(semitones)), "43")
        on = np.flatnonzero(np.abs(x) > 1e-4)
        assert (on[-1] - on[0]) / SR == pytest.approx(seconds, abs=0.002)


def test_a_choke_group_cuts_the_open_hat(tmp_path):
    """Open hat at 0 s, closed hat at 0.5 s in one choke group: the open hat
    stops 5 ms after 0.5 s. Without the group, or on mono=False retriggers,
    it rings on."""
    pads = {"46": {"path": wav(tmp_path, "oh.wav", noise(1.0)), "transpose": 0, "role": "hat", "choke_group": 1},
            "42": {"path": wav(tmp_path, "ch.wav", noise(0.05, 1)), "transpose": 0, "role": "hat",
                   "choke_group": 1}}
    m = render_plan(plan(pads, [note(46, 0.0), note(42, 1.0)]), tmp_path / "choke")
    open_hat = rendered(tmp_path / "choke", m, "46")
    assert np.abs(open_hat[: int(0.5 * SR)]).max() > 0.4
    assert np.abs(open_hat[int(0.5 * SR) + samples(5) + 1:]).max() == 0.0
    pads["46"]["choke_group"] = None
    free = rendered(tmp_path / "free", render_plan(plan(pads, [note(46, 0.0), note(42, 1.0)]), tmp_path / "free"),
                    "46")
    assert np.abs(free[int(0.8 * SR):int(0.9 * SR)]).max() > 0.4
    # one pad hit again at 0.5 s: mono leaves only the second hit after the fade, the default sums both
    # (velocities at unity, "flat", so the samples compare as they are)
    twice = [note(46, 0.0), note(46, 1.0)]
    mono = rendered(tmp_path / "mono", render_plan(plan(pads, twice), tmp_path / "mono", "flat", mono=True), "46")
    poly = rendered(tmp_path / "poly", render_plan(plan(pads, twice), tmp_path / "poly", "flat"), "46")
    hit, x = int(0.5 * SR), noise(1.0)
    after = slice(hit + samples(5) + 1, SR)
    second = x[after.start - hit:SR - hit]
    assert np.allclose(mono[after], second) and np.allclose(poly[after], second + x[after])


def test_velocity_law_faders_and_the_drums_sum(kit, tmp_path):
    """A hit plays at the track fader x the pad's Volume x its velocity's
    level under Live 12.4.6's law at the pad's "Vol < Vel" (0.35 unless the
    pad says): velocity 127 7.88 dB over the Volume, 64 0.94 dB under it,
    and 127 10.16 dB over it at 0.45, as the battery measured (7.8844,
    -0.9356 and 10.1657 dB). v/127 and a flat law on request."""
    assert (velocity_db(127), velocity_db(64), velocity_db(127, 0.45)) == pytest.approx(
        (7.8844, -0.9356, 10.1657), abs=0.002)
    pads = {"36": {"path": wav(tmp_path, "kick.wav", kit["kick"]), "transpose": 0, "role": "kick"},
            "38": {"path": wav(tmp_path, "snare.wav", kit["snare"]), "transpose": 0, "role": "snare",
                   "volume_db": -6.0}}
    p = plan(pads, [note(36, 0.0, 127), note(36, 2.0, 64), note(38, 1.0, 127)])
    p["levels"]["drums_track_db"] = -3.0
    m = render_plan(p, tmp_path / "v")
    kick, snare = rendered(tmp_path / "v", m, "36"), rendered(tmp_path / "v", m, "38")
    g, loud, soft = 10 ** (-3 / 20), 10 ** (velocity_db(127) / 20), 10 ** (velocity_db(64) / 20)
    assert np.abs(kick[: SR // 2]).max() == pytest.approx(g * loud, rel=1e-4)
    assert np.abs(kick[SR: SR + SR // 2]).max() == pytest.approx(g * soft, rel=1e-4)
    assert np.abs(snare).max() == pytest.approx(g * 10 ** (-6 / 20) * loud, rel=1e-4)
    drums = sf.read(tmp_path / "v" / m["drums"], dtype="float32")[0]
    assert np.allclose(drums, kick + snare, atol=1e-6)
    gains = [e["gain"] for e in estimate_from_plan(p)["drums"]]
    assert gains == pytest.approx([g * loud, g * 10 ** (-6 / 20) * loud, g * soft], rel=1e-4)
    linear = render_plan(p, tmp_path / "linear", velocity_curve="linear")
    assert np.abs(rendered(tmp_path / "linear", linear, "36")[SR:]).max() == pytest.approx(g * 64 / 127, rel=1e-4)
    flat = render_plan(p, tmp_path / "flat", velocity_curve="flat")
    assert np.abs(rendered(tmp_path / "flat", flat, "36")[SR:]).max() == pytest.approx(g, rel=1e-4)
    pads["36"]["vol_vel"] = 0.45  # a pad loaded through the browser, as Live 12.4.6 leaves it
    browser = rendered(tmp_path / "45", render_plan(p, tmp_path / "45"), "36")
    assert np.abs(browser[: SR // 2]).max() == pytest.approx(g * 10 ** (velocity_db(127, 0.45) / 20), rel=1e-4)


def test_the_808_plays_each_note_at_its_pitch_and_stops_at_its_end(tmp_path):
    """A one-shot at C1 (MIDI 24) with the build's Transpose +36: MIDI 36
    sounds an octave up, cut 5 ms after its 0.25 s note."""
    t = np.arange(2 * SR) / SR
    shot = wav(tmp_path, "808.wav", 0.8 * np.sin(2 * np.pi * 32.703 * t))
    t808 = {"sample_path": shot, "root_midi": 24, "transpose": 36,
            "notes": [note(36, 0.0, duration=2.0), note(31, 4.0, duration=0.5)]}
    m = render_plan(plan({}, [], t808), tmp_path / "b")
    x = rendered(tmp_path / "b", m, "808")
    seg = x[int(0.1 * SR):int(0.9 * SR)] * np.hanning(int(0.8 * SR))
    spectrum = np.abs(np.fft.rfft(seg, 1 << 19))
    assert np.argmax(spectrum) * SR / (1 << 19) == pytest.approx(65.406, abs=0.1)
    assert np.abs(x[int(1.0 * SR) + samples(5) + 1:int(2.0 * SR)]).max() == 0.0
    assert np.abs(x[int(2.25 * SR) + samples(5) + 1:]).max() == 0.0
    notes = estimate_from_plan(plan({}, [], t808))["bass808"]
    assert [n["midi"] for n in notes] == [36.0, 31.0] and notes[0]["end"] == 1.0
    t808["cents"] = 30.0  # the one-shot's own tuning over root_midi
    assert estimate_from_plan(plan({}, [], t808))["bass808"][0]["midi"] == pytest.approx(36.3)


def test_levels_from_the_render(kit, tmp_path):
    """The truth stem plays the snare at full level; the rebuild plays it at
    velocity 64, which Live 12.4.6 plays 0.94 dB under the pad's Volume at
    "Vol < Vel" 35 %: -0.94 dB on every hit, and 0 dB of spread."""
    times = [0.5, 1.5, 2.5, 3.5]
    stem = np.zeros(4 * SR, np.float32)
    for t in times:
        stem[int(t * SR):int(t * SR) + len(kit["snare"])] += kit["snare"]
    (tmp_path / "truth").mkdir()
    wav(tmp_path / "truth", "snare.wav", stem)
    lead = attack_time(kit["snare"], "snare")
    truth = {"schema": SCHEMA, "kind": "truth", "scenario": "t", "seed": 0, "sounds": {"snare": {"role": "snare"}},
             "audio": {"sr": SR, "n_samples": len(stem), "stems": {"snare": "snare.wav"}},
             "drums": [{"sound": "snare", "role": "snare", "t": t, "t_attack": t + lead, "gain": 1.0} for t in times]}
    p = plan({"38": {"path": wav(tmp_path, "s.wav", kit["snare"]), "transpose": 0, "role": "snare"}},
             [note(38, 2 * t, 64) for t in times])
    render_plan(p, tmp_path / "r", n_samples=len(stem))
    s = score(truth, estimate_from_plan(p), tmp_path / "r", truth_dir=tmp_path / "truth")
    level = s["levels"]["sounds"]["snare"]
    assert level["level_db"] == pytest.approx(velocity_db(64), abs=0.01) and level["spread_p90_db"] < 0.01
    assert s["drums"]["sounds"]["snare"]["f1_20"] == 1.0 and level["n"] == 4
    json.dumps(s)
    # a stem written halved, as a scenario writes one that would clip, reads the same (+6.02 dB without stems_gain)
    wav(tmp_path / "truth", "snare.wav", 0.5 * stem)
    truth["audio"]["master"] = {"stems_gain": {"snare": 0.5}}
    halved = score(truth, estimate_from_plan(p), tmp_path / "r", truth_dir=tmp_path / "truth")
    assert halved["levels"]["sounds"]["snare"]["level_db"] == pytest.approx(level["level_db"], abs=0.01)


def test_a_kit_at_the_truths_levels_reads_where_the_faders_put_it(kit, tmp_path):
    """The plan's faders match each track to a separated stem, here at half
    the truth's level (Demucs separates the clipped mix): a perfect kit then
    reads -6.02 dB on every sound (kit_db), so the snare played at velocity
    64, taken here as a linear level (v/127), is -5.94 dB re the truth and
    +0.08 dB re a perfect kit (vs_kit_db)."""
    times = [0.5, 1.5, 2.5, 3.5]
    stem = np.zeros(4 * SR, np.float32)
    for t in times:
        stem[int(t * SR):int(t * SR) + len(kit["snare"])] += kit["snare"]
    (tmp_path / "truth").mkdir()
    wav(tmp_path / "truth", "snare.wav", stem)
    wav(tmp_path / "truth", "drums.wav", stem)  # the drums group: the snare alone
    lead = attack_time(kit["snare"], "snare")
    truth = {"schema": SCHEMA, "kind": "truth", "scenario": "t", "seed": 0, "sounds": {"snare": {"role": "snare"}},
             "audio": {"sr": SR, "n_samples": len(stem), "stems": {"snare": "snare.wav", "drums": "drums.wav"}},
             "drums": [{"sound": "snare", "role": "snare", "t": t, "t_attack": t + lead, "gain": 1.0} for t in times]}
    p = plan({"38": {"path": wav(tmp_path, "s.wav", kit["snare"]), "transpose": 0, "role": "snare"}},
             [note(38, 2 * t, 64) for t in times])
    render_plan(p, tmp_path / "r", n_samples=len(stem), velocity_curve="linear")
    reference = {"drums": wav(tmp_path, "separated drums.wav", 0.5 * stem), "bass": None}
    lv = score(truth, estimate_from_plan(p), tmp_path / "r", tmp_path / "truth", reference=reference)["levels"]
    assert list(lv["kit"]) == ["drums"] and lv["kit"]["drums"]["kit_db"] == pytest.approx(-6.02, abs=0.01)
    snare = lv["sounds"]["snare"]
    assert snare["level_db"] == pytest.approx(-5.94, abs=0.01)
    assert snare["vs_kit_db"] == pytest.approx(0.08, abs=0.01)


def test_a_planned_808_sounds_where_its_one_shot_is_tuned(tmp_path):
    """fit_808 writes its one-shot at the 808's own tuning (synth808.json
    pitch_hz), here 33.58 Hz, 45.8 cents over root_midi 24; this one starts
    an octave up and falls with a 60 ms time constant. make_plan carries the
    tuning as cents (before, a plan claimed root_midi: 45.8 cents flat), and
    estimate_from_plan's notes settle where the render settles (native_f0,
    the truth's rule) and sound where the render sounds (pitch_track through
    sounding): a 110 ms note 5 semitones up, planned at its own length (no
    longer floored to a quarter beat), sits 2.5-3.5 semitones over its
    settled pitch. A plan without cents is read at the one-shot's measured
    tuning."""
    from refdissect import scenarios as S
    from refdissect.live import make_plan
    from refdissect.synth808 import Params, one_shot
    hz = 33.58
    shot = wav(tmp_path, "808 (synth).wav", one_shot(Params(41.2, 82.4, 60, 1.2, 2, 1.5, 0.05, 0.3), hz))
    for stem in ("drums", "bass"):
        wav(tmp_path, f"{stem}.wav", noise(4.0) * 0.1)
    analysis = {"grid": {"bpm": 120.0}, "source": {"duration_s": 4.0}, "templates": {}, "events": {},
                "notes_808": [{"t": 0.0, "end": 1.5, "pitch": 24, "peak": 1.0},
                              {"t": 2.0, "end": 2.11, "pitch": 29, "peak": 1.0}], "sections": []}
    synth = {"root_midi": 24, "pitch_hz": hz, "file": shot, "file_name": "808 (synth).wav"}
    p = make_plan(analysis, {}, synth, tmp_path)
    assert p["tracks"]["808"]["cents"] == pytest.approx(45.8, abs=0.05)
    x = rendered(tmp_path / "r", render_plan(p, tmp_path / "r"), "808")
    long, short = estimate_from_plan(p)["bass808"]
    assert long["midi_settled"] == pytest.approx(24.458, abs=0.001)
    assert S.hz_to_midi(S.native_f0(x[:int(1.5 * SR)])) == pytest.approx(long["midi_settled"], abs=0.01)
    a, b = int(2.0 * SR), int((2.0 + short["end"] - short["t"]) * SR)
    track = S.pitch_track(x[a:b], hz * 2 ** (5 / 12))
    heard = S.sounding(lambda t: S.hz_to_midi(1.0) + 12 * np.log2(track[0](t)), (b - a) / SR)
    assert short["midi"] == pytest.approx(heard, abs=0.1) and 2.5 < short["midi"] - short["midi_settled"] < 3.5
    del p["tracks"]["808"]["cents"]
    assert estimate_from_plan(p)["bass808"][0]["midi_settled"] == pytest.approx(24.458, abs=0.01)


def test_missing_files_play_nothing(tmp_path):
    p = plan({"36": {"path": str(tmp_path / "gone.wav"), "transpose": 0, "role": "kick"}}, [note(36, 0.0)],
             {"sample_path": "gone 808.wav", "root_midi": 24, "transpose": 36, "notes": [note(24, 0.0)]})
    m = render_plan(p, tmp_path / "m", root=tmp_path)
    assert m["tracks"] == {} and len(m["missing"]) == 2
    est = estimate_from_plan(p, root=tmp_path)
    assert est["drums"] == [] and est["bass808"] == [] and len(est["generator"]["switches"]["missing"]) == 2
