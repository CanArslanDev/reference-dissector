"""render_plan and estimate_from_plan on synthetic samples: what Live would
play, offline, where the answer is known."""

import json

import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import SR, attack_time, samples
from refdissect.evaluate import SCHEMA, estimate_from_plan, score
from refdissect.render import render_plan


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
    twice = [note(46, 0.0), note(46, 1.0)]
    mono = rendered(tmp_path / "mono", render_plan(plan(pads, twice), tmp_path / "mono", mono=True), "46")
    poly = rendered(tmp_path / "poly", render_plan(plan(pads, twice), tmp_path / "poly"), "46")
    hit, x = int(0.5 * SR), noise(1.0)
    after = slice(hit + samples(5) + 1, SR)
    second = x[after.start - hit:SR - hit]
    assert np.allclose(mono[after], second) and np.allclose(poly[after], second + x[after])


def test_velocity_law_faders_and_the_drums_sum(kit, tmp_path):
    pads = {"36": {"path": wav(tmp_path, "kick.wav", kit["kick"]), "transpose": 0, "role": "kick"},
            "38": {"path": wav(tmp_path, "snare.wav", kit["snare"]), "transpose": 0, "role": "snare",
                   "volume_db": -6.0}}
    p = plan(pads, [note(36, 0.0, 127), note(36, 2.0, 64), note(38, 1.0, 127)])
    p["levels"]["drums_track_db"] = -3.0
    m = render_plan(p, tmp_path / "v")
    kick, snare = rendered(tmp_path / "v", m, "36"), rendered(tmp_path / "v", m, "38")
    g = 10 ** (-3 / 20)
    assert np.abs(kick[: SR // 2]).max() == pytest.approx(g, rel=1e-4)
    assert np.abs(kick[SR: SR + SR // 2]).max() == pytest.approx(g * 64 / 127, rel=1e-4)
    assert np.abs(snare).max() == pytest.approx(g * 10 ** (-6 / 20), rel=1e-4)
    drums = sf.read(tmp_path / "v" / m["drums"], dtype="float32")[0]
    assert np.allclose(drums, kick + snare, atol=1e-6)
    gains = [e["gain"] for e in estimate_from_plan(p)["drums"]]
    assert gains == pytest.approx([g, g * 10 ** (-6 / 20), g * 64 / 127], rel=1e-4)
    flat = render_plan(p, tmp_path / "flat", velocity_curve="flat")
    assert np.abs(rendered(tmp_path / "flat", flat, "36")[SR:]).max() == pytest.approx(g, rel=1e-4)


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
    velocity 64: -6.0 dB on every hit, and 0 dB of spread."""
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
    assert level["level_db"] == pytest.approx(20 * np.log10(64 / 127), abs=0.01) and level["spread_p90_db"] < 0.01
    assert s["drums"]["sounds"]["snare"]["f1_20"] == 1.0 and level["n"] == 4
    json.dumps(s)
    # a stem written halved, as a scenario writes one that would clip, reads the same (+6.02 dB without stems_gain)
    wav(tmp_path / "truth", "snare.wav", 0.5 * stem)
    truth["audio"]["master"] = {"stems_gain": {"snare": 0.5}}
    halved = score(truth, estimate_from_plan(p), tmp_path / "r", truth_dir=tmp_path / "truth")
    assert halved["levels"]["sounds"]["snare"]["level_db"] == pytest.approx(level["level_db"], abs=0.01)


def test_missing_files_play_nothing(tmp_path):
    p = plan({"36": {"path": str(tmp_path / "gone.wav"), "transpose": 0, "role": "kick"}}, [note(36, 0.0)],
             {"sample_path": "gone 808.wav", "root_midi": 24, "transpose": 36, "notes": [note(24, 0.0)]})
    m = render_plan(p, tmp_path / "m", root=tmp_path)
    assert m["tracks"] == {} and len(m["missing"]) == 2
    est = estimate_from_plan(p, root=tmp_path)
    assert est["drums"] == [] and est["bass808"] == [] and len(est["generator"]["switches"]["missing"]) == 2
