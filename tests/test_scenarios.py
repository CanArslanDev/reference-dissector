"""Calibration v2 on a library of synthetic one-shots: no sample library needed."""

import hashlib
import json
import math
import shutil

import mido
import numpy as np
import pytest
import soundfile as sf
from scipy.signal import butter, sosfilt

from refdissect import evaluate
from refdissect import scenarios as S
from refdissect.audio import SR, read
from refdissect.calibrate import build_beat
from refdissect.library import inventory
from refdissect.synth808 import Params, voice
from refdissect.voices import CHORDS

F808 = 41.2  # the synthetic 808's settled pitch (Hz)
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the mp3 step needs ffmpeg")


def _burst(seed: int, ms: float, lo: float, hi: float, decay_ms: float, tone_hz: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(ms * SR / 1000)
    t = np.arange(n) / SR
    x = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), rng.standard_normal(n))
    if tone_hz:
        x = x * 0.3 + np.sin(2 * np.pi * tone_hz * t) * np.max(np.abs(x))
    x *= np.exp(-t / (decay_ms / 1000))
    return (x / np.max(np.abs(x))).astype(np.float32)


@pytest.fixture(scope="module")
def rows(tmp_path_factory):
    folder = tmp_path_factory.mktemp("library")
    t = np.arange(9 * SR) / SR
    clap = sum(np.roll(np.pad(_burst(20 + k, 120, 600, 6000, 25), (0, 2000)), k * 400) for k in range(3))
    files = {
        "kicks/kick a.wav": _burst(1, 300, 30, 2000, 120, tone_hz=55.0),
        "kicks/kick b.wav": _burst(11, 350, 30, 1500, 150, tone_hz=48.0),
        "snares/snare a.wav": _burst(2, 250, 200, 9000, 60),
        "snares/snare b.wav": _burst(12, 200, 300, 8000, 50),
        "claps/clap a.wav": clap / np.max(np.abs(clap)),
        "hats/hat a.wav": _burst(3, 80, 6000, 16000, 15),
        "open hats/open hat a.wav": _burst(4, 600, 5000, 16000, 200),
        # a short pitch drop (46 -> 41.2 Hz, 15 ms): settled within 0.1 s even an octave down
        "808s/808 a.wav": voice(Params(F808, 46.0, 15, 1.2, 2, 1.5, 0.05, 0.3), 1.6),
        "percs/perc a.wav": _burst(5, 150, 800, 5000, 40),
        "percs/tom a.wav": _burst(6, 400, 60, 1200, 120, tone_hz=110.0),
        "crashes/crash a.wav": _burst(8, 2000, 3000, 16000, 700),
        "rims/rim a.wav": _burst(9, 60, 1000, 8000, 10),
        "loops/keys loop.wav": 0.3 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277.2 * t))[: int(8.5 * SR)],
        "vocals/vocal take.wav": np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 0.5,
    }
    for rel, x in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        sf.write(folder / rel, x, SR, subtype="PCM_24")
    return inventory([folder])


@pytest.fixture(scope="module")
def built(rows, tmp_path_factory):
    """Every scenario but producer, 4 bars, seed 1."""
    out = tmp_path_factory.mktemp("suite")
    return {name: (S.build(rows, name, 1, out / name, bars=4), out / name) for name in S.SCENARIOS
            if name != "producer"}


def _hashes(folder) -> dict:
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def test_every_scenario_is_valid_and_matches_its_stems(built):
    for name, (ev, folder) in built.items():
        assert S.check_schema(ev) == [], name
        v = S.verify(folder)
        assert v["hits"] > 0 and v["attack_err_max_ms"] <= 2.0, (name, v["worst"])
        # every drum stem played again from the truth (the lead-in's -60 dB noise is not kept)
        assert v["drum_null_db"] < (-40 if name == "leadin" else -90), name
        assert v["stems_vs_mix_db"] < -100, name  # stems sum to the mix without the clipper
        assert v["groups_max_abs"] < 1e-5, name
        assert v["mp3_lag_samples"] == 0 and v["mp3_samples"] == v["mix_samples"], name
        assert abs(v["mix_rms_db"] + 8.6) < 1.5, name
        # the 808's sounding truth read again in its stem: glides move fastest (full: 15.5 cents at most)
        assert v["sounding_notes"] >= 3 and v["sounding_err_max_cents"] < 25, name


def test_the_scorer_reads_every_truth(built):
    """evaluate reads each scenario's truth as written: it validates and
    scores 1.0 against itself on every part it has; a 6/8 grid one bar and
    10 ms late is 10 ms late (time.bpm counts quarter notes there)."""
    for name, (ev, folder) in built.items():
        s = evaluate.score(ev, ev | {"kind": "estimate"}, truth_dir=folder)
        assert s["summary"]["strict_f1_20"] == 1.0 and s["808"]["note_f1_20"] == 1.0, name
        assert s["drums"]["extra"] == [] and s["rhythm"]["beat_f"] == 1.0, name
        assert s["rhythm"]["downbeat"]["first_error_ms"] == 0.0, name
        assert not ev["notes"] or s["melodic"]["pooled"]["note_f1"] == 1.0, name
        assert not ev["chords"] or all(s["chords"][r] in (1.0, None) for r in evaluate.CHORD_RULES), name
    ev = built["meter68"][0]
    t, late = ev["time"], 3 * 60 / ev["time"]["bpm"] + 0.010  # a 6/8 bar holds three quarter notes
    shifted = {**t, "downbeat_s": t["downbeat_s"] + late, "beats_s": [x + late for x in t["beats_s"]],
               "downbeats_s": [x + late for x in t["downbeats_s"]]}
    assert evaluate.rhythm_scores(ev, {"time": shifted})["downbeat"]["first_error_ms"] == pytest.approx(10.0, abs=0.01)


def test_same_seed_same_files(rows, built, tmp_path):
    ev, folder = built["full"]
    again = S.build(rows, "full", 1, tmp_path / "again", bars=4)
    assert _hashes(folder) == _hashes(tmp_path / "again")
    assert json.dumps(again, sort_keys=True) == json.dumps(ev, sort_keys=True)


def test_v1_is_build_beats_beat(rows, tmp_path):
    """Same files, hat gains and hit times as calibrate.build_beat; the mix
    differs only where an 808 note ends (cut at the next note's sample, not
    one sample off it) and in float32 rounding."""
    old = build_beat(rows, 1, tmp_path / "old")
    ev = S.build(rows, "v1", 1, tmp_path / "new")
    paths = {s["role"]: s["path"] for s in ev["sounds"].values()}
    assert {paths[r] for r in ("kick", "hat", "open_hat", "808")} <= set(old["sounds"].values())
    for role in ("kick", "hat", "open_hat"):
        assert np.allclose([e["t"] for e in ev["drums"] if e["sound"] == role], old["positions"][role], atol=1e-5)
    assert np.allclose([e["t"] for e in ev["bass808"]], [t for t, _ in old["positions"]["808"]], atol=1e-5)
    a, b = read(tmp_path / "old" / "mix.wav"), read(tmp_path / "new" / "mix.wav")
    assert 10 * np.log10(np.sum(a ** 2) / np.sum((a - b) ** 2)) > 60


def test_808_truth_pitch(built):
    """Each 808 note's truth against a measurement in its stem: the pitch it
    settles to (the sample's measured pitch plus its transposition, cents
    included) where it has settled, and the pitch it sounds at (midi) over
    its read window. The fixture's 808 starts 191 cents up and has settled
    by 60 ms, so a note sounds at most a few cents over its settled pitch."""
    for name in ("v1", "bass_low", "melodic", "kick808"):
        ev, folder = built[name]
        stem = read(folder / "stems_truth" / "808.flac")
        assert abs(ev["sounds"]["808"]["f0_hz"] - F808) < 0.05
        cents = S.pitch_errors(ev, stem, settle_s=0.12, min_s=0.06)
        assert len(cents) >= 3 and max(abs(c) for c in cents) < 5, (name, cents)
        heard = S.sounding_errors(ev, stem)
        assert len(heard) >= 3 and max(abs(c) for c in heard) < 5, (name, heard)
        over = [100 * (e["midi"] - e["midi_settled"]) for e in ev["bass808"]]
        assert 0 <= min(over) and max(over) < 30, (name, over)
    low = built["bass_low"][0]["bass808"]
    held = [440 * 2 ** ((e["midi_settled"] - 69) / 12) for e in low]
    assert 20 <= min(held) and min(held) <= 30
    assert all(abs((e["midi_settled"] - round(e["midi_settled"])) * 100 - 11) < 0.01 for e in low)  # odd seed
    assert any("retrigger" in e["flags"] for e in low) and any("short" in e["flags"] for e in low)


def test_the_pitch_curve_and_the_sounding_pitch_follow_a_known_drop():
    """A synth 808 at 41.2 Hz that starts an octave up and falls with a 60 ms
    time constant, so its pitch is known at every sample: the curve is
    within 10 cents of it from the second period on (8.3 cents at most
    here), and a note's sounding pitch read through the curve is within 5
    cents of the same reading of the true pitch, from a 60 ms note 7
    semitones up (6.7 semitones over its settled pitch) to a 300 ms one."""
    glide = 0.060
    x = voice(Params(F808, 2 * F808, 1000 * glide, 1.2, 2, 1.5, 0.05, 0.3), 1.6)
    f0, curve = S.tuning(x)
    assert abs(1200 * math.log2(f0 / F808)) < 1
    t, cents = np.array(curve).T
    later = t >= S.crossings(x, f0)[1]
    assert np.abs(cents - 1200 * np.log2(1 + np.exp(-t / glide)))[later].max() < 10
    for semis, dur in ((5, 0.110), (-5, 0.110), (0, 0.300), (7, 0.060)):
        r = 2 ** (semis / 12)
        heard = S.sounding(lambda s, r=r: S.drop(curve, s * r) / 100, dur)
        true = S.sounding(lambda s, r=r: 12 * np.log2(1 + np.exp(-s * r / glide)), dur)
        assert abs(heard - true) < 0.08, (semis, dur, heard, true)
    assert S.read_window(0.110) == pytest.approx((0.060, 0.090)) and S.read_window(0.060) == (0.030, 0.030)


def test_swing_positions(built):
    """16th swing: every hit sits on the grid, the second 16th of each eighth at 58 % of it."""
    ev = built["swing58"][0]
    step = 60 / ev["time"]["bpm"] / 4
    pos = [(e["t"] / step) % 2 for e in ev["drums"] + ev["bass808"]]
    assert all(min(abs(p), abs(p - 1.16), abs(p - 2)) < 1e-3 for p in pos)
    assert sum(abs(p - 1.16) < 1e-3 for p in pos) > 4
    assert ev["time"]["swing"] == {"ratio": 0.58, "level": 16}


def test_meter_downbeats(built):
    for name, beats, unit in (("meter34", 3, 4), ("meter68", 6, 8)):
        t = built[name][0]["time"]
        bar = 3 * 60 / t["bpm"]  # both bars hold three quarter notes
        assert (t["beats_per_bar"], t["beat_unit"]) == (beats, unit)
        assert np.allclose(np.diff(t["downbeats_s"]), bar, atol=1e-5)
        assert np.allclose(np.diff(t["beats_s"]), bar / beats, atol=1e-5)
        kicks = [e["t"] for e in built[name][0]["drums"] if e["sound"] == "kick"]
        assert all(any(abs(k - d) < 1e-5 for k in kicks) for d in t["downbeats_s"][::2])


def test_layers_flams_and_kick_808(built):
    ev = built["layers"][0]
    snares = [e for e in ev["drums"] if e["sound"] == "snare"]
    claps = [e for e in ev["drums"] if e["sound"] == "clap"]
    assert len(claps) == len(snares) and all(c["layer_of"] == "snare" for c in claps)
    assert all(-1e-6 <= c["t"] - s["t"] <= 0.008 + 1e-6 for c, s in zip(claps, snares))
    kicks = sum(e["sound"] == "kick" for e in ev["drums"])
    assert sum(e["sound"] == "kick2" for e in ev["drums"]) == round(kicks / 2)
    fl = built["flams"][0]["drums"]
    grace = [e for e in fl if "grace" in e["flags"]]
    main = [e for e in fl if "flam" in e["flags"] and "grace" not in e["flags"]]
    assert len(grace) == len(main) > 0
    for g, m in zip(grace, main):
        assert 0.010 - 1e-6 <= m["t"] - g["t"] <= 0.030 + 1e-6
        assert abs(20 * math.log10(g["gain"] / m["gain"]) + 6) < 0.01
    k8 = built["kick808"][0]
    alone = [e for e in k8["bass808"] if not e["with_kick"]]
    assert len(alone) == round(0.3 * (len(k8["bass808"]) + sum("no_808" in e["flags"] for e in k8["drums"])))
    kick_t = [e["t"] for e in k8["drums"] if e["sound"] == "kick"]
    lags = [min(abs(b["t"] - k) for k in kick_t) for b in k8["bass808"] if b["with_kick"]]
    assert max(lags) <= 0.020 + 1e-6


def test_lead_in_glides_rolls_and_chords(built):
    lead = built["leadin"][0]["sounds"]
    assert all(0.005 <= lead[k]["attack_s"] <= 0.031 for k in ("kick", "hat", "open_hat"))
    glides = [e["glide"] for e in built["glides"][0]["bass808"] if e["glide"]]
    assert {g["shape"] for g in glides} == {"exp", "lin"}
    assert all(1 <= abs(g["to_midi"] - g["from_midi"]) <= 12 + 1e-6 for g in glides)
    rolls = built["rolls"][0]
    flags = {f for e in rolls["drums"] for f in e["flags"]}
    assert {"roll32", "roll16t", "roll32t", "roll64", "choked"} <= flags
    assert any(rolls["sounds"][e["sound"]]["semitones"] != 0 for e in rolls["drums"])
    mel = built["melodic"][0]
    assert {n["layer"] for n in mel["notes"]} == {"pad", "pluck", "lead"}
    qualities = {c["label"].split(":")[1].split("/")[0] for c in mel["chords"]}
    assert qualities <= set(CHORDS) and len(qualities) >= 8
    assert any("/" in c["label"] for c in mel["chords"])
    assert any("glide" in n["flags"] for n in mel["notes"]) and any("run32" in n["flags"] for n in mel["notes"])


def _midi(path, notes, channel=0, bpm=140):
    """notes: (beat, beats, pitch, velocity)."""
    m = mido.MidiFile(ticks_per_beat=96)
    track = mido.MidiTrack()
    m.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    msgs = []
    for start, length, pitch, vel in notes:
        msgs += [(round(start * 96), 1, mido.Message("note_on", note=pitch, velocity=vel, channel=channel)),
                 (round((start + length) * 96), 0, mido.Message("note_off", note=pitch, channel=channel))]
    now = 0
    for tick, _, msg in sorted(msgs, key=lambda m: (m[0], m[1])):
        track.append(msg.copy(time=tick - now))
        now = tick
    m.save(str(path))


def test_producer_patterns(rows, tmp_path):
    midi = tmp_path / "midi"
    midi.mkdir()
    roll = [(3.5 + i / 8, 1 / 8, 60 + i, 90) for i in range(4)]  # a pitched 32nd roll
    _midi(midi / "hh pattern.mid", [(b / 2, 0.25, 60, 100) for b in range(7)] + roll)
    _midi(midi / "oh pattern.mid", [(1.5, 0.5, 60, 110)])
    _midi(midi / "clap pattern.mid", [(1, 0.25, 60, 120), (3, 0.25, 60, 120)])
    _midi(midi / "808 line.mid", [(0, 1.5, 36, 100), (1, 1, 43, 100), (2.5, 1, 36, 100), (3.75, 0.25, 36, 90)])
    _midi(midi / "hh gm.mid", [(0, 0.25, 42, 100)], channel=9)  # GM drums: skipped
    _midi(midi / "hh long.mid", [(b / 4, 0.25, 60, 100) for b in range(2100)])  # a whole song: skipped
    (midi / "._hh pattern.mid").write_bytes(b"\x00\x05\x16\x07" + bytes(4092))  # AppleDouble stub
    found = S.producer_files(midi)
    assert [p.split("/")[-1] for p in found["hat"]] == ["hh pattern.mid"]
    ev = S.build(rows, "producer", 1, tmp_path / "beat", midi_dir=midi, bars=4)
    assert S.check_schema(ev) == [] and ev["time"]["bpm"] == 140
    pitched = {e["sound"] for e in ev["drums"] if "pitched" in e["flags"]}
    assert {"hat+1", "hat+2", "hat+3"} <= pitched
    assert {"roll32"} <= {f for e in ev["drums"] for f in e["flags"]}
    legato = [e for e in ev["bass808"] if "legato" in e["flags"]]
    assert legato and all(abs(e["glide"]["to_midi"] - e["glide"]["from_midi"] - 7) < 1e-3 for e in legato)
    assert sum(e["sound"] == "kick" for e in ev["drums"]) > 0
    text = (tmp_path / "beat" / "events.json").read_text()
    assert "pattern.mid" not in text and "808 line" not in text
    v = S.verify(tmp_path / "beat")
    assert v["attack_err_max_ms"] <= 2.0 and v["stems_vs_mix_db"] < -100


def _as_v1_rebuild(ev, folder, dest):
    """A built v1 beat in the v1 rebuild's layout: truth.json, stems_truth/*.wav, mix files."""
    dest.mkdir(parents=True)
    for name in ("mix.wav", "mix.mp3", "mix_mp3.wav"):
        (dest / name).write_bytes((folder / name).read_bytes())
    (dest / "stems_truth").mkdir()
    names = {"loop": "melodic"}
    for k, p in ev["audio"]["stems"].items():
        sf.write(dest / "stems_truth" / f"{names.get(k, k)}.wav", read(folder / p), SR, subtype="FLOAT")
    snd = ev["sounds"]
    back = "snare" if "snare" in snd else "clap"
    pos = {k: [e["t"] for e in ev["drums"] if e["sound"] == (back if k == "snare" else k)]
           for k in ("kick", "snare", "hat", "open_hat")}
    pos["808"] = [[e["t"], round(e["midi"] - S.hz_to_midi(snd["808"]["f0_hz"]))] for e in ev["bass808"]]
    keys = ("kick", "snare", "hat", "open_hat", "808")
    truth = {"seed": 1, "bpm": 136.0, "positions": pos, "n_samples": ev["audio"]["n_samples"], "sr": SR,
             "sounds": {k: snd[back if k == "snare" else k]["path"] for k in keys},
             "melodic": snd["loop"]["path"], "vocal": snd["vocal"]["path"],
             "hat_gains": [e["gain"] / 0.22 for e in ev["drums"] if e["sound"] == "hat"], "open_hat_gain": 0.9,
             "master": {"rms_gain": 0.5, "drive_db": 5.0, "out_gain": 0.95,
                        "stem_scale": ev["audio"]["master"]["stem_scale"]},
             "stems_truth": {names.get(k, k): f"stems_truth/{names.get(k, k)}.wav" for k in ev["audio"]["stems"]},
             "mp3_len": ev["audio"]["n_samples"], "clip_residual_db": -16.0}
    (dest / "truth.json").write_text(json.dumps(truth))


def test_convert_v1(rows, tmp_path):
    ev = S.build(rows, "v1", 1, tmp_path / "v1")
    _as_v1_rebuild(ev, tmp_path / "v1", tmp_path / "rebuild" / "beat1")
    out = S.convert_v1(tmp_path / "rebuild", tmp_path / "suite", rows)
    assert len(out) == 1 and out[0]["scenario"] == "v1orig" and out[0]["seed"] == 1
    conv = json.loads((tmp_path / "suite" / "v1orig-1" / "events.json").read_text())
    assert S.check_schema(conv) == []
    for key in ("sound", "velocity"):
        assert [e[key] for e in conv["drums"]] == [e[key] for e in ev["drums"]], key
    for key in ("t", "t_attack", "gain"):
        assert np.allclose([e[key] for e in conv["drums"]], [e[key] for e in ev["drums"]], atol=1e-6), key
    for key in ("t", "t_attack", "end", "midi"):  # build_beat's cuts can sit one sample off the next note
        assert np.allclose([e[key] for e in conv["bass808"]], [e[key] for e in ev["bass808"]], atol=5e-5), key
    assert (tmp_path / "suite" / "v1orig-1" / "mix.wav").stat().st_ino == (
        tmp_path / "rebuild" / "beat1" / "mix.wav").stat().st_ino  # hard-linked
    v = S.verify(tmp_path / "suite" / "v1orig-1")
    assert v["attack_err_max_ms"] <= 2.0 and v["stems_vs_mix_db"] < -100 and v["mp3_lag_samples"] == 0
