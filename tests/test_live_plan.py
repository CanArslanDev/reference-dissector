import mido
import numpy as np
import soundfile as sf

from refdissect.audio import SR
from refdissect.live import make_plan, no_overlaps, velocities, write_midi


def test_no_overlaps_and_velocities():
    notes = [{"pitch": 38, "start_time": 0.0, "duration": 1.0, "velocity": 100},
             {"pitch": 38, "start_time": 0.5, "duration": 1.0, "velocity": 100},
             {"pitch": 42, "start_time": 0.25, "duration": 1.0, "velocity": 90}]
    out = no_overlaps(notes, 1.0)
    first = next(n for n in out if n["pitch"] == 38 and n["start_time"] == 0.0)
    assert first["duration"] < 0.5
    assert velocities([0.5, 1.0, 1.0]) == [64, 127, 127]


def _fixture(tmp_path):
    snare = tmp_path / "snare.wav"
    sf.write(snare, np.random.default_rng(0).standard_normal(SR // 4) * 0.5, SR)
    for stem in ("drums", "bass", "other"):
        sf.write(tmp_path / f"{stem}.wav", np.random.default_rng(1).standard_normal(SR * 4) * 0.1, SR)
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, np.sin(2 * np.pi * 33.2 * np.arange(SR) / SR) * 0.9, SR)
    analysis = {
        "grid": {"bpm": 120.0}, "source": {"duration_s": 4.0},
        "templates": {"sound_1": {"detected_hits": 20, "clean": 1.0},
                      "sound_2": {"detected_hits": 20, "clean": 1.0}},
        "events": {"sound_1": [{"t": 0.5 * k, "gain": 1.0} for k in range(8)],
                   "sound_2": [{"t": 0.25 + 0.5 * k, "gain": 0.5} for k in range(8)]},
        "notes_808": [{"t": 0.0, "end": 1.0, "pitch": 24, "peak": 1.0}],
        "sections": [{"name": "A", "start_bar": 1, "end_bar": 2}]}
    top = [{"path": str(snare), "ncc": 0.97, "semitones": -1.0}, {"path": str(snare), "ncc": 0.5, "semitones": 0.0}]
    matches = {"sound_1": {"role": "snare", "top": top, "verdict": "same sample", "same_as": None},
               "sound_2": {"role": "snare", "top": top, "verdict": "same sound as sound_1", "same_as": "sound_1"}}
    synth = {"root_midi": 24, "file_name": shot.name, "file": str(shot)}
    return analysis, matches, synth


def test_plan(tmp_path):
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert list(pads) == ["38"] and pads["38"]["transpose"] == -1
    notes = plan["tracks"]["drums"]["notes"]
    assert len(notes) == 16 and {n["pitch"] for n in notes} == {38}  # same_as shares the pad
    assert notes[1]["start_time"] == 0.5  # 0.25 s at 120 BPM
    assert plan["tracks"]["808"]["transpose"] == 36  # MIDI 24 plays the sample's own pitch
    assert plan["stems"]["drums"]["mute"] and not plan["stems"]["other"]["mute"]
    assert plan["locators"] == [["A (1)", 0.0]]
    assert "drums_track_db" in plan["levels"] and "808_track_db" in plan["levels"]


def test_pitched_copies_get_transposed_pads(tmp_path):
    """Hits of a pitched copy (detect: semitones) play on pads of their own,
    the sound's sample transposed that much further, the plain ones on the
    sound's pad; and the plan scores them as sounds of their own."""
    from refdissect import evaluate
    analysis, matches, synth = _fixture(tmp_path)
    analysis["events"]["sound_1"] += [{"t": 0.2 + 0.5 * k, "gain": 1.0, "semitones": 1 + k % 2} for k in range(6)]
    analysis["events"]["sound_2"].append({"t": 3.9, "gain": 0.5, "semitones": 1})  # its twin's copy: the same pad
    plan = make_plan(analysis, matches, synth, tmp_path)
    pads = plan["tracks"]["drums"]["pads"]
    assert pads["38"]["transpose"] == -1 and (pads["54"]["transpose"], pads["55"]["transpose"]) == (0, 1)
    assert len(pads) == 3
    notes = plan["tracks"]["drums"]["notes"]
    assert sum(n["pitch"] == 54 for n in notes) == 4 and sum(n["pitch"] == 55 for n in notes) == 3
    est = evaluate.estimate_from_plan(plan, root=tmp_path)
    assert {s: v["semitones"] for s, v in est["sounds"].items() if s != "808"} == {"38": -1, "54": 0, "55": 1}


def test_locators_on_the_bar_lines(tmp_path):
    """Bar 1 starts at the grid's phase (here 1 s = 2 beats), bars of 3 beats in 3/4."""
    analysis, matches, synth = _fixture(tmp_path)
    analysis["grid"].update(phase=1.0, beats_per_bar=3, beat_unit=4)
    analysis["sections"] = [{"name": "A", "start_bar": 1, "end_bar": 2}, {"name": "B", "start_bar": 3, "end_bar": 4}]
    plan = make_plan(analysis, matches, synth, tmp_path)
    assert plan["locators"] == [["A (1)", 2.0], ["B (3)", 8.0]]


def test_midi_files(tmp_path):
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    written = write_midi(plan, tmp_path / "midi")
    mid = mido.MidiFile(str(written[0]))
    ons = [m for m in mid.tracks[0] if m.type == "note_on"]
    assert len(ons) == 16
    tempo = next(m.tempo for m in mid.tracks[0] if m.type == "set_tempo")
    assert round(mido.tempo2bpm(tempo)) == 120


def test_build_spec_for_build_tracks(tmp_path):
    from refdissect.live import build_spec
    analysis, matches, synth = _fixture(tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path)
    spec = {t["name"]: t for t in build_spec(plan)}
    t808, drums = spec["808 (synth)"], spec["Drums (library)"]
    # the one-shot is named after the song, so plans for different songs do not collide
    assert t808["instrument"] == {"sample": plan["tracks"]["808"]["sample"], "warping": False}
    assert t808["instrument"]["sample"].endswith("808 (synth).wav")
    assert t808["settings"]["Transpose"] == 36.0 and t808["clips"][0]["notes"]
    pad = drums["instrument"]["drum_rack"]["38"]
    assert pad["sample"] == "snare.wav" and pad["settings"]["Transpose"] == -1.0
    assert len(drums["clips"][0]["notes"]) == 16
    ref = spec["REF drums"]
    assert ref["kind"] == "audio" and ref["mute"] and ref["audio_clips"][0]["warping"] is False
    assert not spec["REF other"]["mute"]


def test_melodic_tracks_in_the_plan(tmp_path):
    """melodic.json adds one muted MIDI track per stem; the drums and the 808
    are planned exactly as without it."""
    from refdissect.live import MELODIC_INSTRUMENT, build_spec
    analysis, matches, synth = _fixture(tmp_path)
    melodic = {"stems": {"other": {"tuning_cents": 12, "notes": [
        {"pitch": 64, "start_time": 1.0, "duration": 2.0, "velocity": 90, "t": 0.5, "level_db": -20.0},
        {"pitch": 64, "start_time": 2.0, "duration": 1.0, "velocity": 80, "t": 1.0, "level_db": -21.0}]}}}
    plain = make_plan(analysis, matches, synth, tmp_path)
    plan = make_plan(analysis, matches, synth, tmp_path, melodic=melodic)
    assert "melodic" not in plain["tracks"]
    assert {k: plan["tracks"][k] for k in ("drums", "808")} == {k: plain["tracks"][k] for k in ("drums", "808")}
    track = plan["tracks"]["melodic"]["other"]
    assert track["instrument"] == MELODIC_INSTRUMENT and track["cents"] == 12
    assert [n["duration"] for n in track["notes"]] == [0.995, 1.0]  # ends before the next note of its pitch
    spec = {t["name"]: t for t in build_spec(plan)}
    assert spec["other (notes)"]["mute"] is True and spec["other (notes)"]["instrument"] == MELODIC_INSTRUMENT
    assert spec["other (notes)"]["clips"][0]["notes"] == track["notes"]
    written = write_midi(plan, tmp_path / "midi")
    assert [w.name for w in written] == ["drums.mid", "808.mid", "other.mid"]
    ons = [m for m in mido.MidiFile(str(written[2])).tracks[0] if m.type == "note_on"]
    assert [m.note for m in ons] == [64, 64] and ons[0].channel == 0
