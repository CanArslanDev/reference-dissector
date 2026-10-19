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
