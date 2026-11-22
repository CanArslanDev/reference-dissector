"""The melody check on a synthetic stem whose notes are known: no model, no
Live, no library, no network."""

import json

import numpy as np
import pytest
import soundfile as sf

from refdissect import melodycheck as MC
from refdissect.cli import main
from refdissect.grid import Grid

SR = 22050
GRID = Grid(bpm=120.0, phase=0.25)  # a beat 0.5 s, a bar 2 s, bar 1 starts at 0.25 s


def freq_of(x: np.ndarray, sr: int = SR) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[np.argmax(spec)])


def note(pitch: int, bar: int, beat: float, length: float = 0.5) -> dict:
    """A melodic.json note at `beat` (from 0) of `bar` (from 1), `length` beats long."""
    t = GRID.phase + (bar - 1) * GRID.bar + beat * GRID.beat
    return {"t": t, "end": t + length * GRID.beat, "pitch": pitch, "cents": 0}


@pytest.fixture
def work(tmp_path):
    """Four bars of a stem; bars 2 and 3 hold four notes each (bar 3 an octave up)."""
    notes = [note(60 + 2 * k, 2, k) for k in range(4)] + [note(72 + 2 * k, 3, k) for k in range(4)]
    x = np.zeros(int((GRID.phase + 4 * GRID.bar) * SR))
    for n in notes:
        y = MC.tone(n["pitch"], n["end"] - n["t"], SR)
        a = int(round(n["t"] * SR))
        x[a:a + len(y)] += 0.3 * y
    (tmp_path / "stems").mkdir()
    sf.write(str(tmp_path / "stems" / "other.wav"), np.stack([x, x], 1), SR)
    (tmp_path / "analysis.json").write_text(json.dumps({
        "source": {"stems": str(tmp_path / "stems")},
        "grid": {"bpm": GRID.bpm, "phase": GRID.phase, "beats_per_bar": 4, "beat_unit": 4}}))
    (tmp_path / "melodic.json").write_text(json.dumps({"stems": {"other": {"tuning_cents": 0, "notes": notes}}}))
    return tmp_path


def test_bar_span_counts_bars_from_one():
    assert MC.bar_span(GRID, 1, 1) == (0.25, 2.25)
    assert MC.bar_span(GRID, 2, 3) == (2.25, 6.25)
    with pytest.raises(ValueError):
        MC.bar_span(GRID, 0, 2)
    with pytest.raises(ValueError):
        MC.bar_span(GRID, 3, 2)


def test_notes_in_span_are_in_beats_from_the_first_bar():
    notes = [note(60, 1, 3.0, length=2.0), note(62, 2, 0.0), note(64, 2, 3.5, length=4.0), note(65, 3, 0.0),
             note(67, 1, 3.75, length=0.3)]
    got = MC.notes_in_span(notes, GRID, *MC.bar_span(GRID, 2, 2))
    assert [(n["pitch"], n["start"], n["length"]) for n in got] == [(60, 0.0, 1.0), (62, 0.0, 0.5), (64, 3.5, 0.5)]
    assert got[0]["tied"] and "tied" not in got[1]  # 67 ends under a sixteenth into the bar: left out


def test_the_tone_is_at_the_notes_pitch_and_the_stems_tuning():
    assert abs(freq_of(MC.tone(69, 1.0, SR)) - 440.0) < 1.0
    y = MC.render([{"pitch": 69, "start": 1.0, "length": 2.0}], 120.0, 2.0, SR, cents=50)
    assert not np.any(y[: int(0.5 * SR) - 1])  # silent until beat 1 (0.5 s)
    assert abs(freq_of(y[int(0.6 * SR):int(1.4 * SR)]) - 440.0 * 2 ** (0.5 / 12)) < 1.5
    assert len(y) == 2 * SR and abs(y[-1]) < 1e-9  # cut at the span's end without a click


def test_midi_written_and_read_back(tmp_path):
    notes = [{"pitch": 60, "start": 0.0, "length": 0.25}, {"pitch": 60, "start": 0.25, "length": 0.25},
             {"pitch": 67, "start": 0.25, "length": 3.75}, {"pitch": 72, "start": 6.5, "length": 1.0}]
    path = MC.write_midi(notes, 137.0, tmp_path / "m.mid")
    assert MC.read_notes(path) == notes
    (tmp_path / "clip.json").write_text(json.dumps({"notes": [
        {"pitch": 67, "start_time": 0.25, "duration": 3.75, "velocity": 90},
        {"pitch": 60, "start_time": 0.0, "duration": 0.25}]}))
    assert MC.read_notes(tmp_path / "clip.json") == [notes[0], notes[2]]


def test_midi_cuts_a_note_where_its_pitch_starts_again(tmp_path):
    """Two sounding notes of one pitch cannot be told apart in MIDI: the first ends where the second starts."""
    notes = [{"pitch": 60, "start": 0.0, "length": 2.0}, {"pitch": 60, "start": 1.0, "length": 0.5},
             {"pitch": 64, "start": 0.0, "length": 2.0}]
    got = MC.read_notes(MC.write_midi(notes, 120.0, tmp_path / "m.mid", meter=(7, 8)))
    assert got == [{"pitch": 60, "start": 0.0, "length": 1.0}, {"pitch": 64, "start": 0.0, "length": 2.0},
                   {"pitch": 60, "start": 1.0, "length": 0.5}]
    for notes, want in (([(0, 2), (0, 4), (1, 0.5)], [(0.0, 1.0), (1.0, 0.5)]),      # equal starts: the longest
                        ([(0, 2), (0, 4), (1, 3)], [(0.0, 1.0), (1.0, 3.0)]),
                        ([(0, 8), (1, 4), (2, 1)], [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)])):
        path = MC.write_midi([{"pitch": 60, "start": a, "length": b} for a, b in notes], 120.0, tmp_path / "e.mid")
        assert [(n["start"], n["length"]) for n in MC.read_notes(path)] == want
    assert not np.any(MC.render([{"pitch": 60, "start": -1.0, "length": 0.5}], 120.0, 1.0, SR))
    import mido
    sig = [m for m in mido.MidiFile(str(tmp_path / "m.mid")).tracks[0] if m.type == "time_signature"][0]
    assert (sig.numerator, sig.denominator) == (7, 8)


def test_check_keeps_a_note_from_before_the_first_bar_and_refuses_a_bad_split(work, tmp_path):
    (tmp_path / "n.json").write_text(json.dumps([{"pitch": 60, "start": -0.5, "length": 1.0},
                                                 {"pitch": 62, "start": -2.0, "length": 1.0},
                                                 {"pitch": 64, "start": 1.0, "length": 0.5},
                                                 {"pitch": 65, "start": 3.5, "length": 2.0},
                                                 {"pitch": 67, "start": 0.0, "length": 0.0}]))
    m = MC.check(work, "other", (2, 2), out_dir=tmp_path / "o", notes_file=tmp_path / "n.json")
    assert [(n["pitch"], n["start"], n["length"]) for n in m["notes"]] == [
        (60, 0.0, 0.5), (67, 0.0, 0.0), (64, 1.0, 0.5), (65, 3.5, 0.5)]
    with pytest.raises(ValueError, match="bars_per_file"):
        MC.check(work, "other", (2, 3), bars_per_file=0)


def test_diff_counts_each_kind_of_change():
    reading = [{"pitch": 60, "start": 0.0, "length": 0.5}, {"pitch": 62, "start": 1.0, "length": 0.5},
               {"pitch": 64, "start": 2.0, "length": 0.5}, {"pitch": 65, "start": 3.0, "length": 0.5}]
    corrected = [{"pitch": 60, "start": 0.05, "length": 1.0},  # kept (within a 32nd)
                 {"pitch": 63, "start": 1.0, "length": 0.5},   # repitched
                 {"pitch": 64, "start": 2.5, "length": 0.5},   # moved
                 {"pitch": 70, "start": 3.5, "length": 0.5}]   # added; the reading's 65 removed
    d = MC.diff(reading, corrected)
    assert (d["kept"], d["repitched"], d["moved"], d["removed"], d["added"]) == (1, 1, 1, 1, 1)
    assert d["precision"] == 0.25 and d["recall"] == 0.25
    same = MC.diff(reading, reading)
    assert same["kept"] == 4 and same["precision"] == 1.0 and same["removed"] == same["added"] == 0
    assert MC.diff([], corrected)["precision"] is None


def test_check_writes_the_files_and_the_tone_follows_the_notes(work):
    m = MC.check(work, "other", (2, 3))
    out = work / "melody-check"
    assert m["notes_per_bar"] == [4, 4] and m["notes_from"] == "melodic.json" and m["against_reading"] is None
    assert m["span_s"] == [2.25, 6.25]
    assert json.loads((out / "melody-check.json").read_text())["notes"] == m["notes"]
    assert [n["pitch"] for n in MC.read_notes(out / "melody.mid")] == [60, 62, 64, 66, 72, 74, 76, 78]
    over, sr = sf.read(str(out / "melody_overlay.wav"))
    assert sr == SR and over.shape == (4 * SR, 2)
    for k, pitch in enumerate([60, 62, 64, 66]):  # each half-second note: the stem left, the same pitch right
        seg = over[int((k * 0.5 + 0.05) * SR):int((k * 0.5 + 0.45) * SR)]
        want = 440.0 * 2 ** ((pitch - 69) / 12)
        assert abs(freq_of(seg[:, 0]) - want) < 3 and abs(freq_of(seg[:, 1]) - want) < 3
    ab, _ = sf.read(str(out / "melody_AB.wav"))
    assert m["ab"][0]["bars"] == [2, 3] and m["ab"][0]["notes"] == 8
    assert abs(len(ab) / SR - m["ab"][0]["seconds"]) < 0.01 and m["ab"][0]["seconds"] > 8.0  # stem + beep + tone


def test_check_reads_a_corrected_file_back(work, tmp_path):
    first = MC.check(work, "other", (2, 3))
    fixed = [dict(n) for n in first["notes"]]
    fixed[1]["pitch"] = 61           # repitched
    del fixed[7]                     # removed
    fixed.append({"pitch": 84, "start": 7.5, "length": 0.5})   # added
    fixed.append({"pitch": 90, "start": 9.0, "length": 0.5})   # past bar 3: not part of the check
    MC.write_midi(fixed, 120.0, tmp_path / "fixed.mid")
    m = MC.check(work, "other", (2, 3), out_dir=tmp_path / "second", notes_file=tmp_path / "fixed.mid")
    d = m["against_reading"]
    assert (d["reading"], d["corrected"], d["kept"], d["repitched"], d["removed"], d["added"]) == (8, 8, 6, 1, 1, 1)
    assert m["notes_per_bar"] == [4, 4] and 90 not in [n["pitch"] for n in m["notes"]]
    over, _ = sf.read(str(tmp_path / "second" / "melody_overlay.wav"))
    seg = over[int(0.55 * SR):int(0.95 * SR), 1]  # the second note now sounds a semitone up
    assert abs(freq_of(seg) - 440.0 * 2 ** ((61 - 69) / 12)) < 3


def test_check_splits_the_ab_files_and_skips_empty_bars(work):
    m = MC.check(work, "other", (1, 4), bars_per_file=1)
    assert [r["bars"] for r in m["ab"]] == [[1, 1], [2, 2], [3, 3], [4, 4]]
    assert [r["notes"] for r in m["ab"]] == [0, 4, 4, 0]
    assert m["ab"][0]["file"] is None and m["ab"][1]["file"].endswith("melody_AB_bars2-2.wav")
    assert not (work / "melody-check" / "melody_AB_bars1-1.wav").exists()
    assert (work / "melody-check" / "melody_AB_bars3-3.wav").is_file()


def test_check_refuses_bars_outside_the_stem_and_unknown_stems(work):
    with pytest.raises(ValueError, match="outside"):
        MC.check(work, "other", (4, 6))
    with pytest.raises(ValueError, match="no stem"):
        MC.check(work, "vocals", (1, 2))


def test_check_against_a_layer_file_melodic_json_does_not_know(work, tmp_path):
    """A layer cut by another separation: its own audio and notes, no reading to compare with."""
    x, sr = sf.read(str(work / "stems" / "other.wav"))
    sf.write(str(tmp_path / "piano.wav"), 0.5 * x, sr)
    notes = [{"pitch": 60, "start": 0.0, "length": 0.5}, {"pitch": 62, "start": 1.0, "length": 0.5}]
    (tmp_path / "piano.json").write_text(json.dumps(notes))
    m = MC.check(work, "piano", (2, 2), out_dir=tmp_path / "o", notes_file=tmp_path / "piano.json",
                 audio=tmp_path / "piano.wav")
    assert m["audio"] == str(tmp_path / "piano.wav") and m["against_reading"] is None and m["tuning_cents"] == 0.0
    assert m["notes_per_bar"] == [2] and (tmp_path / "o" / "melody_AB.wav").is_file()
    over, _ = sf.read(str(tmp_path / "o" / "melody_overlay.wav"))
    assert abs(freq_of(over[int(0.55 * SR):int(0.95 * SR), 1]) - 440.0 * 2 ** ((62 - 69) / 12)) < 3
    with pytest.raises(ValueError, match="needs its own notes"):
        MC.check(work, "piano", (2, 2), audio=tmp_path / "piano.wav")
    with pytest.raises(ValueError, match="needs both audio and notes"):
        MC.check(work, "piano", (2, 2), notes_file=tmp_path / "piano.json")
    # a stem melodic.json knows, against another file: its reading is the stem's, so nothing is compared
    m = MC.check(work, "other", (2, 2), out_dir=tmp_path / "o2", notes_file=tmp_path / "piano.json",
                 audio=tmp_path / "piano.wav")
    assert m["against_reading"] is None
    with pytest.raises(ValueError, match="needs its own notes"):  # the stem's notes are not the layer's
        MC.check(work, "other", (2, 2), out_dir=tmp_path / "o4", audio=tmp_path / "piano.wav")
    (work / "melodic.json").unlink()
    m = MC.check(work, "piano", (2, 2), out_dir=tmp_path / "o3", notes_file=tmp_path / "piano.json",
                 audio=tmp_path / "piano.wav")
    assert len(m["notes"]) == 2


def test_cli_runs_the_check_and_reports_the_correction(work, tmp_path, capsys):
    assert main(["melody-check", str(work), "--bars", "2-3"]) in (0, None)
    assert "8 notes from melodic.json in bars 2-3 of other (per bar: 4 4)" in capsys.readouterr().out
    notes = MC.read_notes(work / "melody-check" / "melody.mid")[:6]
    MC.write_midi(notes, 120.0, tmp_path / "c.mid")
    main(["melody-check", str(work), "--bars", "2-3", "--notes", str(tmp_path / "c.mid"), "--out", str(tmp_path / "o")])
    text = capsys.readouterr().out
    assert "6 kept, 0 repitched, 0 moved, 2 removed, 0 added" in text and "precision 0.75, recall 1.0" in text
    assert (tmp_path / "o" / "melody_AB.wav").is_file()
    main(["melody-check", str(work), "--stem", "layer", "--bars", "2-3", "--notes", str(tmp_path / "c.mid"),
          "--audio", str(work / "stems" / "other.wav"), "--out", str(tmp_path / "o2")])
    text = capsys.readouterr().out
    assert "6 notes" in text and "of layer" in text and "against the reading" not in text
