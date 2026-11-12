"""Style candidates (candidates.py) on a synthetic work folder: a plan whose
statistics are known, its chords, and stems rendered from it. No library,
no Demucs."""

import json

import numpy as np
import pytest
import soundfile as sf

from refdissect import candidates as cd
from refdissect.audio import SR
from refdissect.render import render_plan

BPM = 140.0
BARS = 16


def _wav(path, x):
    sf.write(path, np.asarray(x, np.float32), SR, subtype="FLOAT")
    return str(path)


def _note(pitch, beat, velocity=100, duration=0.25):
    return {"pitch": pitch, "start_time": round(beat, 4), "duration": duration, "velocity": velocity}


def reference_plan(folder, kit) -> dict:
    """16 bars of 4/4: kicks on 16ths 0 and 10 (and 12 every other bar),
    the snare on 8, hats on the 8ths with a 32nd-triplet roll on beat 3 of
    every other bar; the 808 on every kick (A1, D2 on the 10th 16th), and
    in bars 4n+3 a slide from the 12th 16th's A1 up to A2 on the 14th."""
    pads = {"36": {"path": _wav(folder / "kick.wav", kit["kick"]), "transpose": 0, "role": "kick"},
            "38": {"path": _wav(folder / "snare.wav", kit["snare"]), "transpose": 0, "role": "snare"},
            "42": {"path": _wav(folder / "hat.wav", kit["hat"]), "transpose": 0, "role": "hat"}}
    t = np.arange(int(1.2 * SR)) / SR
    shot = _wav(folder / "808.wav", np.sin(2 * np.pi * 55.0 * t) * np.exp(-t / 0.6))
    drums, b808 = [], []
    for b in range(BARS):
        at = 4.0 * b
        kicks = [0, 10] + ([12] if b % 2 else [])
        drums += [_note(36, at + s / 4, 110) for s in kicks]
        drums.append(_note(38, at + 2.0, 105))
        for s in range(0, 16, 2):
            if b % 2 and s in (8, 10):
                continue
            drums.append(_note(42, at + s / 4, 80))
        if b % 2:
            drums += [_note(42, at + 2.0 + k / 12, 70, 1 / 12) for k in range(12)]
        for s in kicks:
            b808.append(_note(38 if s == 10 else 33, at + s / 4, 120))
        if b % 4 == 3:
            b808.append(_note(45, at + 3.5, 120))
    drums.sort(key=lambda n: n["start_time"])
    b808.sort(key=lambda n: n["start_time"])
    for k, n in enumerate(b808):  # held to the next; the octave note slides in (the one before held over it)
        nxt = b808[k + 1]["start_time"] if k + 1 < len(b808) else 4.0 * BARS
        n["duration"] = round(nxt - n["start_time"] + (0.02 if k + 1 < len(b808) and b808[k + 1]["pitch"] == 45
                                                        else -0.001), 4)
    return {"tempo": BPM, "beats_per_bar": 4, "beat_unit": 4, "length_beats": 4.0 * BARS,
            "tracks": {"drums": {"pads": pads, "notes": drums},
                       "808": {"sample_path": shot, "root_midi": 33, "notes": b808, "voices": 1,
                               "glide_mode": "Glide", "glide_time_ms": 80.0}},
            "stems": {}, "locators": [["A (1)", 0.0], ["B (9)", 32.0]],
            "levels": {"headroom_db": 4.0, "drums_track_db": -6.0, "808_track_db": -6.0}}


def melodic() -> dict:
    """A:min for a bar, F:maj for a bar, over and over."""
    segs = [{"start_time": 8.0 * k + 4.0 * h, "start": (8.0 * k + 4.0 * h) * 60 / BPM, "label": lab}
            for k in range(BARS // 2) for h, lab in ((0, "A:min"), (1, "F:maj"))]
    segs = [s for s in segs if s["start_time"] < 4.0 * BARS]
    return {"chords": {"segments": segs}, "stems": {}}


@pytest.fixture(scope="module")
def work(tmp_path_factory, kit):
    """plan.json, melodic.json and stems/ (drums, bass: the 808, other: the
    chords ducked under the kicks) rendered from the plan."""
    folder = tmp_path_factory.mktemp("work")
    plan = reference_plan(folder, kit)
    m = melodic()
    r = render_plan(plan, folder / "render")
    n = r["n_samples"]
    drums = sf.read(folder / "render" / "drums.wav")[0]
    bass = sf.read(folder / "render" / "808.wav")[0]
    line = cd.chord_line(m)
    chord_notes = []
    for k, s in enumerate(line):
        end = line[k + 1]["start"] if k + 1 < len(line) else 4.0 * BARS
        root, tones = cd.chord_pitches(s["label"])
        chord_notes += [_note(48 + root + t, s["start"], 90, end - s["start"]) for t in tones]
    other = cd.render_chords(chord_notes, BPM, n)
    other *= cd.duck_gain(n, cd.kick_seconds(plan), {"s": [0.0, 0.003, 0.15], "db": [0.0, -12.0, 0.0]})
    stems = folder / "stems"
    stems.mkdir()
    for name, x in {"drums": drums, "bass": bass, "other": other}.items():
        _wav(stems / f"{name}.wav", np.stack([x, x], 1))
    plan["stems"] = {k: {"path": str(stems / f"{k}.wav"), "mute": True} for k in ("drums", "bass", "other")}
    (folder / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (folder / "melodic.json").write_text(json.dumps(m), encoding="utf-8")
    return folder


def _plan(work):
    return json.loads((work / "plan.json").read_text())


def test_the_statistics_read_the_plans_placement(work):
    s = cd.style_stats(_plan(work), None, melodic())
    kick, snare = s["drums"]["kick"]["p"], s["drums"]["snare"]["p"]
    assert (kick[0], kick[10], kick[12], kick[4]) == (1.0, 1.0, 0.5, 0.0)
    assert snare[8] == 1.0 and sum(snare) == 1.0
    assert s["rolls"]["rate"] == pytest.approx(8 / 64) and set(s["rolls"]["divisions"]) == {12}
    assert s["swing"]["ratio_16"] is None  # hats on the 8ths: nothing off the 8ths to read
    b = s["808"]
    assert b["with_kick"] == 1.0 and max(b["off_kick"]) == pytest.approx(4 / 16)  # the octave note, every 4th bar
    assert b["slide_rate"] > 0 and set(b["slide_intervals"]) == {12}  # A1 to A2
    assert b["register"][1] == 33 and b["glide_time_ms"] == 80.0
    assert set(s["chords"]["vocabulary"]) == {"A:min", "F:maj"}
    assert s["chords"]["moves"] == {"A:min > F:maj": 8, "F:maj > A:min": 7}
    assert s["chords"]["half_bars"] == {"2": 16}


def test_a_section_reads_its_own_bars(work):
    plan = _plan(work)
    assert cd.section_bars(plan, "B") == list(range(8, 16))
    assert cd.style_stats(plan, section="B")["bars"] == 8
    with pytest.raises(ValueError, match="No section"):
        cd.section_bars(plan, "C")


def test_swing_moves_the_off_16ths(work):
    assert cd._swung(1, {"ratio_16": 0.66}) == pytest.approx(0.33)
    assert cd._swung(3, {"ratio_16": 0.66}) == pytest.approx(0.83)
    assert cd._swung(2, {"ratio_16": None, "ratio_8": 0.6}) == pytest.approx(0.6)
    assert cd._swung(6, {"ratio_16": None, "ratio_8": None}) == pytest.approx(1.5)
    # the analysis' swing is taken over the plan's hats
    s = cd.style_stats(_plan(work), {"grid": {"swing": {"ratio_16": 0.6, "ratio_8": None}}})
    assert s["swing"] == {"ratio_16": 0.6, "ratio_8": None, "from": "analysis"}


def test_the_same_seed_draws_the_same_candidate_whatever_their_number(work):
    s = cd.style_stats(_plan(work), None, melodic())
    a, b = cd.draw(s, 7, 3), cd.draw(s, 7, 3)
    assert a == b
    assert a != cd.draw(s, 7, 2) and a != cd.draw(s, 8, 3)
    ref = cd.reference_events(_plan(work), melodic())
    one = cd.candidate(s, _plan(work), ref, 7, 3)
    assert one == cd.candidate(s, _plan(work), ref, 7, 3)
    assert cd.digest(one["notes"]) == cd.digest(cd.candidate(s, _plan(work), ref, 7, 3)["notes"])


def test_a_candidate_keeps_the_style(work):
    plan = _plan(work)
    s = cd.style_stats(plan, None, melodic())
    for number in range(1, 6):
        notes = cd.draw(s, 1, number)
        assert {n["pitch"] for n in notes["drums"]} <= {36, 38, 42}
        snares = sorted(n["start_time"] for n in notes["drums"] if n["pitch"] == 38)
        assert snares == [4.0 * b + 2.0 for b in range(8)]  # the snare plays its 16th in every bar
        assert all(0 <= n["start_time"] < 32 for part in ("drums", "808", "chords") for n in notes[part])
        assert {n["label"] for n in notes["chords"]} <= {"A:min", "F:maj"}
        assert all(21 <= n["pitch"] <= 57 for n in notes["808"])
        kicks = {n["start_time"] for n in notes["drums"] if n["pitch"] == 36}
        starts = {n["start_time"] for n in notes["808"]}
        assert len(kicks & starts) >= 0.7 * len(kicks)  # the 808 comes with the kicks (with_kick 1)
        cp = cd.candidate_plan(plan, s, notes, 8, number)
        assert cp["length_beats"] == 32.0 and cp["tempo"] == BPM
        assert cp["tracks"]["808"]["sample_path"] == plan["tracks"]["808"]["sample_path"]
        if notes["slides"]:
            assert cp["tracks"]["808"]["glide_mode"] == "Glide"
            assert cp["tracks"]["808"]["glide_time_ms"] == 80.0


def test_slides_hold_the_note_before_over_the_next(work):
    s = cd.style_stats(_plan(work), None, melodic())
    s["808"]["slide_rate"] = 1.0
    notes = cd.draw(s, 1, 1)
    n808 = notes["808"]
    assert len(notes["slides"]) == len(n808) - 1
    for a, b in zip(n808, n808[1:]):
        assert a["start_time"] + a["duration"] > b["start_time"]
        assert abs(b["pitch"] - a["pitch"]) == 12


def test_copy_score_and_drawing_again(work):
    plan = _plan(work)
    ref = cd.reference_events(plan, melodic())
    assert cd.copy_score(ref, ref) == {"mean": 1.0, "max": 1.0}
    assert cd.copy_score({0: {("kick", 3)}}, ref)["mean"] == 0.0
    # with every chance at 0 or 1, the reference's own bars come back: drawn again, then flagged
    s = cd.style_stats(plan, None, melodic(), "A")
    for role in s["drums"].values():
        role["p"] = [round(x) for x in role["p"]]
        role["change"] = 0.0
    s["rolls"]["rate"], s["808"]["change"] = 0.0, 0.0
    c = cd.candidate(s, plan, ref, 1, 1, tries=3, max_copy=0.2)
    if not c["under_max_copy"]:
        assert c["tries"] == 3 and c["copy"]["mean"] > 0.2
    loose = cd.candidate(s, plan, ref, 1, 1, tries=3, max_copy=1.0)
    assert loose["tries"] == 1 and loose["under_max_copy"]


def test_duck_gain_follows_the_curve_from_each_kick():
    curve = {"s": [0.0, 0.02, 0.1], "db": [0.0, -12.0, 0.0]}
    g = 20 * np.log10(cd.duck_gain(SR, [0.1, 0.15], curve))
    assert g[int(0.05 * SR)] == pytest.approx(0.0)
    assert g[int(0.12 * SR)] == pytest.approx(-12.0, abs=0.01)
    assert g[int(0.16 * SR)] == pytest.approx(-6.0, abs=0.1)  # the second kick's -6, under the first's -3
    assert g[int(0.17 * SR)] == pytest.approx(-12.0, abs=0.01)
    assert g[int(0.3 * SR)] == 0.0
    assert np.all(cd.duck_gain(100, [0.0], {"s": [0.0, 0.5], "db": [3.0, 3.0]}) == 1.0)  # never a boost


def test_fit_ducking_reads_the_references_curve(work):
    from refdissect.symmetric import load_stems
    duck = cd.fit_ducking(load_stems(work / "stems"))
    assert duck["chords"]["source"] == "reference"
    db = dict(zip(duck["chords"]["s"], duck["chords"]["db"]))
    assert db[0.05] < -4 and db[0.3] == pytest.approx(0.0, abs=1.5)
    assert cd.fit_ducking({})["808"]["source"] == "default"


@pytest.mark.slow
def test_run_renders_compares_and_rounds_reproducibly(work, tmp_path):
    m = cd.run(work, tmp_path / "a", n=2, seed=4, section="A", log=lambda *_: None)
    assert m["reference_span_s"][0] == 0.0 and m["reference_span_s"][1] == pytest.approx(8 * 4 * 60 / BPM, abs=1e-3)
    for row in m["candidates"]:
        d = tmp_path / "a" / f"{row['number']:02d}"
        for f in ("plan.json", "midi/drums.mid", "midi/808.mid", "midi/chords.mid", "render/drums.wav",
                  "chords.wav", "808-ducked.wav", f"candidate-{row['number']:02d}.wav", "compare.json"):
            assert (d / f).is_file(), f
        x, sr = sf.read(row["mix"])
        assert sr == SR and len(x) == int(round(32 / (BPM / 60) * SR))
        c = row["compare"]
        assert c["ducking_808_db"] is not None and c["ducking_chords_db"] is not None
        assert set(c["drum_roles_db"]) == {"kick", "snare", "hat"}
    assert [f.name for f in sorted((tmp_path / "a" / "round").glob("*.wav"))] == ["1.wav", "2.wav"]
    assert json.loads((tmp_path / "a" / "candidates.json").read_text())["candidates"][0]["digest"]
    again = cd.run(work, tmp_path / "b", n=2, seed=4, section="A", log=lambda *_: None)
    for r1, r2 in zip(m["candidates"], again["candidates"]):
        assert r1["digest"] == r2["digest"] and r1["compare"] == r2["compare"]
        assert np.array_equal(sf.read(r1["mix"])[0], sf.read(r2["mix"])[0])


def test_the_command_without_stems_or_comparison(work, tmp_path):
    from refdissect.cli import main
    plan = _plan(work)
    plan["stems"] = {}
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "plan.json").write_text(json.dumps(plan))
    assert main(["candidates", str(tmp_path / "w"), "--out", str(tmp_path / "o"), "--n", "1", "--seed", "2",
                 "--no-compare"]) == 0
    m = json.loads((tmp_path / "o" / "candidates.json").read_text())
    assert m["ducking"]["808"]["source"] == "default" and "round" not in m
    assert m["stats"]["chords"] is None and m["candidates"][0]["notes"]["chords"] == []
    assert (tmp_path / "o" / "01" / "candidate-01.wav").is_file()
