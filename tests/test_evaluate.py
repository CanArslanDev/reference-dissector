"""evaluate on events whose scores can be counted by hand."""

import json

import numpy as np
import pytest

from refdissect import evaluate
from refdissect.audio import PRE, SR, attack_time, samples
from refdissect.evaluate import SCHEMA, drum_scores, score, validate

# 16 s at 120 BPM: kick on beats 1 and 3, snare on 2 and 4, hats on every 8th (on the kicks and snares
# too, as in a real beat), an 808 note with every kick
KICK = np.arange(16) * 1.0
SNARE = 0.5 + np.arange(16) * 1.0
HAT = np.arange(64) * 0.25
KIT = (("36", "kick", KICK), ("38", "snare", SNARE), ("42", "hat", HAT))


def hits(sound, role, times, lead=0.0):
    return [{"sound": sound, "role": role, "t": float(t), "t_attack": float(t) + lead, "gain": 1.0,
             "velocity": None, "layer_of": None, "flags": []} for t in times]


def notes808(times, midi=36.0):
    return [{"sound": "808", "t": float(t), "t_attack": float(t), "end": float(t) + 0.4, "midi": midi,
             "glide": None, "with_kick": True, "gain": 1.0, "flags": []} for t in times]


def events(kind, drums=(), bass=(), **parts):
    sounds = {e["sound"]: {"role": e["role"]} for e in drums} | ({"808": {"role": "808"}} if bass else {})
    return {"schema": SCHEMA, "kind": kind, "scenario": "test", "seed": 0, "sounds": sounds, "drums": list(drums),
            "bass808": list(bass), **parts}


def truth():
    return events("truth", [e for role, times in (("kick", KICK), ("snare", SNARE), ("hat", HAT))
                            for e in hits(role, role, times)], notes808(KICK))


def estimate(shift=0.0, pads=KIT, midi=36.0):
    return events("estimate", [e for pad, role, times in pads for e in hits(pad, role, np.asarray(times) + shift)],
                  notes808(KICK + shift, midi))


def test_a_perfect_estimate_scores_one():
    s = score(truth(), estimate())
    assert s["summary"] == {"strict_f1_20": 1.0, "v1_f1": 1.0, "over": ["hat", "kick", "snare", "808 onsets"]}
    assert {k: v["estimated"] for k, v in s["drums"]["sounds"].items()} == {"hat": ["42"], "kick": ["36"],
                                                                           "snare": ["38"]}
    assert s["808"]["note_f1_20"] == 1.0 and s["drums"]["extra"] == []
    json.dumps(s)  # plain JSON all the way down


def test_a_constant_offset_fails_the_strict_score_and_passes_v1s():
    """25 ms late everywhere: nothing within 20 ms, everything within 50 ms
    and after v1's offset removal."""
    s = score(truth(), estimate(shift=0.025))
    for v in s["drums"]["sounds"].values():
        assert (v["f1_20"], v["f1_50"], v["v1"]["f1"]) == (0.0, 1.0, 1.0)
        assert v["error_ms"]["mean"] == pytest.approx(25.0, abs=0.01) and v["v1"]["offsets_ms"] == [25.0]
    assert (s["summary"]["strict_f1_20"], s["summary"]["v1_f1"]) == (0.0, 1.0)


def test_a_doubled_pad_halves_precision():
    s = drum_scores(truth(), estimate(pads=(*KIT, ("40", "snare", SNARE))))
    snare = s["sounds"]["snare"]
    assert snare["estimated"] == ["38", "40"] and snare["n_est"] == 32
    assert (snare["precision_20"], snare["recall_20"]) == (0.5, 1.0)
    assert snare["f1_20"] == pytest.approx(2 / 3, abs=1e-4)


def test_an_unrelated_sound_is_extra():
    """Hits 125 ms from every 8th match nothing; a pad with most hits on the
    kicks but 5 of 8 elsewhere is extra too, and names what it came closest to."""
    odd = 0.125 + np.arange(32) * 0.5
    stray = np.r_[KICK[:3], odd[:5]]
    s = drum_scores(truth(), estimate(pads=(*KIT, ("50", "perc", odd), ("51", "perc", stray))))
    assert s["extra"] == [{"sound": "50", "closest": None, "hits": 32, "matched_50": 0, "moments": 32, "share": 0.0},
                          {"sound": "51", "closest": "kick", "hits": 8, "matched_50": 3, "moments": 8,
                           "share": 0.375}]
    assert s["macro"] == {"f1_20": 1.0, "f1_50": 1.0, "v1_f1": 1.0}


def test_doubled_hits_score_alike_on_one_pad_or_two():
    """16 snares. A pad that plays each one twice is pooled (precision 0.5,
    F1 2/3); with one stray hit too it still is (16 of its 17 moments match:
    F1 32/49), where counting hits made it extra (16 of 33) and scored the
    snare 0. With 8 strays, one pad and the same 40 hits on two pads score
    the same, F1 4/7."""
    snares = events("truth", hits("snare", "snare", SNARE))
    stray = 0.25 + np.arange(8) * 1.0  # 250 ms from every snare

    def f1(*pads):
        d = drum_scores(snares, events("estimate", [e for name, times in pads for e in hits(name, "snare", times)]))
        return d["sounds"]["snare"]["f1_20"], d["extra"]
    assert f1(("38", np.r_[SNARE, SNARE])) == (pytest.approx(2 / 3, abs=1e-4), [])
    assert f1(("38", np.r_[SNARE, SNARE, stray[:1]])) == (pytest.approx(32 / 49, abs=1e-4), [])
    one, two = f1(("38", np.r_[SNARE, SNARE, stray])), f1(("38", SNARE), ("40", np.r_[SNARE, stray]))
    assert one == two == (pytest.approx(4 / 7, abs=1e-4), [])
    many = f1(("38", np.r_[SNARE, 0.25 + np.arange(17) * 1.0]))  # 16 snares among 33 moments: extra
    assert many[0] == 0.0 and many[1][0]["moments"] == 33 and many[1][0]["share"] == pytest.approx(16 / 33, abs=1e-3)


def test_a_snare_pad_with_a_stray_hit_on_a_hat_stays_the_snare():
    """Every snare also sits on a hat. A snare pad with one more hit on a hat
    position matches the hat 17 times and the snare 16: assigned by matches
    it went to the hat and the snare scored 0; by F1 at 50 ms (0.97 against
    0.42) it is the snare's."""
    s = drum_scores(truth(), estimate(pads=(KIT[0], ("38", "snare", np.r_[SNARE, HAT[1]]), KIT[2])))
    assert s["assignment"]["38"]["to"] == "snare" and s["assignment"]["38"]["matched_50"] == 16
    assert s["sounds"]["snare"]["f1_20"] == pytest.approx(32 / 33, abs=1e-4) and s["sounds"]["hat"]["f1_20"] == 1.0


def test_a_missing_sound_scores_zero_and_a_layer_is_a_sound_of_its_own():
    """A clap layered on every snare: the snare pad ties on both (same hits,
    same F1) and goes to the sound of its own role; the clap is not played."""
    t = truth()
    t["drums"] += [dict(e, layer_of="snare") for e in hits("clap", "clap", SNARE)]
    t["sounds"]["clap"] = {"role": "clap"}
    s = drum_scores(t, estimate(pads=KIT[:2]))
    assert s["sounds"]["hat"]["f1_20"] == 0.0 and s["sounds"]["hat"]["estimated"] == []
    assert (s["sounds"]["snare"]["f1_20"], s["sounds"]["clap"]["f1_20"]) == (1.0, 0.0)
    assert s["sounds"]["clap"]["layer_of"] == "snare" and s["macro"]["f1_20"] == 0.5


def test_a_pad_that_plays_its_sounds_file_is_scored_for_pitch():
    """A plan's pads: the snare pad plays the truth snare's file at -1
    semitone, 30 cents off its -0.70 (Transpose without Detune), the hat pad
    its file at its +0.25, the kick pad another file (no pitch to compare).
    Two of three truth sounds are played by their own file, 30 and 0 cents
    off; with Detune +30 both are right. An analysis has no such pads."""
    from refdissect.suite import headline
    t, e = truth(), estimate()
    t["sounds"] |= {"kick": {"role": "kick", "sha256": "k", "semitones": 2.0, "cents": 0.0},
                    "snare": {"role": "snare", "sha256": "s", "semitones": -1.0, "cents": 30.0},
                    "hat": {"role": "hat", "sha256": "h", "semitones": 0.0, "cents": 25.0}}
    e["sounds"] |= {"36": {"role": "kick", "sha256": "x", "semitones": 2.0, "cents": 0.0},
                    "38": {"role": "snare", "sha256": "s", "semitones": -1.0, "cents": 0.0},
                    "42": {"role": "hat", "sha256": "h", "semitones": 0.0, "cents": 25.0}}
    d = drum_scores(t, e, files=True)
    assert d["pitch"] == {"same_file": 0.6667, "pads": 2, "cents_abs_mean": 15.0, "cents_abs_max": 30.0}
    assert d["sounds"]["snare"]["same_file_cents"] == {"38": -30.0} and d["sounds"]["kick"]["same_file_cents"] is None
    e["sounds"]["38"]["cents"] = 30.0
    e["generator"] = {"switches": {"from": "plan"}}
    h = headline(score(t, e))
    assert (h["drums.same_file"], h["drums.pitch_cents_abs_max"], h["drums.pitch_cents_abs_mean"]) == (0.6667, 0, 0)
    assert "pitch" not in drum_scores(t, e) and "drums.same_file" not in headline(score(t, estimate()))


def test_an_808_60_cents_off_has_its_onsets_and_none_of_its_notes():
    b = evaluate.bass_scores(truth(), estimate(midi=36.6))
    assert (b["f1_20"], b["note_f1_20"], b["note_f1_50"]) == (1.0, 0.0, 0.0)
    assert b["cents"]["median"] == pytest.approx(60.0, abs=0.01)
    assert evaluate.bass_scores(truth(), estimate(midi=36.4))["note_f1_50"] == 1.0


def test_an_808_heard_right_but_written_a_semitone_high_and_its_legato_notes():
    """bass_low-1's failure: the notes heard within 50 cents (they never
    settle), the settled pitch read 60 cents sharp and the tuning 29 cents
    flat, so every note is written a semitone high; the heard-pitch scores
    pass it, the settled and written ones do not. A legato note is right
    where its slide starts within 50 ms at the pitch it slides to."""
    ref, est = truth(), estimate()
    for k, (r, e) in enumerate(zip(ref["bass808"], est["bass808"])):
        r["midi_settled"] = 20.11 + (3 if k % 2 else 0)
        e["midi_settled"] = r["midi_settled"] + 0.597
    b = evaluate.bass_scores(ref, est)
    assert b["note_f1_50"] == 1.0 and b["settled"]["within_50"] == 0.0 and b["settled"]["written_equal"] == 0.0
    assert b["settled"]["tuning_cents"] == {"truth": 11.0, "est": -29.3}
    assert b["settled"]["median"] == pytest.approx(59.7, abs=0.01)
    for e in est["bass808"]:
        e["midi_settled"] -= 0.597
    assert evaluate.bass_scores(ref, est)["settled"]["written_equal"] == 1.0
    for e in est["bass808"]:  # a plan's heard Detune past half a semitone: its own MIDI notes are what it wrote
        e["midi_settled"] += 0.6
        e["pitch"] = round(e["midi_settled"] - 0.6 - 0.11)
    assert evaluate.bass_scores(ref, est)["settled"]["written_equal"] == 1.0
    for e in est["bass808"]:
        e["midi_settled"] -= 0.6
    assert evaluate.bass_scores(ref, est)["legato"] == {"n_truth": 0, "n_est": 0, "matched": 0, "precision": None,
                                                        "recall": None}
    for n in ref["bass808"][2:4] + est["bass808"][2:5]:
        n["flags"] = ["legato"]
    est["bass808"][3]["t"] += 0.04  # within 50 ms
    est["bass808"][4]["midi_settled"] += 1.0  # another pitch, and no truth legato note there
    assert evaluate.bass_scores(ref, est)["legato"] == {"n_truth": 2, "n_est": 3, "matched": 2,
                                                        "precision": 0.6667, "recall": 1.0}


def test_chord_accuracy_on_a_known_sequence():
    """8 s: the estimate gets every root, misses the 7th of A:min7 and G:7, and
    stops a second early (no chord there)."""
    ref = [{"t": 0.0, "end": 2.0, "label": "C:maj"}, {"t": 2.0, "end": 4.0, "label": "A:min"},
           {"t": 4.0, "end": 6.0, "label": "F:maj"}, {"t": 6.0, "end": 8.0, "label": "G:7"}]
    est = [{"t": 0.0, "end": 2.0, "label": "C:maj"}, {"t": 2.0, "end": 4.0, "label": "A:min7"},
           {"t": 4.0, "end": 6.0, "label": "F:maj"}, {"t": 6.0, "end": 7.0, "label": "G:maj"}]
    c = evaluate.chord_scores(events("truth", chords=ref), events("estimate", chords=est))
    # root, majmin, mirex: right for 7 of 8 s; sevenths, tetrads: C and F only, 4 of 8 s
    assert c == {"span_s": [0.0, 8.0], "root": 0.875, "majmin": 0.875, "sevenths": 0.5, "tetrads": 0.5,
                 "mirex": 0.875}


@pytest.mark.parametrize("ratio, name", [(1.0, "x1"), (1.03, "x1"), (2.0, "x2"), (0.5, "x1/2"), (1.5, "x3/2"),
                                         (2 / 3, "x2/3"), (1.2, "other"), (3.0, "other")])
def test_tempo_octave_classes(ratio, name):
    assert evaluate.octave(ratio) == name


def grid(bpm, first, bars, beats_per_bar=4):
    beats = (first + np.arange(bars * beats_per_bar) * 60 / bpm).tolist()
    return {"bpm": bpm, "beats_per_bar": beats_per_bar, "beat_unit": 4, "downbeat_s": first, "beats_s": beats,
            "downbeats_s": beats[::beats_per_bar], "swing": None, "tempo_map": None}


def test_rhythm_double_time_and_a_downbeat_one_bar_and_10_ms_late():
    bar = 4 * 60 / 136
    t = events("truth", time=grid(136.0, 0.1, 8))
    r = evaluate.rhythm_scores(t, events("estimate", time=grid(272.0, 0.1 + bar + 0.010, 8)))
    assert r["tempo"]["octave"] == "x2" and r["tempo"]["rel_error"] == pytest.approx(1.0)
    assert r["downbeat"]["first_error_ms"] == pytest.approx(10.0, abs=0.01)
    assert r["meter"]["equal"]
    same = evaluate.rhythm_scores(t, events("estimate", time=grid(136.0, 0.1, 9)))  # a bar more, outside the truth
    assert (same["downbeat"]["f"], same["beat_f"], same["tempo"]["octave"]) == (1.0, 1.0, "x1")
    t["time"]["swing"] = {"ratio": 0.6, "level": 16}
    swing = {"ratio": 0.55, "level": 16}
    swung = evaluate.rhythm_scores(t, events("estimate", time={**grid(136.0, 0.1, 8), "swing": swing}))
    assert swung["swing"]["ratio_error"] == -0.05 and swung["swing"]["level_equal"]


def test_a_6_8_bar_is_six_eighths_at_a_quarter_note_tempo():
    """bpm counts quarter notes in 6/8 too (beat_unit 8, beats in eighths), as
    the scenarios write it: a downbeat one bar (1.32 s at 136 BPM) and 10 ms
    late is 10 ms late. Six beats of 60/bpm would make the bar 2.65 s and
    read -1313.5 ms."""
    t = events("truth", time={**grid(272.0, 0.1, 8, 6), "bpm": 136.0, "beat_unit": 8})
    late = {**grid(272.0, 0.1 + 6 * 30 / 136 + 0.010, 8, 6), "bpm": 136.0, "beat_unit": 8}
    r = evaluate.rhythm_scores(t, events("estimate", time=late))
    assert r["downbeat"]["first_error_ms"] == pytest.approx(10.0, abs=0.01)
    assert r["meter"] == {"truth": [6, 8], "est": [6, 8], "equal": True} and r["tempo"]["octave"] == "x1"


def analysis_grid(**over):
    """An analysis.json grid as grid.estimate writes it (the keys scoring reads)."""
    return {"bpm": 136.0, "phase": 0.4, "beats_per_bar": 4, "beat_unit": 4, "tempo_map": None,
            "swing": {"ratio_16": 0.55, "ratio_8": None}, "downbeat": {"downbeats": 16, "ambiguous": False}} | over


def test_the_analysis_grid_scores_every_rhythm_number():
    """A grid read right (the full scenario's: first bar line at 0.4 s, 16ths
    swung 54 %, read 55 %) scores each rhythm number as right: the bars count
    from its first bar line, and the beat 41 ms before the audio is a pickup
    (within BAR_SLACK, as Grid.bar_of counts)."""
    from refdissect.suite import headline
    t = events("truth", time={**grid(136.0, 0.4, 16), "swing": {"ratio": 0.54, "level": 16}})
    tm = evaluate._analysis_time(analysis_grid(), 30.0)
    assert tm["beats_s"][:3] == [-0.04118, 0.4, 0.84118] and tm["downbeat_s"] == 0.4
    assert tm["downbeats_s"][:2] == pytest.approx([0.4, 0.4 + 240 / 136], abs=1e-5)
    s = score(t, events("estimate", time=tm))
    r = s["rhythm"]
    assert (r["tempo"]["octave"], r["meter"]["equal"], r["downbeat"], r["beat_f"]) == (
        "x1", True, {"first_error_ms": 0.0, "first_error_beats": 0.0, "unsure": False, "f": 1.0}, 1.0)
    assert r["swing"]["ratio_error"] == pytest.approx(0.01) and r["swing"]["level_equal"]
    assert {k: v for k, v in headline(s).items() if k.startswith("rhythm.")} == {
        "rhythm.meter_equal": 1.0, "rhythm.octave_x1": 1.0, "rhythm.tempo_error_abs": 0.0,
        "rhythm.downbeat_error_ms_abs": 0.0, "rhythm.downbeat_f": 1.0, "rhythm.downbeat_unsure": 0.0,
        "rhythm.beat_f": 1.0, "rhythm.swing_ratio_error": pytest.approx(0.01)}


def test_the_analysis_grid_in_6_8_with_a_pickup():
    """6/8 at 136 BPM, first bar line 0.9 s in: beats are eighths (0.22 s),
    the four before the bar line are beats too (a pickup), and bars are six
    of them (1.32 s)."""
    eighth = 30 / 136
    tm = evaluate._analysis_time(analysis_grid(phase=0.9, beats_per_bar=6, beat_unit=8), 6.0)
    assert tm["beats_s"][:5] == pytest.approx([0.9 - 4 * eighth, 0.9 - 3 * eighth, 0.9 - 2 * eighth,
                                               0.9 - eighth, 0.9], abs=1e-5)
    assert tm["downbeats_s"] == pytest.approx([0.9 + 6 * eighth * k for k in range(4)], abs=1e-5)
    assert (tm["downbeat_s"], tm["beats_per_bar"], tm["beat_unit"]) == (0.9, 6, 8)
    t = events("truth", time={**grid(272.0, 0.9, 4, 6), "bpm": 136.0, "beat_unit": 8})
    r = evaluate.rhythm_scores(t, events("estimate", time=tm))
    assert r["meter"]["equal"] and r["downbeat"]["first_error_ms"] == 0.0 and r["beat_f"] == 1.0


def test_the_analysis_grid_swing_flags_and_tempo_map():
    """Swing at the finer level read (16ths, else 8ths); a bar line without
    a tracker, or one its vote calls ambiguous, is flagged; a tempo map moves
    each beat by its offset there (an eighth halfway between two of its
    quarter notes)."""
    def time(**over):
        return evaluate._analysis_time(analysis_grid(**over), 8.0)
    assert time()["swing"] == {"ratio": 0.55, "level": 16} and time()["flags"] == []
    assert time(swing={"ratio_16": None, "ratio_8": 0.62})["swing"] == {"ratio": 0.62, "level": 8}
    assert time(swing={"ratio_16": None, "ratio_8": None})["swing"] is None
    assert time(downbeat=None)["flags"] == time(downbeat={"ambiguous": True})["flags"] == ["downbeat_unsure"]
    q = 60 / 136
    lines = 0.4 + np.arange(16) * q
    tempo_map = (lines + np.linspace(0.0, 0.03, 16)).round(4).tolist()  # 30 ms ahead by the end
    moved, still = time(tempo_map=tempo_map)["beats_s"], time()["beats_s"]
    assert moved[1:17] == pytest.approx(tempo_map, abs=1e-5) and moved[0] == still[0]  # the pickup: before the map
    assert moved[-1] == pytest.approx(still[-1] + 0.03, abs=1e-4)  # held after it (the map is in 0.1 ms)
    eighths = time(tempo_map=tempo_map, beats_per_bar=6, beat_unit=8)["beats_s"]
    j = eighths.index(0.4)
    assert eighths[j + 1] == pytest.approx((tempo_map[0] + tempo_map[1]) / 2, abs=1e-5)


def melodic_json(stems: dict, segments=None) -> dict:
    """melodic.json as pipeline.melodic writes it: {stem: (cents, [(t, end, pitch, voted)])}."""
    out = {"stems": {s: {"tuning_cents": c, "notes": [
        {"t": t, "end": e, "pitch": p, "cents": c, "velocity": 90, **({"voted": "restored"} if v else {})}
        for t, e, p, v in notes]} for s, (c, notes) in stems.items()}}
    if segments is not None:
        out["chords"] = {"segments": [{"start": a, "end": b, "start_time": 2 * a, "label": label, "bass_from": None}
                                      for a, b, label in segments]}
    return out


def test_melodic_json_scored_per_stem_with_the_other_stems_extra(tmp_path):
    """Truth layers pad and lead, both in the other stem (cents: its tuning,
    on top of Basic Pitch's semitone); the vocals stem's notes belong to a
    sound the truth lists none for: extra, not false notes (pooled with
    them, note F1 would be 0.667). Without melodic.json, neither notes nor
    chords are scored."""
    pad = [{"layer": "pad", "t": 2.0 * k, "end": 2.0 * k + 1.9, "midi": 60, "cents": 0.0, "velocity": 90,
            "flags": []} for k in range(4)]
    lead = [{**n, "layer": "lead", "t": n["t"] + 0.5, "end": n["t"] + 0.8, "midi": 72} for n in pad]
    chords = [{"t": 0.0, "end": 4.0, "label": "C:maj"}, {"t": 4.0, "end": 8.0, "label": "A:min"}]
    t = events("truth", notes=pad + lead, chords=chords)
    played = [(n["t"], n["end"], n["midi"], k == 3) for k, n in enumerate(pad + lead)]
    vocal = [(1.0 + k, 1.5 + k, 64, False) for k in range(8)]
    analysis = {"source": {"duration_s": 8.0}, "grid": {"bpm": 120.0, "phase": 0.0}, "templates": {}, "events": {},
                "notes_808": []}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    (tmp_path / "melodic.json").write_text(json.dumps(melodic_json(
        {"other": (20, played), "vocals": (-5, vocal)}, [(0.0, 4.0, "C:maj"), (4.0, 8.0, "A:min")])))
    est = validate(evaluate.estimate_from_analysis(tmp_path))
    first = est["notes"][0]
    assert (first["layer"], first["midi"], first["cents"], first["velocity"]) == ("other", 60, 20.0, 90)
    assert [n["flags"] for n in est["notes"] if n["layer"] == "other"].count(["restored"]) == 1
    s = score(t, est)
    m = s["melodic"]
    assert {k: v["estimated_layer"] for k, v in m["layers"].items()} == {"lead": "other", "pad": "other"}
    assert m["pooled"]["note_f1"] == 1.0 and m["extra"] == [{"layer": "vocals", "notes": 8}]
    assert evaluate._note_block(pad + lead, est["notes"])["note_f1"] == pytest.approx(0.6667, abs=1e-4)
    assert s["chords"]["majmin"] == 1.0
    (tmp_path / "melodic.json").unlink()
    bare = score(t, evaluate.estimate_from_analysis(tmp_path))
    assert bare["melodic"] is None and bare["chords"] is None


def test_the_plans_melodic_tracks_are_its_notes():
    """The plan's melodic notes at their beats (120 BPM) and the track's
    cents; a plan holds no chords, so they are not scored at plan level."""
    notes = [{"pitch": 60, "start_time": 4.0 * k, "duration": 3.8, "velocity": 90} for k in range(4)]
    plan = {"tempo": 120.0, "length_beats": 16.0,
            "tracks": {"drums": {"pads": {}, "notes": []}, "808": {"notes": []},
                       "melodic": {"other": {"instrument": "Drift", "cents": -12, "notes": notes}}}}
    est = validate(evaluate.estimate_from_plan(plan))
    assert [(n["t"], n["end"], n["midi"], n["cents"]) for n in est["notes"][:2]] == [(0.0, 1.9, 60, -12.0),
                                                                                       (2.0, 3.9, 60, -12.0)]
    t = events("truth", notes=[{"layer": "pad", "t": 2.0 * k, "end": 2.0 * k + 1.9, "midi": 60, "cents": 0.0,
                                "velocity": 90, "flags": []} for k in range(4)],
               chords=[{"t": 0.0, "end": 8.0, "label": "C:maj"}])
    s = score(t, est)
    assert s["melodic"]["pooled"]["note_f1"] == 1.0 and s["chords"] is None and "chords" not in est
    del plan["tracks"]["melodic"]
    assert "notes" not in evaluate.estimate_from_plan(plan)


def test_validate_names_the_problems():
    assert validate(truth()) is not None
    bad = truth() | {"schema": "other"}
    bad["drums"][0] = {"sound": "kick", "t": "0.0"}
    bad["chords"] = [{"t": 0.0, "end": 1.0, "label": "C:nonsense"}]
    with pytest.raises(ValueError) as e:
        validate(bad)
    for problem in ("schema", "drums[0]: no t_attack", "drums[0]: t is not a number", "chords[0]"):
        assert problem in str(e.value)
    with pytest.raises(ValueError, match="not in sounds"):
        validate(events("truth", hits("kick", "kick", KICK)) | {"sounds": {}})


def test_estimate_from_analysis_reads_the_attack_in_the_roles_band(kit, tmp_path):
    """A template whose snare starts 20 ms after its window's full-band
    attack (the tail of a kick before it): its hit at t attacks at
    t - PRE + 25 ms. The role comes from matches.json, not the guess."""
    template = np.r_[np.zeros(PRE + samples(20), np.float32), kit["snare"]]
    (tmp_path / "templates").mkdir()
    np.save(tmp_path / "templates" / "sound_1.npy", template)
    analysis = {"source": {"duration_s": 4.0}, "grid": {"bpm": 120.0, "phase": 0.01, "beats_per_bar": 4},
                "templates": {"sound_1": {"role_guess": "kick"}},
                "events": {"sound_1": [{"t": 1.0, "ncc": 0.9, "gain": 1.0}]},
                "notes_808": [{"t": 1.0, "end": 1.5, "midi_float": 24.33, "slide": -2.0, "peak": 0.5, "pitch": 24}]}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    (tmp_path / "matches.json").write_text(json.dumps({"sound_1": {"role": "snare"}}))
    est = validate(evaluate.estimate_from_analysis(tmp_path))
    hit = est["drums"][0]
    lead = attack_time(template, "snare") - PRE / SR
    assert hit["role"] == "snare" and hit["t_attack"] == pytest.approx(1.0 + lead, abs=1e-5)
    assert abs(lead - 0.020) < 0.001
    note = est["bass808"][0]
    assert (note["midi"], note["flags"], note["with_kick"]) == (24.33, ["slide:-2.0"], False)
    assert est["time"]["downbeats_s"][:2] == [0.01, 2.01] and len(est["time"]["beats_s"]) == 8


def test_estimate_from_analysis_takes_the_templates_own_attack(kit, tmp_path):
    """A template that says where its sound starts (attack_s, as discover()
    reads it on all its hits) times hits that carry no t_attack there, not
    where attack_time reads its averaged audio (20 ms later here): an
    analysis from a detect() that writes only t keeps discover()'s timing."""
    template = np.r_[np.zeros(PRE + samples(20), np.float32), kit["snare"]]
    (tmp_path / "templates").mkdir()
    np.save(tmp_path / "templates" / "sound_1.npy", template)
    analysis = {"source": {"duration_s": 4.0}, "grid": {"bpm": 120.0, "phase": 0.01, "beats_per_bar": 4},
                "templates": {"sound_1": {"role_guess": "snare", "attack_s": PRE / SR + 0.003}},
                "events": {"sound_1": [{"t": 1.0, "ncc": 0.9, "gain": 1.0}, {"t": 2.0, "t_attack": 2.001}]},
                "notes_808": []}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    est = validate(evaluate.estimate_from_analysis(tmp_path))
    assert [e["t_attack"] for e in est["drums"]] == [pytest.approx(1.003, abs=1e-5), 2.001]
    assert est["sounds"]["sound_1"]["attack_s"] == pytest.approx(0.003, abs=1e-5)


def test_analysis_times_land_on_the_attacks_of_a_synthetic_beat(beat, kit, tmp_path):
    """detect() reports a hit PRE (5 ms) after its template window starts: with
    the template's own attack added, the snares of the synthetic beat land
    within 1 ms of where they attack (+5 ms without the -PRE)."""
    from refdissect.detect import detect, discover
    from refdissect.grid import fit_grid
    g, _ = fit_grid(beat["audio"], (100, 180))
    templates = discover(beat["audio"], g)
    found = detect(beat["audio"], templates)
    (tmp_path / "templates").mkdir()
    for tp in templates:
        np.save(tmp_path / "templates" / f"{tp.name}.npy", tp.audio)
    analysis = {"source": {"duration_s": len(beat["audio"]) / SR}, "grid": g.to_dict(),
                "templates": {tp.name: {"role_guess": tp.role} for tp in templates}, "events": found,
                "notes_808": []}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    t = events("truth", [e for name, times in beat["truth"].items()
                         for e in hits(name, name, times, attack_time(kit[name], name))])
    snare = drum_scores(t, evaluate.estimate_from_analysis(tmp_path))["sounds"]["snare"]
    assert snare["f1_20"] > 0.95 and abs(snare["error_ms"]["median"]) < 1.0
