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
    assert s["extra"] == [{"sound": "50", "closest": None, "hits": 32, "matched_50": 0, "share": 0.0},
                          {"sound": "51", "closest": "kick", "hits": 8, "matched_50": 3, "share": 0.375}]
    assert s["macro"] == {"f1_20": 1.0, "f1_50": 1.0, "v1_f1": 1.0}


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


def test_an_808_60_cents_off_has_its_onsets_and_none_of_its_notes():
    b = evaluate.bass_scores(truth(), estimate(midi=36.6))
    assert (b["f1_20"], b["note_f1_20"], b["note_f1_50"]) == (1.0, 0.0, 0.0)
    assert b["cents"]["median"] == pytest.approx(60.0, abs=0.01)
    assert evaluate.bass_scores(truth(), estimate(midi=36.4))["note_f1_50"] == 1.0


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
