"""Melodic notes and chords on inputs whose answer is known: synthetic tones,
hand-made posteriorgrams and fake worker output (no model)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from refdissect import melodic as M
from refdissect.audio import SR
from refdissect.basicpitch import block_times
from refdissect.grid import Grid


def tone(midi: float, seconds: float, gain: float = 0.5) -> np.ndarray:
    """A plucked harmonic tone (four partials, 2 ms attack, exponential decay)."""
    t = np.arange(int(seconds * SR)) / SR
    f = 440 * 2 ** ((midi - 69) / 12)
    x = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, 5))
    return (gain * x * np.minimum(1, t / 0.002) * np.exp(-t / 0.25)).astype(np.float32)


def render(notes: list[tuple[float, int, float]], seconds: float) -> np.ndarray:
    x = np.zeros(int(seconds * SR), np.float32)
    for t, midi, gain in notes:
        y = tone(midi, 0.4, gain)
        a = int(t * SR)
        x[a:a + len(y)] += y[: len(x) - a]
    return x


def test_floor_above_808():
    notes = [{"pitch": p} for p in (28, 28, 31, 31, 39)]  # 39 heard once: a pitch-tracking slip
    hz, top = M.floor_above_808(notes)
    assert top == 31 and hz == pytest.approx(440 * 2 ** ((32 - 69) / 12), abs=0.01)
    assert M.floor_above_808([]) == (None, None)


def test_seconds_to_beats_on_a_known_grid():
    g = Grid(120.0, 0.5)  # a beat is 0.5 s, the first beat at 0.5 s
    assert M.to_beats(1.5, g) == (3.0, 0.0)  # from the start of the audio, like the drum notes
    late = 0.5 + 3 * 0.125 + 0.004  # 4 ms after the fourth 16th
    beats, moved = M.to_beats(late, g, snap_ms=10)
    assert beats == pytest.approx(1.75) and moved == pytest.approx(-4.0)
    assert M.to_beats(late, g, snap_ms=3) == (pytest.approx(2 * late), 0.0)
    assert M.to_beats(0.5 + 0.25 + 0.01, g, snap_ms=20, per_beat=2)[0] == pytest.approx(1.5)


def test_levels_follow_the_notes_and_give_velocities():
    gains = [0.05, 0.1, 0.2, 0.4, 0.8]
    x = render([(0.5 + 0.6 * k, 64, g) for k, g in enumerate(gains)], 4.0)
    notes = [{"t": 0.5 + 0.6 * k, "pitch": 64} for k in range(len(gains))]
    levels = M.note_levels(x, notes)
    assert np.all(np.diff(levels) > 0)
    assert np.diff(levels) == pytest.approx([6.02] * 4, abs=1.0)  # each note twice as loud
    v = M.velocities(list(10 ** (levels / 20)))
    assert v == sorted(v) and v[-1] == 127


def test_stem_notes_from_fake_worker_output():
    g = Grid(120.0, 0.25)
    raw = [{"start_s": 0.25, "end_s": 0.6, "midi": 60, "amplitude": 0.4, "onset": 0.9, "bend": [0, 0, 1]},
           {"start_s": 0.75, "end_s": 1.1, "midi": 67, "amplitude": 0.7, "onset": 0.8, "bend": [0, 0, 0]}]
    x = render([(0.25, 60, 0.2), (0.75, 67, 0.6)], 2.0)
    notes, report = M.stem_notes(raw, x, g, tuning=0.12)
    assert report == [] and [n["pitch"] for n in notes] == [60, 67]
    first, second = notes
    assert first["cents"] == 12 and first["start_time"] == 0.5 and second["start_time"] == 1.5
    assert first["duration"] == pytest.approx(0.7) and first["bend"] == [0, 0, 1] and "bend" not in second
    assert second["velocity"] > first["velocity"]  # from the measured level, not Basic Pitch's amplitude


# ── Loop voting ──────────────────────────────────────────────────────────────

PATTERN = ((0.0, 60), (0.5, 64), (1.0, 67), (1.5, 72))  # one bar of 2 s at 120 BPM


def note(t, pitch, onset=0.8):
    return {"t": t, "end": t + 0.3, "pitch": pitch, "amplitude": 0.5, "onset": onset, "bend": []}


def toy_loop():
    """A 1-bar loop played four times: bar 3 misses its third note, bar 2 has
    a weak extra note (bleed), bar 4 a clearly attacked one (a variation)."""
    notes = [note(2.0 * b + o, p) for b in range(4) for o, p in PATTERN if not (b == 2 and p == 67)]
    notes += [note(2.0 + 0.75, 61, onset=0.62), note(6.0 + 0.25, 70, onset=0.9)]
    return Grid(120.0, 0.0), notes, [{"name": "A", "start_bar": 1, "end_bar": 4}]


def test_loop_groups_find_the_period_and_nothing_where_bars_differ():
    g, notes, sections = toy_loop()
    by_bar = M.bar_notes(notes, g)
    groups = M.loop_groups(sections, lambda a, b: M.note_similarity(by_bar.get(a, []), by_bar.get(b, []), a, b, g))
    assert [(x["period_bars"], x["positions"]) for x in groups] == [(1, [[0, 1, 2, 3]])]
    rng = np.random.default_rng(0)
    random_bars = [note(2.0 * b + o, int(rng.integers(48, 84))) for b in range(4) for o, _ in PATTERN]
    by_bar = M.bar_notes(random_bars, g)
    assert M.loop_groups(sections, lambda a, b: M.note_similarity(by_bar.get(a, []), by_bar.get(b, []), a, b, g)) == []


def test_vote_restores_the_majority_and_drops_only_on_request():
    g, notes, sections = toy_loop()
    by_bar = M.bar_notes(notes, g)
    groups = M.loop_groups(sections, lambda a, b: M.note_similarity(by_bar.get(a, []), by_bar.get(b, []), a, b, g))
    voted, report = M.vote(notes, g, groups)
    restored = [n for n in voted if n.get("voted") == "restored"]
    assert [(n["t"], n["pitch"]) for n in restored] == [(5.0, 67)]
    assert len(voted) == len(notes) + 1 and report[0]["restored"] == 1 and report[0]["dropped"] == 0
    strict, report = M.vote(notes, g, groups, drop_below=0.7)
    assert report[0]["dropped"] == 1 and not any(n["pitch"] == 61 for n in strict)
    assert any(n["pitch"] == 70 for n in strict)  # the clearly attacked variation stays


def test_stem_notes_vote_and_drop_on_request():
    g, notes, sections = toy_loop()
    raw = [{"start_s": n["t"], "end_s": n["end"], "midi": n["pitch"], "amplitude": 0.5, "onset": n["onset"],
            "bend": []} for n in notes]
    x = np.zeros(9 * SR, np.float32)
    kept, report = M.stem_notes(raw, x, g, sections=sections)
    assert report[0]["restored"] == 1 and report[0]["dropped"] == 0 and len(kept) == len(raw) + 1
    assert next(n for n in kept if n.get("voted") == "restored")["start_time"] == 10.0  # beat 10, bar 3
    dropped, report = M.stem_notes(raw, x, g, sections=sections, drop_below=0.7)
    assert report[0]["dropped"] == 1 and len(dropped) == len(raw)
    assert M.stem_notes(raw, x, g, sections=sections, voting=False)[1] == []


def test_vote_keeps_a_same_pitch_roll_apart():
    g = Grid(120.0, 0.0)
    notes = [note(2.0 * b + o, 60) for b in range(4) for o in (0.0, 0.0625)]  # two 32nds at 120 BPM
    voted, report = M.vote(notes, g, [{"section": "A", "period_bars": 1, "similarity": 1.0,
                                       "positions": [[0, 1, 2, 3]]}])
    assert len(voted) == 8 and report[0]["restored"] == 0


def test_no_note_before_the_audio_starts(tmp_path):
    """The first beat 4 ms in, downbeats played 8 ms early and the first one
    missed: restored, it would sit at -4 ms; snapped, a note at 3 ms would go
    to the 16th line at -20 ms. Both stay at or after 0 and the MIDI file is
    written (a negative time made mido raise)."""
    from refdissect.live import write_midi
    g = Grid(120.0, 0.004)
    raw = [{"start_s": g.phase + 2.0 * b + o, "end_s": g.phase + 2.0 * b + o + 0.3, "midi": p, "amplitude": 0.5,
            "onset": 0.8, "bend": []} for b in range(4) for o, p in ((-0.008, 60), (0.5, 64), (1.0, 67))
           if not (b == 0 and p == 60)]
    notes, report = M.stem_notes(raw, np.zeros(9 * SR, np.float32), g,
                                 sections=[{"name": "A", "start_bar": 1, "end_bar": 4}])
    assert report[0]["restored"] == 1 and notes[0]["t"] == 0.0 and notes[0]["start_time"] == 0.0
    late = Grid(120.0, 0.105)  # 16th lines at -0.02, 0.105, 0.23 ...
    assert M.to_beats(0.003, late, snap_ms=25) == (pytest.approx(0.006), 0.0)
    plan = {"tempo": 120.0, "tracks": {"drums": {"notes": []}, "808": {"notes": []},
                                       "melodic": {"other": {"notes": [{k: n[k] for k in ("pitch", "start_time",
                                                                                          "duration", "velocity")}
                                                                       for n in notes]}}}}
    assert (tmp_path / "other.mid") in write_midi(plan, tmp_path)


# ── Chords ───────────────────────────────────────────────────────────────────

def posteriorgram(spans: list[tuple[float, float, tuple[int, ...]]], seconds: float) -> np.ndarray:
    """A note posteriorgram (frames on Basic Pitch's block times) with these
    pitches sounding (posterior 0.9) over these spans."""
    n = int(seconds * 86.6) + 2
    t = block_times(n)
    pg = np.zeros((n, 88), np.float32)
    for a, b, pitches in spans:
        for p in pitches:
            pg[(t >= a) & (t < b), p - 21] = 0.9
    return pg


CHART = [  # (start, end, chord tones, 808 note or None, expected label); 120 BPM, a bar is 2 s
    (0.0, 2.0, (60, 64, 67), 24, "C:maj"),
    (2.0, 3.0, (60, 64, 67), 28, "C:maj/3"),         # the 808 on E: first inversion
    (3.0, 4.0, (53, 57, 60, 64), 29, "F:maj7"),       # a change on beat 3
    (4.0, 4.75, (55, 60, 62), 31, "G:sus4"),
    (4.75, 6.0, (55, 59, 62, 65), 35, "G:7/3"),       # a change on the "and" of 2, the 808 on B
    (6.0, 8.0, (52, 55, 60), None, "C:maj/3"),        # no 808: the lowest chord tone (E3) names the bass
]


def test_chords_with_inversions_and_changes_inside_the_bar():
    pg = posteriorgram([(a, b, tones) for a, b, tones, _, _ in CHART], 9.0)
    n808 = [{"t": a, "end": b, "pitch": p} for a, b, _, p, _ in CHART if p is not None]
    out = M.chords([pg], Grid(120.0, 0.0), 9.0, n808, lowest_midi=36)
    got = [(s["start"], s["label"], s["bass_from"]) for s in out["segments"]]
    want = [(a, label, "808" if p is not None else "melodic") for a, _, _, p, label in CHART] + [(8.0, "N", None)]
    assert got == want
    assert out["segments"][3]["start_time"] == 8.0  # beats


def test_viterbi_change_penalty_smooths_a_one_span_blip():
    scores = np.zeros((9, 3))
    scores[:, 0] = 1.0
    scores[4, 1] = 1.1
    assert M.viterbi(scores, 0.15).tolist() == [0] * 9
    assert M.viterbi(scores, 0.01).tolist()[4] == 1


def test_chord_vote_across_loop_repeats():
    """One repeat of a 1-bar loop played eight times has its C buried under an
    F (bleed): averaged with the other seven, it reads A minor like them."""
    spans = [(2.0 * b, 2.0 * b + 2.0, (57, 60, 64) if b != 5 else (57, 64, 65)) for b in range(8)]
    pg = posteriorgram(spans, 16.0)
    n808 = [{"t": 2.0 * b, "end": 2.0 * b + 2.0, "pitch": 33} for b in range(8)]
    sections = [{"name": "A", "start_bar": 1, "end_bar": 8}]
    alone = M.chords([pg], Grid(120.0, 0.0), 16.0, n808, lowest_midi=36)
    voted = M.chords([pg], Grid(120.0, 0.0), 16.0, n808, lowest_midi=36, sections=sections)
    assert [s["label"] for s in alone["segments"]] == ["A:min", "F:maj7/3", "A:min"]
    assert [s["label"] for s in voted["segments"]] == ["A:min"] and voted["loops"][0]["period_bars"] == 1


def test_harte_labels():
    assert M.harte(9, "min7", 7) == "A:min7/b7"
    assert M.harte(0, "maj", 0) == "C:maj" and M.harte(0, "maj(9)", None) == "C:maj(9)"
    assert M.harte(7, "7", 11) == "G:7/3"


# ── The step and the plan, with the worker faked ─────────────────────────────

def fake_transcribe(audio, out_dir, settings):
    """What the worker writes, for one bar of a C major arpeggio per stem."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"stems": {}, "model": "basic-pitch 0.4.0", "backend": "FAKE", "python": "3.11", "settings": settings}
    for path in audio:
        name = Path(path).stem
        notes = [{"start_s": 0.5 + 0.5 * k, "end_s": 0.9 + 0.5 * k, "start_frame": 0, "end_frame": 1, "midi": m,
                  "amplitude": 0.5, "onset": 0.9, "bend": [0]} for k, m in enumerate((60, 64, 67))]
        (out_dir / f"{name}.json").write_text(json.dumps({"stem": name, "notes": {"melodic": notes, "chord": notes}}))
        np.savez_compressed(out_dir / f"{name}.npz", note=posteriorgram([(0.0, 3.0, (60, 64, 67))], 3.0))
        result["stems"][name] = {"notes": str(out_dir / f"{name}.json"), "posteriorgram": str(out_dir / f"{name}.npz")}
    return result


def test_melodic_step_and_the_plan(monkeypatch, tmp_path):
    from refdissect import basicpitch, pipeline
    from refdissect.live import build_spec
    stems = tmp_path / "stems"
    stems.mkdir()
    x = render([(0.5, 60, 0.3), (1.0, 64, 0.3), (1.5, 67, 0.3)], 3.0)
    for name in ("other", "piano"):
        sf.write(stems / f"{name}.wav", x, SR)
    for name in ("drums", "bass"):
        sf.write(stems / f"{name}.wav", np.zeros_like(x), SR)
    analysis = {"source": {"stems": str(stems), "duration_s": 3.0}, "grid": {"bpm": 120.0, "phase": 0.5},
                "tuning": {"per_stem": {"other": -0.03}, "consensus": 0.0}, "templates": {}, "events": {},
                "notes_808": [{"t": 0.5, "end": 2.5, "pitch": 36, "peak": 1.0}] * 2, "sections": []}
    (tmp_path / "analysis.json").write_text(json.dumps(analysis))
    monkeypatch.setattr(basicpitch, "transcribe", fake_transcribe)
    out = pipeline.melodic(tmp_path, log=lambda *a: None)
    assert out["source"]["settings"]["melodic"]["min_hz"] == pytest.approx(69.3, abs=0.01)  # above the 808's C2
    assert set(out["stems"]) == {"other", "piano"} and out["stems"]["other"]["tuning_cents"] == -3
    assert [n["start_time"] for n in out["stems"]["other"]["notes"]] == [1.0, 2.0, 3.0]
    assert [c["label"] for c in out["chords"]["segments"]] == ["C:maj"]  # the 808 (C2) is the bass
    plan = pipeline.plan(tmp_path, log=lambda *a: None)
    track = plan["tracks"]["melodic"]["other"]
    assert len(track["notes"]) == 3 and track["cents"] == -3
    spec = {t["name"]: t for t in build_spec(plan)}
    assert spec["other (notes)"]["mute"] is True and spec["other (notes)"]["clips"][0]["notes"] == track["notes"]
    assert (tmp_path / "midi" / "other.mid").is_file()
