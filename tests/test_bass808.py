"""The 808 transcription on synthetic lines whose answer is known: an 808
voice (synth808.voice) played as a sampler plays it (pitch and length
change together), each note cut by the next with a 5 ms fade, kicks on half
the notes in the drum stem and leaking into the bass stem, as a separator
leaves them."""

import numpy as np
import pytest

from refdissect import bass808 as B
from refdissect.audio import SR, attack_time, samples
from refdissect.synth808 import Params, voice


def kick(seconds: float = 0.12) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    y = np.sin(2 * np.pi * np.cumsum(55 + 90 * np.exp(-t / 0.01)) / SR) * np.exp(-t / 0.04)
    y[:samples(2)] += np.random.default_rng(0).standard_normal(samples(2)) * 0.3
    return y.astype(np.float32)


def line(notes, root_hz: float = 32.70, cents: float = 0.0, bpm: float = 140.0, bars: int = 4, leak: float = 0.15,
         drop: float = 5.0, glide_ms: float = 50.0, slide_ms: float = 90.0, seed: int = 1
         ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """A bass stem, a drum stem and the truth of `bars` bars of notes (16th
    step, length in 16ths, semitones over the root, and "legato" for a note
    that slides in from the one before over slide_ms without starting the
    sample again), the 808 settling at root_hz + cents and starting `drop`
    semitones over it; a kick on every other attacked note, leaking into the
    bass stem at `leak`."""
    f = root_hz * 2 ** (cents / 1200)
    sample = voice(Params(f, f * 2 ** (drop / 12), glide_ms, 1.5, 3, 2.0, 0.05, 0.3), 2.5)
    step = 60 / bpm / 4
    n = int((bars * 16 * step + 1.5) * SR)
    bass, drums = np.zeros(n, np.float32), np.zeros(n, np.float32)
    truth, k = [], kick()
    chains: list[list[tuple]] = []
    for b in range(bars):
        for s, length, semis, *kind in notes:
            note = (int(round((b * 16 + s) * step * SR)), int(round(length * step * SR)), semis)
            if kind and kind[0] == "legato" and chains:
                chains[-1].append(note)
            else:
                chains.append([note])
    for i, chain in enumerate(chains):
        a = chain[0][0]
        length = chain[-1][0] + chain[-1][1] - a
        semis = np.full(length, float(chain[0][2]))
        for at, _, st in chain[1:]:
            o = at - a
            ramp = np.minimum(np.arange(1, length - o + 1) / samples(slide_ms), 1)
            semis[o:] = semis[o - 1] + (st - semis[o - 1]) * ramp
        y = np.interp(np.r_[0.0, np.cumsum(2 ** (semis[:-1] / 12))], np.arange(len(sample)), sample).astype(np.float32)
        y[-samples(5):] *= np.linspace(1, 0, samples(5))
        bass[a:a + len(y)] = y  # mono: the next chain cuts this one
        if i % 2 == 0:
            drums[a:a + len(k)] += k
            bass[a:a + len(k)] += leak * k
        for j, (at, _, st) in enumerate(chain):
            truth.append({"t": at / SR, "t_attack": (at if j else a + attack_time(y, "808") * SR) / SR,
                          "midi": 69 + 12 * np.log2(f / 440) + st, "legato": bool(j)})
    noise = np.random.default_rng(seed).standard_normal((2, n)).astype(np.float32) * 1e-3
    return bass + noise[0], drums + noise[1], truth


def found(truth: list[dict], notes: list[dict], window: float = 0.02, cents: float = 50.0) -> float:
    """The share of the truth's notes with an estimated note attacking within
    `window` whose settled pitch is within `cents`."""
    t = np.array([n["t_attack"] for n in notes])
    hit = 0
    for n in truth:
        j = int(np.argmin(np.abs(t - n["t_attack"])))
        hit += abs(t[j] - n["t_attack"]) <= window and abs(notes[j]["midi_settled"] - n["midi"]) * 100 <= cents
    return hit / len(truth)


# a bar of 16ths: retriggers at one pitch, 16th notes, a fifth up, a fourth up, a tone down
BAR = [(0, 4, 0), (4, 2, 0), (6, 1, 7), (7, 1, 7), (8, 4, 0), (12, 2, 5), (14, 2, -2)]


def test_qifft_reads_a_low_driven_tone_to_a_tenth_of_a_cent():
    for hz in (21.83, 32.92, 49.0):
        p = Params(hz, hz, 10, 4.0, 2, 2.0, 0.05, 0.3)
        y = voice(p, 0.7)[samples(300):samples(600)]
        assert 1200 * abs(np.log2(B.qifft(y, hz * 0.8, hz * 1.25) / hz)) < 0.1


def test_settled_pitch_of_a_drop_that_has_not_settled():
    """A 330 ms template of an 808 that falls 8 semitones with a 120 ms time
    constant is still 64 cents sharp at its end; the fitted drop's
    asymptote is within 3 cents."""
    hz = 32.92
    y = voice(Params(hz, hz * 2 ** (8 / 12), 120, 1.5, 3, 2.0, 0.05, 0.3), 0.33)
    assert abs(1200 * np.log2(B._settled_hz(y, 36.0) / hz)) < 3


def test_reattacks_find_a_cut_and_a_note_after_a_rest():
    bass, _, truth = line([(0, 4, 0), (4, 4, 0), (12, 4, 7)], bars=1, leak=0.0)
    onsets = np.array([o["i"] for o in B.reattacks(bass)]) / SR
    for n in truth:
        assert np.min(np.abs(onsets - n["t"])) < 0.003


def test_drum_attacks_and_notes_found_late_against_them():
    """The kicks of a drum stem, each at the foot of its rise (2.7 ms late
    here, the 5 ms RMS rising to a tenth of the kick; the one at the file's
    first sample has nothing before it, 15 ms); notes found 60 ms after them
    are late by that, notes 0-20 ms after them (an 808 that starts with its
    kick or just after) are not."""
    _, drums, truth = line(BAR, bars=2)
    kicks = B.drum_attacks(B.rms_envelope(B.band(drums, "low")))
    played = [n["t"] for i, n in enumerate(truth) if i % 2 == 0]
    assert len(kicks) == len(played) and np.max(np.abs(kicks[1:] / SR - played[1:])) < 0.004
    late = [int((t + 0.060) * SR) for t in played]
    assert abs(B._late(late, kicks) / SR - 0.060) < 0.003
    assert B._late([int((t + d) * SR) for t, d in zip(played, np.linspace(0, 0.02, len(played)))], kicks) == 0


def test_a_line_with_retriggers_sixteenths_and_kicks():
    """Every note's attack within 20 ms and its pitch within 50 cents (28 of
    28 measured); the 808's own tuning (+11 cents) within 3 (+11.6); the
    written pitches whole notes under it."""
    bass, drums, truth = line(BAR, cents=11.0)
    b = B.transcribe(bass, drums)
    assert found(truth, b["notes"]) >= 0.9
    assert abs(b["template"]["tuning_cents"] - 11.0) < 3
    assert {n["pitch"] for n in b["notes"] if abs(n["cents"] - 11.0) < 25} >= {24, 29, 31, 22}


def test_a_low_root_with_a_flat_808():
    """A root of 21.8 Hz (F0, under every learned tracker's floor) 20 cents
    flat: the notes (28 of 28 measured), and the tuning within 5 cents
    (-18.7)."""
    bass, drums, truth = line(BAR, root_hz=21.83, cents=-20.0)
    b = B.transcribe(bass, drums)
    assert found(truth, b["notes"]) >= 0.9
    assert abs(b["template"]["tuning_cents"] + 20.0) < 5


def test_a_slide_without_an_attack_is_a_legato_note():
    """A note that slides in over 120 ms from the one before (no attack: the
    sample goes on) is a legato note starting within 50 ms of where the
    slide starts, at its pitch, gliding linearly; not a few short restarts.
    Measured on this line: 4 of 4 slides, 24 notes for 24 (with 60 ms
    slides, none of 4 and 28 notes: the notes either side keep the slide)."""
    notes = [(0, 3, 0), (3, 1, 0), (4, 4, 5, "legato"), (8, 2, 0), (10, 2, 7), (12, 4, 0)]
    bass, drums, truth = line(notes, cents=11.0, slide_ms=120.0)
    b = B.transcribe(bass, drums)
    legato = [n for n in b["notes"] if "legato" in n["flags"]]
    slides = [m for m in truth if m["legato"]]
    right = [m for m in slides if any(abs(n["t"] - m["t"]) <= 0.05 and abs(n["midi_settled"] - m["midi"]) < 0.5
                                      for n in legato)]
    assert len(right) >= 3 and len(b["notes"]) <= 1.2 * len(truth)
    assert all(n["t"] == n["t_attack"] and n["glide"]["shape"] == "lin" for n in legato)


def test_tune_808_splits_a_fraction_into_transpose_and_detune():
    """Simpler plays a sample at its own pitch on C3 (MIDI 60): a sample
    sounding at MIDI 23.746 (32.228 Hz) needs Transpose +36 and Detune +25
    for MIDI 24 to sound at 24; one at the reference's own tuning, none."""
    from refdissect.live import tune_808
    assert tune_808(23.746, 0.0) == (36, 25)
    assert tune_808(24.117, 11.7) == (36, 0)
    assert tune_808(24.4, -20.0) == (35, 40)
    semis, detune = tune_808(30.62, 0.0)
    assert abs(30.62 + (24 - 60) + semis + detune / 100 - 24) <= 0.005


def test_plan_808_writes_the_notes_where_they_attack(tmp_path):
    """The plan's notes are the written pitches at their own lengths, each
    moved so that the one-shot, played at the note's pitch, attacks where the
    analysis heard the note attack; estimate_from_plan then hears the 808
    at the written pitch plus the reference's tuning."""
    import soundfile as sf

    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import plan_808
    hz = 32.70 * 2 ** (11.7 / 1200)
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, voice(Params(hz, hz * 2, 40, 1.2, 4, 2.0, 0.05, 0.3), 2.0), SR, subtype="FLOAT")
    analysis = {"tuning": {"808": 0.117},
                "notes_808": [{"t": 1.0, "t_attack": 1.006, "end": 1.4, "pitch": 24, "peak": 1.0},
                              {"t": 1.4, "t_attack": 1.405, "end": 1.5, "pitch": 31, "peak": 0.5}]}
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}
    track = plan_808(analysis, synth, tmp_path / "stems", 2.0)
    assert (track["transpose"], track["detune_cents"], track["tuning_cents"]) == (36, 0, 11.7)
    plan = {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": {}, "notes": []}, "808": track},
            "levels": {}}
    heard = estimate_from_plan(plan)["bass808"]
    assert [n["t_attack"] for n in heard] == pytest.approx([1.006, 1.405], abs=0.0005)
    own = 69 + 12 * np.log2(hz / 440)  # C1 + 11.7 cents
    assert [n["midi_settled"] for n in heard] == pytest.approx([own, own + 7], abs=0.002)
    for n, h, p in zip(analysis["notes_808"], heard, track["notes"]):  # until the note ends: 0.2 beats, no floor
        assert p["duration"] == pytest.approx(2 * (n["end"] - h["t"]), abs=0.001)


def test_plan_808_detunes_the_one_shot_to_the_pitch_heard(tmp_path):
    """Notes heard 15 cents over their settled pitch (an 808 still falling
    over the read window) played by a one-shot that does not fall: Detune
    +15, and estimate_from_plan hears them there; notes that disagree (half
    at their settled pitch, half a semitone over) leave Detune alone."""
    import soundfile as sf

    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import plan_808
    hz = 440 * 2 ** ((24 - 69) / 12)
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, voice(Params(hz, hz, 10, 3.0, 4, 1.0, 0.0, 0.3), 3.0), SR, subtype="FLOAT")
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}

    def notes(heard):
        return [{"t": 0.5 * i, "t_attack": 0.5 * i + 0.005, "end": 0.5 * i + 0.4, "pitch": 24, "peak": 1.0,
                 "midi_settled": 24.0, "midi_float": 24.0 + h} for i, h in enumerate(heard)]

    track = plan_808({"tuning": {"808": 0.0}, "notes_808": notes([0.15] * 8)}, synth, tmp_path / "stems", 2.0)
    assert (track["transpose"], track["detune_cents"]) == (36, 15) and track["heard_cents"] == pytest.approx(15, abs=1)
    plan = {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": {}, "notes": []}, "808": track},
            "levels": {}}
    assert [n["midi"] for n in estimate_from_plan(plan)["bass808"]] == pytest.approx([24.15] * 8, abs=0.01)
    track = plan_808({"tuning": {"808": 0.0}, "notes_808": notes([0.0, 0.5] * 4)}, synth, tmp_path / "stems", 2.0)
    assert (track["transpose"], track["detune_cents"], track["heard_cents"]) == (36, 0, None)


def test_a_legato_note_slides_in_the_plan_and_in_its_render(tmp_path):
    """A legato note is planned where its glide starts, with the note before
    held over it, one voice and Glide; what Live would play (render_plan,
    estimate_from_plan) slides there without a new attack and settles at
    the new pitch."""
    import soundfile as sf

    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import LEGATO_S, plan_808
    from refdissect.render import render_plan
    hz = 32.70
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, voice(Params(hz, hz, 10, 3.0, 4, 1.0, 0.0, 0.3), 3.0), SR, subtype="FLOAT")
    glide = {"from_midi": 24.0, "to_midi": 29.0, "t": 1.0, "ms": 80.0, "shape": "lin", "kind": "legato"}
    analysis = {"tuning": {"808": 0.0},
                "notes_808": [{"t": 0.5, "t_attack": 0.505, "end": 1.0, "pitch": 24, "peak": 1.0},
                              {"t": 1.0, "t_attack": 1.0, "end": 1.6, "pitch": 29, "peak": 1.0, "glide": glide,
                               "flags": ["legato"]}]}
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}
    track = plan_808(analysis, synth, tmp_path / "stems", 2.0)
    assert (track["voices"], track["glide_mode"], track["glide_time_ms"]) == (1, "Glide", 80.0)
    first, second = track["notes"]
    assert second["start_time"] == 2.0 and first["start_time"] + first["duration"] == pytest.approx(
        2.0 + 2 * LEGATO_S, abs=1e-3)
    plan = {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": {}, "notes": []}, "808": track},
            "levels": {}}
    heard = estimate_from_plan(plan)["bass808"]
    assert heard[1]["t_attack"] == heard[1]["t"] == 1.0 and heard[1]["flags"] == ["legato"]
    assert heard[1]["midi_settled"] == pytest.approx(29.0, abs=0.01)
    assert heard[1]["midi"] == pytest.approx(29.0, abs=0.1)
    m = render_plan(plan, tmp_path / "r", faders=False)
    x = sf.read(tmp_path / "r" / m["tracks"]["808"], dtype="float32")[0]
    assert B.qifft(x[int(1.2 * SR):int(1.55 * SR)], 35, 50) == pytest.approx(440 * 2 ** ((29 - 69) / 12), rel=0.002)
    env = B.rms_envelope(x, 5.0)
    assert env[int(0.98 * SR):int(1.05 * SR)].min() > 0.5 * env[int(0.9 * SR)]  # no cut, no new attack
