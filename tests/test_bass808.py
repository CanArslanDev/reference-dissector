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
         drop: float = 5.0, glide_ms: float = 50.0, slide_ms: float = 90.0, seed: int = 1, tail: float = 0.0
         ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """A bass stem, a drum stem and the truth of `bars` bars of notes (16th
    step, length in 16ths, semitones over the root, and "legato" for a note
    that slides in from the one before over slide_ms without starting the
    sample again), the 808 settling at root_hz + cents and starting `drop`
    semitones over it (and `tail` more semitones falling with a time
    constant of 150 ms, as real 808s' drops end); a kick on every other
    attacked note, leaking into the bass stem at `leak`."""
    f = root_hz * 2 ** (cents / 1200)
    sample = voice(Params(f, f * 2 ** (drop / 12), glide_ms, 1.5, 3, 2.0, 0.05, 0.3), 2.5)
    if tail:
        t = np.arange(len(sample)) / SR
        sample = np.interp(np.cumsum(2 ** (tail * np.exp(-t / 0.15) / 12)), np.arange(len(sample)), sample)
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


def test_a_slide_without_an_attack_is_a_legato_note_on_request():
    """A note that slides in over 120 ms from the one before (no attack: the
    sample goes on) is, with slides, a legato note starting within 50 ms of
    where the slide starts, at its pitch, gliding linearly; not a few short
    restarts. By default it is the note before going on: no legato note."""
    notes = [(0, 3, 0), (3, 1, 0), (4, 4, 5, "legato"), (8, 2, 0), (10, 2, 7), (12, 4, 0)]
    bass, drums, truth = line(notes, cents=11.0, slide_ms=120.0)
    b = B.transcribe(bass, drums, slides=True)
    legato = [n for n in b["notes"] if "legato" in n["flags"]]
    slides = [m for m in truth if m["legato"]]
    right = [m for m in slides if any(abs(n["t"] - m["t"]) <= 0.05 and abs(n["midi_settled"] - m["midi"]) < 0.5
                                      for n in legato)]
    assert len(right) >= 3 and len(b["notes"]) <= 1.2 * len(truth)
    assert all(n["t"] == n["t_attack"] and n["glide"]["shape"] == "lin" for n in legato)
    plain = B.transcribe(bass, drums)["notes"]
    assert not any("legato" in n["flags"] for n in plain) and len(plain) <= len(truth)


def test_legato_keeps_a_slide_and_merges_a_drift_or_a_short_reading():
    """A note found where nothing attacks inside a held note: a legato note
    where the period-by-period pitch has moved a semitone or more; the held
    note going on where it has moved less than 0.75 semitone (a sustain
    still falling: 60 cents here), where the note found lasts under 80 ms,
    or where the pitch read is farther from the template than any note's
    (20 semitones down). The slide starts where the pitch leaves the note before
    for good, not where it first strays (60 ms up a fifth, 60 ms in)."""
    from refdissect.audio import band
    hz = 32.70
    tpl = B.Template(np.zeros(1), np.zeros(1), [], hz, "qifft_300_600ms", [[0.0, 0.0], [2.0, 0.0]])
    # (where the pitch moves, by how much, the note found's start and end, notes out, the legato note's start)
    cases = ((0.5, -0.6, 0.5, 1.5, 1, None), (0.5, -1.0, 0.5, 1.5, 2, 0.5), (0.5, 5.0, 0.5, 1.5, 2, 0.5),
             (0.5, 5.0, 0.5, 0.56, 1, None), (0.4, 5.0, 0.55, 1.5, 2, 0.4), (0.5, -20.0, 0.5, 1.5, 1, None))
    for at, moved, t, end, notes_out, start in cases:
        f = np.full(int(1.5 * SR), hz)
        f[int(at * SR):] *= 2 ** (moved / 12)
        f[int(0.06 * SR):int(0.12 * SR)] *= 2 ** (7 / 12)
        lo = band(np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32), "808")
        notes = [{"t": 0.0, "t_attack": 0.0, "end": t, "midi_settled": tpl.midi, "ncc": 0.95, "flags": []},
                 {"t": t, "t_attack": t, "end": end, "midi_settled": tpl.midi + moved, "ncc": 0.6, "flags": []}]
        out = B.legato(notes, lo, tpl, np.array([0.0]), np.zeros(0), slides=True)
        assert len(out) == notes_out
        assert len(B.legato(notes, lo, tpl, np.array([0.0]), np.zeros(0))) == 1  # slides off: merged
        if notes_out == 1:
            assert out[0]["end"] == end
        else:
            assert out[1]["flags"] == ["legato"] and out[1]["midi_settled"] == pytest.approx(tpl.midi + moved, abs=0.01)
            assert out[0]["end"] == out[1]["t"] == pytest.approx(start, abs=0.035)


def test_a_kick_inside_a_held_note_is_the_note_going_on():
    """kick808-2's failure: a note held at MIDI 22.1, a kick leaking into
    the bass stem read as a 70 ms note at 17.1 (attacked: a drum attack),
    then the held note's sustain found again with no attack. The pitch,
    period by period, is back at 22.1: the kick and the follower are the
    held note going on. Where the pitch moved to the kick note's own, the
    kick note stays (the follower is it going on)."""
    from refdissect.audio import band
    hz = 440 * 2 ** ((22.1 - 69) / 12)
    tpl = B.Template(np.zeros(1), np.zeros(1), [], 32.70, "qifft_300_600ms", [[0.0, 0.0], [2.0, 0.0]])
    for after, kept in ((22.1, 1), (17.1, 2)):
        f = np.full(int(1.6 * SR), hz)
        f[int(0.6 * SR):] = 440 * 2 ** ((after - 69) / 12)
        f[int(0.53 * SR):int(0.6 * SR)] = 440 * 2 ** ((17.1 - 69) / 12)
        lo = band(np.sin(2 * np.pi * np.cumsum(f) / SR).astype(np.float32), "808")
        notes = [{"t": 0.0, "t_attack": 0.005, "end": 0.53, "midi_settled": 22.1, "ncc": 0.97, "flags": []},
                 {"t": 0.53, "t_attack": 0.535, "end": 0.6, "midi_settled": 17.1, "ncc": 0.8, "flags": []},
                 {"t": 0.6, "t_attack": 0.6, "end": 1.5, "midi_settled": 14.0, "ncc": 0.6, "flags": []}]
        out = B.legato(notes, lo, tpl, np.array([0.0]), np.array([0.53]))
        assert len(out) == kept and out[-1]["end"] == 1.5
        assert [n["midi_settled"] for n in out] == [22.1, 17.1][:kept]


def tone(parts, hz: float = 32.70, seconds: float = 3.0) -> np.ndarray:
    """The 808 band of a tone at hz that starts again (from zero phase, at
    level 1) at each (start, end, decay seconds, gain curve or None) of
    parts and is silent between them, with noise at -100 dB."""
    from refdissect.audio import band
    y = np.random.default_rng(0).standard_normal(int(seconds * SR)) * 1e-5
    for a, b, decay, gain in parts:
        t = np.arange(int((b - a) * SR)) / SR
        g = np.exp(-t / decay) * (gain(t) if gain else 1.0)
        y[int(a * SR):int(a * SR) + len(t)] = np.sin(2 * np.pi * hz * t) * g
    return band(y.astype(np.float32), "808")


def note(t: float, end: float, ncc: float, midi: float = 24.0) -> dict:
    return {"t": t, "t_attack": t + 0.005, "end": end, "midi_settled": midi, "ncc": ncc, "flags": []}


FLAT = B.Template(np.zeros(1), np.zeros(1), [], 32.70, "qifft_300_600ms", [[0.0, 0.0], [2.0, 0.0]])


def test_a_note_never_holds_across_a_silence():
    """PIRAMIT's failure: an 808 note, 1.0 s of silence, then the 808 alone
    (no kick) on a note whose onset sits 27 ms before its envelope rises
    and whose template fit is 0.89; a note found in the rest, 60 dB under
    the others. Both were the note before going on, which then held the
    rest (15.3 s on PIRAMIT). Now the note before ends where it falls
    silent (0.4 s), the note after the rest is a note of its own, and the
    note in the rest is none."""
    lo = tone([(0.0, 0.4, 1.0, None), (1.4, 2.4, 1.0, None), (0.9, 1.0, 1.0, lambda t: 1e-3)])
    notes = [note(0.0, 0.9, 0.97), note(0.9, 1.373, 0.4), note(1.373, 2.4, 0.89)]
    feet = np.array([o["i"] for o in B.reattacks(lo)]) / SR
    feet = feet[(feet < 1.38) | (feet > 1.42)]  # its rise aside: the rest alone makes it a note
    out = B.legato(B.audible(notes, B.rms_envelope(lo, 25.0)), lo, FLAT, feet, np.zeros(0))
    assert [n["t"] for n in out] == [0.0, 1.373]
    assert out[0]["end"] == pytest.approx(0.4, abs=0.02) and out[1]["end"] == 2.4


def test_a_rise_after_the_onset_attacks_the_note():
    """PIRAMIT's 808 alone, re-attacking from a sustain 16 dB down (no
    silence, no kick): its envelope rises 12-21 dB 27 ms after the onset
    the template's fit gives (0.89). The rise attacks the note. A weak rise
    there (4 dB: the suite's notes merged with a foot 20-40 ms late rise
    -5.6 to +7.9 dB) does not."""
    for jump_db, kept in ((16.0, 2), (4.0, 1)):  # the sustain decays to jump_db under the new note's level
        lo = tone([(0.0, 1.427, 1.427 / np.log(10 ** (jump_db / 20)), None), (1.427, 2.6, 1.0, None)])
        notes = [note(0.0, 1.4, 0.97), note(1.4, 2.6, 0.89)]
        feet = np.array([o["i"] for o in B.reattacks(lo)]) / SR
        out = B.legato(notes, lo, FLAT, feet, np.zeros(0))
        assert len(out) == kept and out[-1]["end"] == 2.6


def test_a_note_after_a_rest_starts_on_its_rise_at_the_pitch_its_periods_hold():
    """PIRAMIT's notes after a rest, kept by the rest alone (no kick, a
    template fit under 0.9): segment() placed one 27 ms before its rise at
    +13 semitones (C#2 where the stem holds C1) and one 45 ms after its rise
    at +84.5 cents. Each now starts on its rise, at the template's whole
    transposition nearest the pitch its periods settle at (C1). A pitch
    within half a semitone of the stem's is kept."""
    lo = tone([(0.0, 0.4, 1.0, None), (1.4, 1.9, 1.0, None), (2.4, 3.0, 1.0, None)], seconds=3.2)
    feet = np.array([o["i"] for o in B.reattacks(lo)]) / SR
    rises = [feet[np.argmin(np.abs(feet - t))] for t in (1.4, 2.4)]  # the band filter puts them 8 ms early
    for second, third in ((37.0, 24.845), (24.3, 24.3)):
        notes = [note(0.0, 0.9, 0.97), note(rises[0] - 0.027, 2.4, 0.89, second),
                 note(rises[1] + 0.045, 3.2, 0.78, third)]
        out = B.legato(B.audible(notes, B.rms_envelope(lo, 25.0)), lo, FLAT, feet, np.zeros(0))
        settled = [FLAT.midi if m > 24.5 else m for m in (second, third)]
        assert [n["t"] for n in out[1:]] == pytest.approx(rises, abs=1e-5)
        assert [n["midi_settled"] for n in out[1:]] == pytest.approx(settled, abs=0.01)
        pre = B.PRE / SR  # the attack moves with the onset, read at the note's new rate
        lead = [(0.005 + pre) * 2 ** ((m - s) / 12) - pre for m, s in zip((second, third), settled)]
        assert [n["t_attack"] - n["t"] for n in out[1:]] == pytest.approx(lead, abs=1e-4)


def test_a_note_under_a_sidechain_duck_stays_one_note():
    """A held 808 ducked 20 or 30 dB for 80 ms by a sidechained kick every
    0.4 s (the deepest dip inside the suite's truth notes was 26.8 dB under
    the note's peak), recovering over 60 ms: the notes found in a duck
    where nothing attacks (70 ms in: the duck's own start shows an
    envelope foot) are the note going on, never cut at a duck."""
    for depth in (20.0, 30.0):
        def duck(t, depth=depth):
            g = np.ones_like(t)
            for k in (0.4, 0.8, 1.2):
                d = t - k
                g = np.where((d >= 0) & (d < 0.08), 10 ** (-depth / 20), g)
                g = np.where((d >= 0.08) & (d < 0.14), 10 ** (-depth / 20 * (1 - (d - 0.08) / 0.06)), g)
            return g
        lo = tone([(0.0, 1.6, 3.0, duck)])
        notes = [note(0.0, 0.47, 0.97), note(0.47, 0.87, 0.8), note(0.87, 1.6, 0.8)]
        feet = np.array([o["i"] for o in B.reattacks(lo)]) / SR
        out = B.legato(B.audible(notes, B.rms_envelope(lo, 25.0)), lo, FLAT, feet, np.zeros(0))
        assert len(out) == 1 and out[0]["end"] == 1.6


def test_a_kick_split_off_the_next_note_at_the_edge_of_the_search():
    """PIRAMIT's C#2 notes: 60 ms on a kick at +1310 cents (past the
    search's whole semitones), followed at once by a C1 note: the C1 note
    starts on the kick. A short note inside the search, or off any kick,
    stays."""
    def notes(c, kick_at):
        return ([{"t": 1.0, "t_attack": 1.006, "end": 1.06, "transposition": c, "flags": []},
                 {"t": 1.06, "t_attack": 1.064, "end": 1.6, "transposition": 0.0, "flags": []}],
                np.array([kick_at]))
    out = B.kick_splits(*notes(1310.0, 0.995))
    assert len(out) == 1 and (out[0]["t"], out[0]["t_attack"], out[0]["flags"]) == (1.0, 1.004, ["split"])
    assert len(B.kick_splits(*notes(1100.0, 0.995))) == 2 and len(B.kick_splits(*notes(1310.0, 0.5))) == 2


def test_sixteenths_at_one_pitch_at_160_bpm():
    """The bar at 160 BPM: its pair of 16th notes a fifth up (94 ms each)
    came out as one note a semitone sharp (24 notes for 28, 20 found within
    20 ms and 50 cents); split where the 808 starts again (retriggers), 26
    notes, 24 found."""
    bass, drums, truth = line(BAR, cents=11.0, bpm=160.0)
    b = B.transcribe(bass, drums)
    assert found(truth, b["notes"]) >= 0.85 and len(b["notes"]) >= 26


def test_sparse_reattacks_learn_from_the_kicks():
    """At 170 BPM the envelope re-attacks find 6-10 of the 28 notes (a
    driven 33 Hz tone's notch at a cut is 0.5-2.3 dB), too few to learn a
    template from, and the line came out empty; with the kicks as onsets
    too, 28 of 28 notes and the tuning within 3 cents (-1.3 measured)."""
    bass, drums, truth = line(BAR, cents=11.0, bpm=170.0)
    b = B.transcribe(bass, drums)
    assert b["template"]["learnt_from"].endswith("+kicks")
    assert found(truth, b["notes"]) >= 0.95 and abs(b["template"]["tuning_cents"] - 11.0) < 3


def test_a_line_with_no_long_note_still_has_its_notes():
    """16th notes of 107 ms only (no note rings 200 ms): a template of short
    notes still places every note (attacks within 20 ms); their settled
    pitch cannot be read where they never settle."""
    bass, drums, truth = line([(s, 1, st) for s, _, st in BAR], cents=11.0)
    b = B.transcribe(bass, drums)
    t = np.array([n["t_attack"] for n in b["notes"]])
    assert b["notes"] and np.mean([np.min(np.abs(t - m["t_attack"])) <= 0.02 for m in truth]) >= 0.9


def test_the_settled_pitch_waits_for_the_drop_to_settle():
    """Long notes of an 808 falling 8 semitones with a 120 ms time
    constant: over 300-600 ms the drop is still 26 cents over its settled
    pitch, and the tuning came out +25.3 cents off; the first window whose
    halves agree, else the fitted drop, reads it within 3 cents (+0.5)."""
    bass, drums, truth = line([(0, 8, 0), (8, 4, 5), (12, 2, 0), (14, 2, 7)], cents=11.0, drop=8.0, glide_ms=120.0)
    b = B.transcribe(bass, drums)
    assert abs(b["template"]["tuning_cents"] - 11.0) < 3


def test_short_notes_read_their_settled_pitch_past_the_template():
    """bass_low's failure: 110-330 ms notes of an 808 whose drop ends in a
    slow tail (2.5 semitones more, falling with a 150 ms time constant)
    never settle, and one exponential fitted to the template's drop leaves
    the tail out: the tuning read 35.0 cents sharp (+46.0 for +11). Notes
    up a fifth play the sample 1.5 times as fast and reach past the
    template's end; read on them, 20.4 cents. A template whose notes
    settled (QIFFT window) keeps its own reading."""
    bass, drums, _ = line([(0, 2, 0), (2, 1, 0), (3, 1, 0), (4, 3, 0), (7, 2, 0), (9, 1, 5), (10, 3, 7), (13, 3, 0)],
                          root_hz=21.83, cents=11.0, bpm=136.0, drop=8.0, glide_ms=40.0, tail=2.5)
    b = B.transcribe(bass, drums)
    assert b["template"]["method"].startswith("notes_") and abs(b["template"]["tuning_cents"] - 11.0) < 25
    assert B.settle(B.band(bass, "808"), FLAT, [(0.0, 0.3, 700.0)] * 3) is FLAT


def test_settle_refuses_notes_that_read_apart():
    """producer-2's failure: the notes in settle()'s window read 105-186
    cents apart (some an octave off, some sliding to their cut), and the
    median of all their periods, under a 30 ms margin or 5 notes, moved its
    template 64 cents the wrong way. Three notes at the 808's settled pitch
    and three a semitone under it, the template 30 cents over: the first
    reading took the lower ones' (the median of 10 periods: 100 cents
    under); the notes' spread now refuses it. Six at the settled pitch are
    read within 2 cents."""
    sharp = B.Template(np.zeros(1), np.zeros(1), [], 32.70 * 2 ** (30 / 1200), "drop_fit", [[0.0, 0.0], [2.0, 0.0]])
    starts = [0.1 + 0.8 * k for k in range(6)]
    spans = [(a, a + 0.45, 0.0) for a in starts]

    def lo(under: float) -> np.ndarray:
        return tone([(a, a + 0.45, 3.0, None) for a in starts[:3]], seconds=5.0) \
            + tone([(a, a + 0.45, 3.0, None) for a in starts[3:]], hz=32.70 * 2 ** (-under / 1200), seconds=5.0)

    assert B.settle(lo(100.0), sharp, spans) is sharp
    assert abs(1200 * np.log2(B.settle(lo(0.0), sharp, spans).hz / 32.70)) < 2


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


def test_plan_808_durations_leave_no_overlap_to_glide(tmp_path):
    """With Glide on, an overlap is a slide: a note 0.031 beat before the
    next (no legato) ends before it, where a 0.05-beat floor used to hold
    it over (6 pairs on bass_low-2, htdemucs_ft); the legato notes' notes
    before still hold over them."""
    from refdissect.live import plan_808
    glide = {"from_midi": 24.0, "to_midi": 29.0, "t": 1.0, "ms": 60.0, "shape": "lin", "kind": "legato"}
    legato = [{"t": t, "t_attack": t, "end": t + 0.3, "pitch": 29, "peak": 1.0, "flags": ["legato"],
               "glide": dict(glide, t=t)} for t in (1.0, 2.0, 3.0)]
    plain = [{"t": t, "t_attack": t, "end": t + 0.4, "pitch": 24, "peak": 1.0} for t in (0.5, 1.5, 2.5)]
    tight = [{"t": 3.6, "t_attack": 3.6, "end": 3.62, "pitch": 19, "peak": 1.0},
             {"t": 3.6155, "t_attack": 3.6155, "end": 4.0, "pitch": 9, "peak": 1.0}]
    notes = sorted(plain + legato + tight, key=lambda n: n["t"])
    track = plan_808({"tuning": {"808": 0.0}, "notes_808": notes}, None, tmp_path / "stems", 2.0)
    out = track["notes"]
    assert track["voices"] == 1 and [n["pitch"] for n in out] == [n["pitch"] for n in notes]
    for a, b, n in zip(out, out[1:], notes[1:]):
        ends = a["start_time"] + a["duration"]
        assert ends > b["start_time"] if "legato" in n.get("flags", []) else ends < b["start_time"]


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
    +15, and estimate_from_plan hears them there. Detune is one per
    Simpler: it follows the line's most common pitch (5 notes heard +15)
    over the others (3 notes a fifth up heard +50)."""
    import soundfile as sf

    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import plan_808
    hz = 440 * 2 ** ((24 - 69) / 12)
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, voice(Params(hz, hz, 10, 3.0, 4, 1.0, 0.0, 0.3), 3.0), SR, subtype="FLOAT")
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}

    def notes(heard, pitches=None):
        pitches = pitches or [24] * len(heard)
        return [{"t": 0.5 * i, "t_attack": 0.5 * i + 0.005, "end": 0.5 * i + 0.4, "pitch": p, "peak": 1.0,
                 "midi_settled": float(p), "midi_float": p + h} for i, (h, p) in enumerate(zip(heard, pitches))]

    track = plan_808({"tuning": {"808": 0.0}, "notes_808": notes([0.15] * 8)}, synth, tmp_path / "stems", 2.0)
    assert (track["transpose"], track["detune_cents"]) == (36, 15) and track["heard_cents"] == pytest.approx(15, abs=1)
    plan = {"tempo": 120.0, "length_beats": 8.0, "tracks": {"drums": {"pads": {}, "notes": []}, "808": track},
            "levels": {}}
    assert [n["midi"] for n in estimate_from_plan(plan)["bass808"]] == pytest.approx([24.15] * 8, abs=0.01)
    mixed = notes([0.15, 0.5, 0.15, 0.5, 0.15, 0.15, 0.5, 0.15], [24, 31, 24, 31, 24, 24, 31, 24])
    track = plan_808({"tuning": {"808": 0.0}, "notes_808": mixed}, synth, tmp_path / "stems", 2.0)
    assert (track["transpose"], track["detune_cents"]) == (36, 15) and track["heard_cents"] == pytest.approx(15, abs=1)


def test_heard_808_reads_each_note_from_where_the_plan_starts_it():
    """estimate_from_plan reads a note's drop from where the plan starts the
    note: bass_low-1's one-shot, started 7 ms after the onset, read its
    notes under 120 ms 38 cents over the analysis'. heard_808 reads each
    note from the start it is given (the note's onset by default), so its
    Detune follows what the plan plays."""
    from refdissect.live import heard_808
    from refdissect.scenarios import drop, sounding, tuning
    shot = voice(Params(32.70, 65.4, 30, 1.5, 3, 2.0, 0.05, 0.3), 2.0)
    f0, curve = tuning(shot)
    own = 69 + 12 * np.log2(f0 / 440)

    def heard(start, end):
        return sounding(lambda q: drop(curve, q) / 100, end - start)
    notes = [{"t": 1.0 * i, "t_attack": 1.0 * i + 0.007, "end": 1.0 * i + d, "pitch": 24, "midi_settled": own,
              "midi_float": own + heard(1.0 * i, 1.0 * i + d)} for i, d in enumerate((0.1, 0.11, 0.115, 0.3, 0.5))]
    late = [n["t"] + 0.007 for n in notes]
    expected = float(np.median([100 * (heard(n["t"], n["end"]) - heard(s, n["end"])) for n, s in zip(notes, late)]))
    assert heard_808(notes, shot, -24.0) == pytest.approx(0.0, abs=0.5)
    assert abs(expected) > 10 and heard_808(notes, shot, -24.0, late) == pytest.approx(expected, abs=0.5)


def test_plan_808_reads_its_correction_from_where_it_starts_the_notes(tmp_path):
    """A one-shot that attacks 7 ms in, not laid out as its template: the
    plan starts each note that much before its attack, and heard_808 reads
    the notes from those starts (heard_cents), as estimate_from_plan plays
    them. Read from the onsets, the short notes' correction was more than
    10 cents off."""
    import soundfile as sf

    from refdissect.audio import read
    from refdissect.live import heard_808, plan_808, tune_808
    from refdissect.scenarios import drop, sounding, tuning
    x = voice(Params(32.70, 65.4, 30, 1.5, 3, 2.0, 0.05, 0.3), 2.0)
    t = np.arange(len(x)) / SR
    w = np.clip((t - 0.007) / 0.002 + 1, 0.0, 1.0)
    path = tmp_path / "808 (synth).wav"
    sf.write(path, (x * (0.15 + 0.85 * w * w * (3 - 2 * w))).astype(np.float32), SR, subtype="FLOAT")
    shot = read(path)
    f0, curve = tuning(shot)
    own = 69 + 12 * np.log2(f0 / 440)
    notes = [{"t": 0.5 + i, "t_attack": 0.5 + i, "end": 0.5 + i + d, "pitch": 24, "peak": 1.0, "midi_settled": own,
              "midi_float": own + sounding(lambda q: drop(curve, q) / 100, d)}
             for i, d in enumerate((0.1, 0.11, 0.115, 0.3, 0.5))]
    synth = {"root_midi": 24, "pitch_hz": f0, "file": str(path), "file_name": path.name}
    track = plan_808({"tuning": {"808": own - 24}, "notes_808": notes}, synth, tmp_path / "stems", 2.0)
    starts = [p["start_time"] / 2.0 for p in track["notes"]]
    assert max(n["t"] - s for n, s in zip(notes, starts)) > 0.005
    semis, detune = tune_808(own, 100 * (own - 24))
    shift0 = semis + detune / 100 - 60
    from_starts, from_onsets = heard_808(notes, shot, shift0, starts), heard_808(notes, shot, shift0)
    assert abs(from_starts - from_onsets) > 10 and track["heard_cents"] == pytest.approx(from_starts, abs=0.1)


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
    def pair(t, to):
        return [{"t": t, "t_attack": t + 0.005, "end": t + 0.5, "pitch": 24, "peak": 1.0},
                {"t": t + 0.5, "t_attack": t + 0.5, "end": t + 1.1, "pitch": to, "peak": 1.0, "flags": ["legato"],
                 "glide": {"from_midi": 24.0, "to_midi": float(to), "t": t + 0.5, "ms": 80.0, "shape": "lin",
                           "kind": "legato"}}]
    analysis = {"tuning": {"808": 0.0}, "notes_808": pair(0.5, 29) + pair(2.0, 27) + pair(3.5, 31)}
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}
    lone = plan_808(dict(analysis, notes_808=analysis["notes_808"][:2]), synth, tmp_path / "stems", 2.0)
    assert "voices" not in lone and lone["notes"][0]["start_time"] + lone["notes"][0]["duration"] < 2.0  # GLIDE_LEAST
    track = plan_808(analysis, synth, tmp_path / "stems", 2.0)
    assert (track["voices"], track["glide_mode"], track["glide_time_ms"]) == (1, "Glide", 80.0)
    first, second = track["notes"][:2]
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


def test_a_legato_note_after_the_one_shot_has_ended_plays_nothing(tmp_path, monkeypatch):
    """A chain held past the end of its one-shot (0.5 s here) is silent from
    there on: a legato note that starts after it is left out of what Live
    plays, not given a pitch it never sounds at."""
    import soundfile as sf

    from refdissect import live
    from refdissect.evaluate import estimate_from_plan
    from refdissect.live import plan_808
    monkeypatch.setattr(live, "GLIDE_LEAST", 1)
    hz = 32.70
    shot = tmp_path / "808 (synth).wav"
    sf.write(shot, voice(Params(hz, hz, 10, 3.0, 4, 1.0, 0.0, 0.3), 0.5), SR, subtype="FLOAT")
    glide = {"from_midi": 24.0, "to_midi": 29.0, "t": 1.2, "ms": 80.0, "shape": "lin", "kind": "legato"}
    analysis = {"tuning": {"808": 0.0},
                "notes_808": [{"t": 0.5, "t_attack": 0.505, "end": 1.2, "pitch": 24, "peak": 1.0},
                              {"t": 1.2, "t_attack": 1.2, "end": 1.6, "pitch": 29, "peak": 1.0, "glide": glide,
                               "flags": ["legato"]}]}
    synth = {"root_midi": 24, "pitch_hz": hz, "file": str(shot), "file_name": shot.name}
    plan = {"tempo": 120.0, "length_beats": 8.0, "levels": {},
            "tracks": {"drums": {"pads": {}, "notes": []}, "808": plan_808(analysis, synth, tmp_path / "stems", 2.0)}}
    assert len(plan["tracks"]["808"]["notes"]) == 2
    heard = estimate_from_plan(plan)["bass808"]
    assert len(heard) == 1 and heard[0]["t"] == pytest.approx(0.5, abs=0.01) and np.isfinite(heard[0]["midi"])


def test_each_start_and_transposition_is_fitted_once(monkeypatch):
    """candidates() fits the template at a start and transposition once:
    neighbouring semitones share fine steps (+75 cents of one is -25 of the
    next), 21 % of the fits on a suite beat (v1-1, 65,037)."""
    bass, drums, _ = line(BAR, cents=11.0)
    fitted, again = set(), []
    best_fit, candidates = B._best_fit, B.candidates

    def counted(x, y, k0, *a):
        again.append((k0, id(y)) in fitted)
        fitted.add((k0, id(y)))
        return best_fit(x, y, k0, *a)

    def fresh(*a, **k):
        fitted.clear()
        return candidates(*a, **k)
    monkeypatch.setattr(B, "_best_fit", counted)
    monkeypatch.setattr(B, "candidates", fresh)
    B.transcribe(bass, drums)
    assert again and not any(again)
