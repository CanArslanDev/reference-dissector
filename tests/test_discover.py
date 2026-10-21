"""discover() on synthetic beats whose answer is known: grooves off the
straight 16ths, meters other than 4/4, rare sounds, sounds that play
together, one sound in two bands, and where a sound starts."""

import numpy as np
import pytest
from conftest import burst

from refdissect.audio import SR, attack_time, samples
from refdissect.detect import count_once, detect, discover, places
from refdissect.evaluate import assign, onset_scores
from refdissect.grid import Grid

ROLE = {"kick": "kick", "snare": "snare", "hat": "hat", "rim": "rim", "perc": "perc", "crash": "crash",
        "hat2": "hat", "late": "kick", "lead": "snare", "boom": "kick"}


@pytest.fixture(scope="module")
def sounds(kit):
    return {**kit, "rim": burst(4, 120, 900, 5000, 18), "perc": burst(5, 150, 1500, 4000, 30, tone_hz=700.0),
            "crash": burst(6, 1500, 3000, 16000, 500), "hat2": burst(7, 80, 3000, 14000, 15)}


def arrange(sounds: dict, hits: list[tuple[str, float, float]], seconds: float) -> tuple[np.ndarray, dict]:
    """Each (sound, start in s, gain) of hits played into seconds of quiet
    noise: (audio, {sound: attack times}), the attack read as the truth reads
    it (audio.attack_time in the sound's role band)."""
    x = np.random.default_rng(9).standard_normal(int(seconds * SR)).astype(np.float32) * 0.003
    truth: dict[str, list[float]] = {}
    for name, t, gain in hits:
        a, y = int(round(t * SR)), sounds[name]
        x[a:a + len(y)] += gain * y[:len(x) - a]
        truth.setdefault(name, []).append(t + attack_time(y, ROLE[name]))
    return x, {k: sorted(v) for k, v in truth.items()}


def pattern(bpm: float, bars: int, steps: dict[str, list[int]], per_bar: int = 16, lead: float = 0.03,
            swing: float = 0.5, gains: dict | None = None) -> tuple[list, Grid]:
    """Hits on 16th steps of each bar (the odd ones at `swing` of their 8th)
    and the grid they sit on."""
    step = 60 / bpm / 4
    hits = []
    rng = np.random.default_rng(3)
    for b in range(bars):
        for name, where in steps.items():
            for s in where:
                at = (b * per_bar + s) * step + (2 * swing - 1) * step * (s % 2)
                g = (gains or {}).get(name)
                hits.append((name, lead + at, float(rng.uniform(*g)) if g else 0.8))
    beats_per_bar = per_bar // 4
    return hits, Grid(bpm, lead, beats_per_bar, 4)


def found(x: np.ndarray, grid: Grid) -> tuple[list, dict]:
    templates = discover(x, grid)
    return templates, count_once(detect(x, templates), templates)


def scored(templates, events, truth: dict) -> dict:
    """As the suite scores them: each template goes to the true sound it
    plays (evaluate.assign), and the t_attack of those given to a sound are
    pooled (a hit played twice is a false one): {sound: onset_scores and the
    templates given to it}."""
    est = {tp.name: np.sort([e["t_attack"] for e in events[tp.name]]) for tp in templates}
    given, _extra = assign({k: np.array(v) for k, v in truth.items()}, est)
    out = {}
    for sound, times in truth.items():
        mine = [e for e, a in given.items() if a["to"] == sound]
        out[sound] = {**onset_scores(times, np.concatenate([est[e] for e in mine]) if mine else []), "templates": mine}
    return out


BACKBEAT = {"kick": [0, 10], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14]}
BAR = 4 * 60 / 137


def test_a_rim_on_swung_16ths_is_found(sounds):
    """A rim only on a swung 16th (62 %, 26 ms late at 137 BPM): the old
    16th slots (+-12 ms) never seeded it."""
    hits, grid = pattern(137, 16, {**BACKBEAT, "rim": [13]}, swing=0.62, gains={"hat": (0.25, 0.4)})
    x, truth = arrange(sounds, hits, 16 * BAR + 1)
    sc = scored(*found(x, grid), truth)
    assert sc["rim"]["f1_20"] > 0.95 and abs(sc["rim"]["error_ms"]["median"]) < 2


def test_triplet_hats(sounds):
    """Hats on 8th-note triplets over a kick and snare in 4/4."""
    bpm, bars = 120.0, 12
    beat = 60 / bpm
    hits = [("hat", 0.03 + (b * 4 + q) * beat + k * beat / 3, 0.35) for b in range(bars) for q in range(4)
            for k in range(3)]
    hits += [("kick", 0.03 + b * 4 * beat, 0.9) for b in range(bars)]
    hits += [("snare", 0.03 + (b * 4 + 2) * beat, 0.8) for b in range(bars)]
    x, truth = arrange(sounds, hits, bars * 4 * beat + 1)
    sc = scored(*found(x, Grid(bpm, 0.03)), truth)
    assert {k: v["f1_20"] > 0.9 for k, v in sc.items()} == {"hat": True, "kick": True, "snare": True}


@pytest.mark.parametrize("meter", ["3/4", "6/8"])
def test_three_four_and_six_eight(sounds, meter):
    """Bars of 12 16ths: kick, snare and hats each found (the old slots,
    taken modulo 16, spread them over the bars)."""
    steps = ({"kick": [0, 6], "snare": [4], "hat": [0, 2, 4, 6, 8, 10]} if meter == "3/4"
             else {"kick": [0], "snare": [6], "hat": [0, 2, 4, 6, 8, 10]})
    hits, _ = pattern(137, 16, steps, per_bar=12, gains={"hat": (0.25, 0.4)})
    n, unit = (3, 4) if meter == "3/4" else (6, 8)
    x, truth = arrange(sounds, hits, 16 * 3 * 60 / 137 + 1)
    sc = scored(*found(x, Grid(137.0, 0.03, n, unit)), truth)
    assert {k: v["f1_20"] > 0.9 for k, v in sc.items()} == {"hat": True, "kick": True, "snare": True}


def test_a_rare_perc_and_a_crash_every_8_bars(sounds):
    """Three perc hits off the grid and a crash on every 8th bar line are
    found by the second pass, which groups what no template explains."""
    hits, grid = pattern(137, 24, BACKBEAT, gains={"hat": (0.25, 0.4)})
    hits += [("perc", t, 0.7) for t in (3.217, 9.731, 15.402)]
    hits += [("crash", 0.03 + b * BAR, 0.6) for b in (0, 8, 16)]
    x, truth = arrange(sounds, hits, 24 * BAR + 2)
    templates, events = found(x, grid)
    sc = scored(templates, events, truth)
    found_by = {tp.name: tp.found for tp in templates}
    assert sc["crash"]["f1_20"] == 1.0 and abs(sc["crash"]["error_ms"]["median"]) < 2
    # every perc hit is found; its template also matches its own ringing (5 hits 36-74 ms after
    # its 3, at a quarter of their level), which detect() keeps
    assert sc["perc"]["recall_20"] == 1.0 and abs(sc["perc"]["error_ms"]["median"]) < 2
    assert {found_by[t] for name in ("perc", "crash") for t in sc[name]["templates"]} == {"residual"}


def test_a_kick_that_always_sits_on_a_hat_stays_a_kick(sounds):
    """Every kick lands on a hat (and most hats have none): the kick is not
    taken for the hat seen in another band."""
    hits, grid = pattern(137, 16, {"kick": [0, 6], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
                         gains={"hat": (0.25, 0.4)})
    x, truth = arrange(sounds, hits, 16 * BAR + 1)
    sc = scored(*found(x, grid), truth)
    assert sc["kick"]["f1_20"] > 0.95 and sc["hat"]["f1_20"] > 0.8


def test_the_same_hat_in_two_bands_is_one_sound(sounds):
    """A hat with as much in the snare band as in the hat band shows up in
    both; it is one sound, its hits counted once."""
    hits, grid = pattern(137, 16, {"kick": [0, 10], "snare": [8], "hat2": [0, 2, 4, 6, 8, 10, 12, 14]},
                         gains={"hat2": (0.3, 0.9)})
    x, truth = arrange(sounds, hits, 16 * BAR + 1)
    sc = scored(*found(x, grid), truth)
    assert len(sc["hat2"]["templates"]) == 1 and sc["hat2"]["precision_20"] == 1.0
    assert sc["hat2"]["f1_20"] > 0.9


def test_a_lead_in_does_not_move_the_attack(sounds, kit):
    """A snare with 20 ms of quiet noise before its attack: its hits are
    timed where it attacks, as the truth times it."""
    lead = np.r_[np.random.default_rng(2).standard_normal(samples(20)).astype(np.float32) * 0.01, kit["snare"]]
    hits, grid = pattern(137, 16, {"kick": [0, 10], "lead": [8], "hat": [0, 2, 4, 6, 10, 12, 14]},
                         gains={"hat": (0.25, 0.4)})
    x, truth = arrange({**sounds, "lead": lead}, hits, 16 * BAR + 1)
    assert truth["lead"][0] - 0.03 - 8 * BAR / 16 == pytest.approx(0.020, abs=0.002)
    sc = scored(*found(x, grid), truth)
    assert sc["lead"]["f1_20"] > 0.95 and abs(sc["lead"]["error_ms"]["median"]) < 2


def test_a_snare_after_a_ringing_kick_is_timed_on_its_attack(sounds):
    """A snare a 16th after a kick that still rings at 55 Hz: in the full
    band its attack sat on the kick's tail (calibration snares 13-24 ms
    early); in the snare band it is where the snare starts."""
    boom = burst(11, 600, 30, 400, 400, tone_hz=55.0)
    hits, grid = pattern(137, 16, {"boom": [7], "snare": [8], "hat": [0, 2, 4, 6, 10, 12, 14]},
                         gains={"hat": (0.25, 0.4)})
    x, truth = arrange({**sounds, "boom": boom}, hits, 16 * BAR + 1)
    sc = scored(*found(x, grid), truth)
    assert sc["snare"]["f1_20"] > 0.95 and abs(sc["snare"]["error_ms"]["median"]) < 2


def test_a_kick_whose_body_comes_late_is_timed_on_its_click(sounds):
    """htdemucs leaves a kick's click in the drum stem and its body 26 ms
    later: the kick starts at the click, though the low band rises late."""
    click = burst(13, 60, 300, 8000, 10)
    body = burst(12, 300, 40, 150, 120, tone_hz=50.0)
    late = np.zeros(samples(26) + len(body), np.float32)
    late[:len(click)] += 0.5 * click
    late[samples(26):] += body
    hits, grid = pattern(137, 16, {"late": [0, 10], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
                         gains={"hat": (0.25, 0.4)})
    x, truth = arrange({**sounds, "late": late}, hits, 16 * BAR + 1)
    truth["late"] = sorted(t for name, t, _ in hits if name == "late")
    sc = scored(*found(x, grid), truth)
    assert sc["late"]["f1_20"] > 0.95 and abs(sc["late"]["error_ms"]["median"]) < 3


def test_places_follow_the_song_not_the_straight_grid():
    """Onsets 26 ms after the odd 16ths (a 62 % swing) are a place of their
    own; a lone onset is in none."""
    grid = Grid(137.0, 0.0)
    t = np.r_[[b * grid.beat + 1.24 * grid.step for b in range(8)], [b * grid.beat for b in range(8)], 0.3333]
    where = places(t, grid, 12.0, 3)
    assert len(set(where[:8])) == 1 and len(set(where[8:16])) == 1 and where[0] != where[8]
    assert where[-1] == -1
