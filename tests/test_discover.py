"""discover() on synthetic beats whose answer is known: grooves off the
straight 16ths, meters other than 4/4, rare sounds, sounds that play
together, one sound in two bands, and where a sound starts."""

import numpy as np
import pytest
from conftest import burst

from refdissect.audio import PRE, SR, attack_time, samples
from refdissect.detect import (
    Candidate,
    Template,
    count_once,
    detect,
    discover,
    min_hits_for,
    one_per_sound,
    places,
    starts_before,
    starts_something,
    unlocked,
)
from refdissect.evaluate import assign, onset_scores
from refdissect.grid import Grid
from refdissect.hits import Group

ROLE = {"kick": "kick", "snare": "snare", "hat": "hat", "rim": "rim", "perc": "perc", "crash": "crash",
        "hat2": "hat", "late": "kick", "lead": "snare", "boom": "kick", "tonal": "snare", "pickup": "hat",
        "sine": "kick"}


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
    return templates, count_once(detect(x, templates), templates, x)


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


def test_a_tonal_snares_ring_is_not_its_hits(sounds):
    """A snare with a 190 Hz body (70 ms decay), a pickup one 32nd before
    each one (140 BPM): the snare template matched its own ring 37-158 ms
    after every snare (101 hits for 32), and the scorer gave it to no
    sound; the template that held the snare in the low band was its view,
    its hits all counted as the other's."""
    tonal = burst(31, 250, 150, 3500, 70, tone_hz=190.0)
    hits, grid = pattern(140, 16, {"kick": [0, 10], "tonal": [4, 12], "hat": [0, 2, 4, 6, 8, 10, 12, 14]},
                         gains={"hat": (0.3, 0.5)})
    step = 60 / 140 / 4
    hits += [("pickup", t - step / 2, 0.25) for name, t, _ in list(hits) if name == "tonal"]
    pickup = burst(32, 60, 4000, 14000, 12)
    x, truth = arrange({**sounds, "tonal": tonal, "pickup": pickup}, hits, 16 * 4 * 60 / 140 + 1)
    sc = scored(*found(x, grid), truth)
    assert sc["tonal"]["f1_20"] > 0.95 and abs(sc["tonal"]["error_ms"]["median"]) < 2


def sine_kick(hz: float = 50.0, ms: float = 400, decay_ms: float = 150, drop: float = 1.5) -> np.ndarray:
    """An 808-style kick: a sine falling `drop` octaves over its first 40
    ms, no click."""
    t = np.arange(samples(ms)) / SR
    f = hz * 2 ** (drop * np.exp(-t / 0.04))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (decay_ms / 1000)) * np.clip(t / 0.002, 0, 1)
    return (x / np.max(np.abs(x))).astype(np.float32)


@pytest.mark.parametrize("bpm, steps", [(140, {"sine": [3, 11], "snare": [8], "hat": [0, 2, 4, 6, 8, 10, 12, 14]}),
                                        (90, {"sine": [0, 7, 10], "snare": [4, 12]})])
def test_a_kick_without_a_click_is_found(sounds, bpm, steps):
    """A clickless kick between 8th hats, or with no hats at all: nothing
    rises in its hat band, as for a kick's body after its click, but
    nothing starts just before it either (it was lost: F1 0)."""
    hits, grid = pattern(bpm, 16, steps, gains={"hat": (0.3, 0.5), "sine": (0.9, 0.9)})
    x, truth = arrange({**sounds, "sine": sine_kick()}, hits, 16 * 4 * 60 / bpm + 1)
    sc = scored(*found(x, grid), truth)
    assert sc["sine"]["f1_20"] > 0.95 and abs(sc["sine"]["error_ms"]["median"]) < 3


def test_a_view_goes_with_its_main():
    """unlocked() drops the quieter of two candidates locked 42 ms apart; a
    view of the one it drops is the same sound and goes with it (it stayed,
    named after a template that was gone: discover() raised KeyError)."""
    x = np.zeros(SR * 4, np.float32)
    t_b = np.arange(0.5, 3.5, 0.25)
    for t, level in [(t, 0.8) for t in t_b] + [(t + 0.042, 0.1) for t in t_b]:
        x[int(t * SR):int(t * SR) + 600] = level

    def cand(band_name, role, times):
        n = len(times)
        return Candidate(band_name, Group(), np.zeros(10, np.float32), PRE, np.sort(times), np.ones(n), np.ones(n),
                         np.zeros(100, np.float32), role)

    main, louder = cand("low", "kick", t_b + 0.042), cand("snare", "snare", t_b)
    view = cand("hat", "kick", np.r_[t_b[:6] + 0.042, t_b[:6] + 0.167])
    view.view_of = main
    assert unlocked(x, [main, view, louder]) == [louder]


def test_lead_ins_of_different_lengths(sounds):
    """As on the suite's leadin beats: quiet noise (-60 dB) before every
    sound, 11 ms before the kick, 20 before the snare, 28 before the hats,
    so the hats on the kicks attack 17 ms after them; the snare has a weak
    click 6 ms before its body. Nothing rises in the hat band within 15 ms
    of a kick, and nothing starts 15-45 ms before it: the kick stays (it
    was dropped as a kick's body: F1 0)."""
    rng = np.random.default_rng(7)

    def lead(y, ms):
        return np.r_[rng.standard_normal(samples(ms)).astype(np.float32) * 0.001, y]

    snare = np.zeros(samples(260), np.float32)
    snare[:samples(2)] = 0.35 * burst(43, 2, 1000, 9000, 1)
    snare[samples(6):samples(256)] += burst(41, 250, 180, 9000, 60)
    kit = {"kick": lead(sounds["kick"], 11), "snare": lead(snare, 20), "hat": lead(sounds["hat"], 28)}
    hits, grid = pattern(137, 16, BACKBEAT, gains={"hat": (0.25, 0.4)})
    x, truth = arrange(kit, hits, 16 * BAR + 1)
    sc = scored(*found(x, grid), truth)
    assert {k: v["f1_20"] > 0.9 for k, v in sc.items()} == {"kick": True, "snare": True, "hat": True}
    assert abs(sc["snare"]["error_ms"]["median"]) < 2 and abs(sc["hat"]["error_ms"]["median"]) < 2


def test_a_clean_template_may_start_a_little_before_its_onsets():
    """Hand-built hits 13 ms before the onsets, as leadin-1's snare template
    started (where its snare band climbs from the floor) before its onsets
    (where the hat band reached 30 %): clean (NCC 0.95) they start
    something; at NCC 0.7 they do not."""
    t = 0.5 + 0.5 * np.arange(16)
    top = np.zeros(SR * 9, np.float32)
    for a in t + 0.013:
        top[int(a * SR):int(a * SR) + 200] = 1.0

    def cand(ncc):
        return Candidate("hat", Group(), np.zeros(10, np.float32), PRE, t, np.ones(16), np.full(16, ncc),
                         np.zeros(100, np.float32), "snare")

    assert starts_something(cand(0.95), t + 0.013, top) and not starts_something(cand(0.7), t + 0.013, top)


def test_a_views_hits_found_again_37_ms_later_count_once():
    """A view finds its main's hits again at one lag: full-1's snare-band
    view of the kick kept 43 hits 37 ms after a kick when hits counted once
    only within 15 ms, none of them on a kick. Within 50 ms they are its
    main's; its own hit stays."""
    main = Template("sound_1", "kick", "low", np.zeros(10, np.float32), 8)
    view = Template("sound_2", "kick", "snare", np.zeros(10, np.float32), 8, same_sound_as="sound_1")

    def hits(times):
        return [{"t": t, "t_attack": t, "ncc": 0.9, "gain": 1.0} for t in times]

    out = count_once({"sound_1": hits([1.0, 2.0, 3.0, 4.0]), "sound_2": hits([1.0, 1.037, 2.037, 3.037, 5.5])},
                     [main, view])
    assert [e["t_attack"] for e in out["sound_2"]] == [5.5] and len(out["sound_1"]) == 4


def test_hits_found_again_do_not_make_a_view(kit):
    """A candidate of a kept sound that finds 4 of its hits, and the same 4
    again 37 ms later, finds nothing new: it is no view of that sound, its
    hits are not counted twice."""
    t = np.arange(1.0, 9.0)

    def cand(band_name, times):
        n = len(times)
        return Candidate(band_name, Group(), np.zeros(10, np.float32), PRE, np.sort(times), np.ones(n),
                         np.full(n, 0.9), kit["kick"][:samples(100)], "kick")

    kept = cand("low", t)
    assert one_per_sound([cand("snare", np.r_[t[:4], t[:4] + 0.037])], [kept]) == [kept]
    assert len(one_per_sound([cand("snare", np.r_[t[:4], 9.5, 10.5, 11.5])], [kept])) == 2


def test_a_template_starts_where_its_hits_do():
    """starts_before() on hand-made bands: every hit starts 19 ms before
    where its template says (a click before a body that rises too slowly to
    start anything, as v1orig-5's kick template did); the template's start
    moves there. Hits that also start where the template says (a flam's
    main hit after its grace note) do not move it."""
    t = 0.5 + 0.25 * np.arange(12)
    click = np.zeros(SR * 4, np.float32)
    for a in t - 0.019:
        click[int(a * SR):int(a * SR) + samples(3)] = 1.0
    swell = np.zeros(SR * 4, np.float32)
    for a in t - 0.010:
        i = int(a * SR)
        swell[i:i + samples(40)] = np.linspace(0.05, 1.0, samples(40))
    assert starts_before([click, swell], t) == pytest.approx(-0.019, abs=0.001)
    hit = np.zeros(SR * 4, np.float32)
    for a in t:
        hit[int(a * SR):int(a * SR) + samples(30)] = 1.0
    assert starts_before([click, hit], t) == 0.0


def test_the_residual_pass_asks_more_of_a_long_song():
    """A residual group needs 3 hits on a 30 s beat, one per 16 bars in a
    longer song (the grid's groups one per 4 bars): with 3, the extra
    templates grew with the song's length."""
    grid = Grid(120.0, 0.0)
    assert [min_hits_for(int(bars * grid.bar * SR), grid, 16) for bars in (16, 72, 96)] == [3, 5, 6]
    assert [min_hits_for(int(bars * grid.bar * SR), grid) for bars in (16, 72, 96)] == [4, 18, 24]
