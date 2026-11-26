"""Loop-like layers: sounds that keep one place of a cycle of several bars."""

import numpy as np

from refdissect import loops
from refdissect.audio import SR, read
from refdissect.grid import Grid

GRID = Grid(120.0, 0.05)


def at(bar: int, step: int, late: float = 0.0) -> float:
    return GRID.phase + bar * GRID.bar + step * GRID.step + late


def test_a_sound_once_every_four_bars_has_a_period_of_four():
    p = loops.period([at(b, 6, 0.03) for b in range(2, 62, 4)], GRID)
    assert (p["bars"], p["places"], p["hits"], p["cycles"]) == (4, [[3, 6]], 15, 15)


def test_a_part_of_every_bar_has_a_period_of_one_and_a_changing_one_none():
    backbeat = [at(b, s) for b in range(32) for s in (4, 12)]
    assert loops.period(backbeat, GRID)["bars"] == 1
    assert loops.period([at(b, 13) for b in range(1, 32, 2)], GRID)["bars"] == 2
    three_places = [at(b, s) for b in range(32) for s in ((0, 13) if b % 2 else (7,))]
    assert loops.period(three_places, GRID) is None  # more than PLACES places at any period: a part, not a hit
    rng = np.random.default_rng(1)
    assert loops.period([at(b, int(rng.integers(16))) for b in range(40)], GRID) is None
    assert loops.period([at(0, 3), at(4, 3), at(8, 3)], GRID) is None  # three cycles: too few


def test_a_sound_that_stops_coming_back_is_not_periodic():
    assert loops.period([at(b, 5) for b in (0, 4, 8, 12, 60, 124, 188)], GRID) is None  # 7 of 48 cycles


def test_a_place_of_four_bars_played_in_some_cycles_has_no_period_of_eight():
    assert loops.period([at(b, 14) for b in (0, 8, 20, 24, 36, 44, 48, 60)], GRID) is None


def test_sounds_at_one_period_make_a_layer_and_a_cycle_is_cut(tmp_path):
    events = {f"hat_{i}": [{"t": at(b, i), "ncc": 0.9} for b in range(i % 4, 32, 4)] for i in range(6)}
    events["snare"] = [{"t": at(b, s), "ncc": 0.95} for b in range(32) for s in (4, 12)]
    events["hat_0"] = events["hat_0"][1:]  # the first cycle lacks a sound
    templates = {k: {"band": "snare" if k == "snare" else "hat"} for k in events}
    (layer,) = loops.layers(events, templates, GRID)
    assert layer["bars"] == 4 and layer["band"] == "hat" and sorted(layer["sounds"]) == sorted(set(events) - {"snare"})
    assert layer["hits"] == 47 and layer["share_of_band"] == 1.0
    assert layer["cycle"] == {"start_bar": 5, "start_s": round(at(4, 0), 5), "sounds": 6}
    files = loops.cut(np.random.default_rng(0).normal(0, 0.1, 70 * SR).astype(np.float32), GRID, layer, tmp_path)
    assert sorted(files) == ["drums", "hat"]
    assert all(len(read(f)) == int(round(4 * GRID.bar * SR)) for f in files.values())


def test_three_periodic_sounds_are_no_layer():
    events = {f"hat_{i}": [{"t": at(b, i)} for b in range(0, 32, 4)] for i in range(3)}
    assert loops.layers(events, {k: {"band": "hat"} for k in events}, GRID) == []
