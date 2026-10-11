"""The drum pipeline on a synthetic beat whose answer is known."""

import numpy as np

from refdissect.calibrate import _f1
from refdissect.detect import detect, discover
from refdissect.grid import fit_grid
from refdissect.structure import bar_activity, sections


def test_snare_and_hats_found_on_their_own(beat):
    grid, _ = fit_grid(beat["audio"], (100, 180))
    templates = discover(beat["audio"], grid)
    events = detect(beat["audio"], templates)
    scores = {}
    for name in beat["truth"]:
        best = max((_f1([e["t"] for e in events[tp.name]], beat["truth"][name]) for tp in templates),
                   key=lambda s: s["f1"])
        scores[name] = best["f1"]
    assert scores["snare"] > 0.95
    assert scores["kick"] > 0.9
    assert scores["hat"] > 0.6


def test_sections_from_activity():
    from refdissect.grid import Grid
    g = Grid(120.0, 0.0)
    events = {"sound_1": [{"t": 1.0 + 2.0 * b} for b in list(range(0, 8)) + list(range(12, 16))]}
    rows = bar_activity(g, 16, events, [{"t": 0.1 + 2.0 * b} for b in range(16)], min_hits=4)
    secs = sections(rows)
    assert [(s["start_bar"], s["end_bar"]) for s in secs] == [(1, 8), (9, 12), (13, 16)]
    assert secs[0]["name"] == secs[2]["name"] != secs[1]["name"]
    assert np.all([("808" in s["plays"]) for s in secs])
