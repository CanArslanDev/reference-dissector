"""grid.estimate on beats whose grid is known, with beat_this mocked (tracked)."""

import numpy as np
import pytest
from conftest import drum_beat, tracked

from refdissect.grid import Grid, estimate, fit_grid, parse_meter, tracker_line


def bar_error_ms(grid: Grid, truth: dict) -> float:
    """How far the first bar line sits from the first true downbeat, within a bar."""
    d = grid.phase - truth["downbeats"][0]
    return 1000 * ((d + grid.bar / 2) % grid.bar - grid.bar / 2)


def test_without_a_tracker_it_is_fit_grid(beat):
    """No tracker, or one with too few beats: fit_grid in bpm_range, unchanged."""
    old, old_info = fit_grid(beat["audio"], (100, 180))
    for tracker in (None, {"beats": [1.0, 2.0], "downbeats": [1.0]}):
        grid, info = estimate(beat["audio"], tracker, (100, 180))
        assert (grid.bpm, grid.phase, grid.beats_per_bar) == (old.bpm, old.phase, old.beats_per_bar)
        assert info["score"] == old_info["score"] and info["tracker"] is None
        assert info["phase_correction_ms"] == 0.0 and info["alternatives"] == []
    assert info["level"]["why"] == "no line through the tracker's 2 beats"


def switching(bpm: float, start: float, n8: int, n4: int) -> np.ndarray:
    """Tracker beats on n8 8ths, then n4 quarters, 5 ms late on 20 ms frames."""
    q = 60 / bpm
    beats = np.r_[start + np.arange(n8 + 1) * q / 2, start + n8 * q / 2 + np.arange(1, n4 + 1) * q]
    return np.round((beats + 0.005) / 0.02) * 0.02


@pytest.mark.parametrize("n8, n4", [(40, 40), (30, 30), (48, 48), (62, 62), (48, 47), (60, 20)])
def test_tracker_line_through_a_change_of_level(n8, n4):
    """beat_this tapped a sparse 72 BPM beat in 8ths, then in quarters. Their
    median interval (0.63 s) is on neither level: 97.9 BPM at 40 + 40, no
    line at all at 48 + 48. The line starts from the level tapped most."""
    beats = switching(72, 0.031, n8, n4)
    bpm, numbers, kept = tracker_line(beats)
    assert abs(bpm - 144) < 0.02 and len(kept) == len(beats) and numbers[-1] == n8 + 2 * n4


def test_estimate_through_a_change_of_level(kit):
    truth = drum_beat(kit, 72, 12)
    beats = switching(72, truth["beats"][0], 24, 24)  # median interval: 0.625 s
    tracker = {**tracked(truth), "beats": beats.tolist()}
    grid, info = estimate(truth["audio"], tracker, (100, 180))
    assert abs(grid.bpm - 144) < 0.05 and abs(bar_error_ms(grid, truth)) < 3
    assert info["drift_ms"] < 10 and info["tempo_map"] is None


@pytest.mark.parametrize("bpm, bars, half, rng, want", [
    (95, 8, False, (100.0, 180.0), 95),  # no octave of 95 in 100-180: the tracker's level (fit_grid: 126.7)
    (72, 8, False, (100.0, 180.0), 144),  # bpm_range prefers the octave it holds
    (72, 8, False, None, 72),  # no range: the tracker's level
    (160, 12, True, (100.0, 180.0), 160),  # the tracker at half tempo, as beat_this read PIRAMIT
])
def test_the_octave(kit, bpm, bars, half, rng, want):
    truth = drum_beat(kit, bpm, bars)
    grid, info = estimate(truth["audio"], tracked(truth, half=half, every=2 if half else None), rng)
    assert abs(grid.bpm - want) < 0.05
    assert abs(bar_error_ms(grid, truth)) < 3
    assert {a["level"] for a in info["alternatives"]} == {"x1", "x2", "x1/2", "x3/2", "x2/3"}
    assert sum(a["chosen"] for a in info["alternatives"]) == 1


def test_fit_grid_alone_misses_95_bpm(kit):
    """The case the tracker fixes: no octave of 95 lies in 100-180."""
    grid, _ = fit_grid(drum_beat(kit, 95, 8)["audio"], (100, 180))
    assert abs(grid.bpm / 95 - 4 / 3) < 0.01


def test_phase_moves_onto_the_attacks(kit):
    """The comb sits on onset-strength peaks, ~7 ms after these attacks."""
    truth = drum_beat(kit, 137, 12)
    old, _ = fit_grid(truth["audio"], (100, 180))
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180))
    assert 5 < 1000 * (old.phase - truth["beats"][0]) < 12
    assert abs(bar_error_ms(grid, truth)) < 2
    assert -12 < info["phase_correction_ms"] < -4 and info["attacks"]["attacks"] > 20


def test_bar_line_after_a_mid_bar_start(kit):
    """Cut 0.1 s before beat 2: the first grid beat is beat 2, the first bar line is bar 2's."""
    truth = drum_beat(kit, 137, 12, crop=0.03 + 60 / 137 - 0.1)
    old, _ = fit_grid(truth["audio"], (100, 180))
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180))
    assert abs(old.phase - truth["beats"][0]) < 0.012  # beat 2, taken for beat 1
    assert abs(grid.phase - truth["downbeats"][0]) < 0.002
    assert not info["downbeat"]["ambiguous"]


def test_three_four_from_the_downbeats(kit):
    truth = drum_beat(kit, 120, 12, beats_per_bar=3)
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180))
    assert (grid.beats_per_bar, grid.beat_unit) == (3, 4) and info["meter"]["accepted"]
    assert abs(grid.bar - 1.5) < 0.001 and grid.steps_per_bar == 12
    assert abs(bar_error_ms(grid, truth)) < 2
    forced, info = estimate(truth["audio"], tracked(truth), (100, 180), meter="4/4")
    assert forced.beats_per_bar == 4 and info["meter"] == {"source": "--meter"}


def test_unsure_meter_is_four_four(kit):
    """Downbeats every 2 tracker beats (beat_this at half tempo): not 3 or 4."""
    truth = drum_beat(kit, 160, 12)
    grid, info = estimate(truth["audio"], tracked(truth, half=True, every=2), (100, 180))
    assert grid.beats_per_bar == 4 and not info["meter"]["accepted"]


@pytest.mark.parametrize("swing, hats, want16", [(0.5, 1, 0.5), (0.66, 1, 0.66), (0.58, 1, 0.58), (0.5, 2, None)])
def test_swing(kit, swing, hats, want16):
    truth = drum_beat(kit, 95, 8, swing=swing, hats=hats)
    _, info = estimate(truth["audio"], tracked(truth), (100, 180))
    if want16 is None:  # hats on 8ths only: no 16ths to read
        assert info["swing"]["ratio_16"] is None
    else:
        assert abs(info["swing"]["ratio_16"] - want16) < 0.02
    assert abs(info["swing"]["ratio_8"] - 0.5) < 0.02


def test_drift_keeps_a_tempo_map_for_a_ramp(kit):
    steady = drum_beat(kit, 136, 16)
    _, info = estimate(steady["audio"], tracked(steady), (100, 180))
    assert info["drift_ms"] < 10 and info["tempo_map"] is None
    ramp = drum_beat(kit, 136, 16, ramp=0.03)
    grid, info = estimate(ramp["audio"], tracked(ramp), (100, 180))
    assert info["drift_ms"] > 50
    tempo_map = np.array(info["tempo_map"])
    assert len(tempo_map) >= 60 and np.all(np.diff(tempo_map) > 0)
    # the map follows the beats where the constant grid cannot
    near = [np.min(np.abs(tempo_map - b)) for b in ramp["beats"][2:-2]]
    assert np.median(near) < 0.01


def test_bpm_forces_the_tempo(kit):
    truth = drum_beat(kit, 137, 12)
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180), bpm=137.0)
    assert grid.bpm == 137.0 and info["level"]["level"] == "forced"
    assert abs(bar_error_ms(grid, truth)) < 2
    alone, _ = estimate(truth["audio"], None, (100, 180), bpm=137.0)
    assert alone.bpm == 137.0


def test_tracker_line_reads_through_frames_and_extra_beats():
    """20 ms frames make the median interval a whole number of frames; an
    extra off-beat is counted as half a beat and left out."""
    beats = np.round((0.031 + np.arange(150) * 60 / 68.5) / 0.02) * 0.02
    assert abs(60 / np.median(np.diff(beats)) - 68.5) > 0.3
    with_extra = np.sort(np.r_[beats, beats[40] + 30 / 68.5])
    bpm, numbers, kept = tracker_line(with_extra)
    assert abs(bpm - 68.5) < 0.01 and len(kept) == 150 and numbers[-1] == 149


def test_meter_text():
    assert parse_meter("6/8") == (6, 8) and Grid(120.0, 0.0, 6, 8).steps_per_bar == 12
    assert Grid(120.0, 0.0, 7, 8).bar == pytest.approx(1.75)
    with pytest.raises(ValueError):
        parse_meter("4/5")
