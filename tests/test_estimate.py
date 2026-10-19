"""grid.estimate on beats whose grid is known, with beat_this mocked (tracked)."""

from dataclasses import replace

import numpy as np
import pytest
from conftest import drum_beat, six_eight, switch, tracked

from refdissect.audio import SR
from refdissect.grid import Grid, _attack, bar_line, drift, estimate, fit_grid, parse_meter, tempo_change, tracker_line
from refdissect.hits import strength_peaks


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
    (90, 8, False, (100.0, 180.0), 180),  # x2 refined to 180.004: the range's edge gives 0.5 %
    (50, 12, False, (100.0, 180.0), 100),  # x2 refined to 99.9996
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


def test_bar_line_votes_against_all_the_others():
    grid = Grid(120.0, 0.0)
    downbeats = [8 * grid.step + k * grid.bar for k in range(14)] + [k * grid.bar for k in range(14, 23)]
    _, down = bar_line(0.0, grid, downbeats)
    assert down["votes_16th_of_bar"] == {"8": 14, "0": 9} and down["ambiguous"]


def test_bar_line_is_unsure_when_the_tracker_bars_disagree(kit):
    """A third of the tracker's downbeats on beat 2 of a 3/4 beat: its bars
    hold 2, 3 or 4 beats, so it gives no meter to vote the bar line in."""
    truth = drum_beat(kit, 136, 16, beats_per_bar=3)
    tracker = tracked(truth)
    downbeats = np.array(tracker["downbeats"])
    downbeats[1::3] += 0.44
    tracker["downbeats"] = downbeats.tolist()
    grid, info = estimate(truth["audio"], tracker, (100, 180))
    assert info["downbeat"]["ambiguous"] and "disagree" in info["downbeat"]["why"]
    _, info = estimate(truth["audio"], tracked(truth), (100, 180))
    assert not info["downbeat"]["ambiguous"]


def test_three_four_from_the_downbeats(kit):
    truth = drum_beat(kit, 120, 12, beats_per_bar=3)
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180))
    assert (grid.beats_per_bar, grid.beat_unit) == (3, 4) and info["meter"]["accepted"]
    assert abs(grid.bar - 1.5) < 0.001 and grid.steps_per_bar == 12
    assert abs(bar_error_ms(grid, truth)) < 2
    forced, info = estimate(truth["audio"], tracked(truth), (100, 180), meter="4/4")
    assert forced.beats_per_bar == 4 and info["meter"] == {"source": "--meter"}


@pytest.mark.parametrize("bpm, rng", [(95, (100.0, 180.0)), (136, (100.0, 180.0)), (95, None)])
def test_six_eight_on_dotted_quarters(kit, bpm, rng):
    """beat_this taps 6/8 in dotted quarters, two to a bar. Its levels x1, x2
    and x1/2 read 95 as 63.3 or 126.7 (2/3 or 4/3 of it) and 136 as 90.7."""
    truth = six_eight(kit, bpm, 16)
    grid, info = estimate(truth["audio"], tracked(truth), rng)
    assert abs(grid.bpm - bpm) < 0.05 and (grid.beats_per_bar, grid.beat_unit) == (6, 8)
    assert info["level"]["level"] == "x3/2" and info["level"]["why"].startswith("dotted quarters")
    assert abs(bar_error_ms(grid, truth)) < 3 and not info["warnings"]


def test_six_eight_tapped_as_twelve_eight_needs_the_meter(kit):
    """Four dotted quarters to a bar (12/8), as beat_this read 6/8 at 136: the
    tracker's level stays, with a warning, unless --meter says x/8."""
    truth = six_eight(kit, 136, 16)
    tracker = tracked(truth, every=4)
    grid, info = estimate(truth["audio"], tracker, (100.0, 180.0))
    assert abs(grid.bpm - 136 * 2 / 3) < 0.05 and "--meter 6/8" in info["warnings"][0]
    grid, info = estimate(truth["audio"], tracker, (100.0, 180.0), meter="6/8")
    assert abs(grid.bpm - 136) < 0.05 and grid.steps_per_bar == 12 and abs(bar_error_ms(grid, truth)) < 3


def test_swing_with_two_beat_bars_stays_binary(kit):
    """66 % swing fits x3/2's 16ths best (on_16ths 1.00), but its onsets sit
    at 1/3 and 5/6 of the beat, never at 2/3: no dotted quarters."""
    truth = drum_beat(kit, 95, 12, swing=0.66)
    grid, info = estimate(truth["audio"], tracked(truth, every=2), (100.0, 180.0))
    assert abs(grid.bpm - 95) < 0.05 and info["level"]["level"] == "x1" and info["meter"]["two"]


def test_unsure_meter_is_four_four(kit):
    """Downbeats every 2 tracker beats (beat_this at half tempo): not 3 or 4."""
    truth = drum_beat(kit, 160, 12)
    grid, info = estimate(truth["audio"], tracked(truth, half=True, every=2), (100, 180))
    assert grid.beats_per_bar == 4 and not info["meter"]["accepted"]


@pytest.mark.parametrize("bpm, swing, hats, want16", [(95, 0.5, 1, 0.5), (95, 0.66, 1, 0.66), (95, 0.58, 1, 0.58),
                                                       (136, 0.66, 1, 0.66), (95, 0.5, 2, None)])
def test_swing(kit, bpm, swing, hats, want16):
    """From the hats' attacks: onset-strength peaks read 66 % up to 3.6
    points low, the hats on the 8th lines peaking later than the swung ones."""
    truth = drum_beat(kit, bpm, 12, swing=swing, hats=hats)
    _, info = estimate(truth["audio"], tracked(truth), (100, 180))
    if want16 is None:  # hats on 8ths only: no 16ths to read
        assert info["swing"]["ratio_16"] is None
    else:
        assert abs(info["swing"]["ratio_16"] - want16) < 0.002
    assert abs(info["swing"]["ratio_8"] - 0.5) < 0.002


def hats_at(truth: dict, kit: dict, times) -> dict:
    x = truth["audio"].copy()
    for t in times:
        a = int(round(t * SR))
        x[a:a + len(kit["hat"])] += 0.3 * kit["hat"][: max(0, min(len(kit["hat"]), len(x) - a))]
    return {**truth, "audio": x}


def test_no_swing_from_straight_beats(kit):
    """Straight beats with hats on 8ths read 0.74 (a kick sample's own second
    transient 158 ms in), 0.43 (a 32nd roll every 4th bar) and 0.51 (8th
    triplets every other bar; 0.60 on a library kit)."""
    kick = np.zeros(int(0.3 * SR), np.float32)
    kick[:len(kit["kick"])] = kit["kick"]
    a = int(0.158 * SR)
    kick[a:a + len(kit["hat"])] += 0.05 * kit["hat"]
    echo = drum_beat({**kit, "kick": kick}, 136, 16, beats_per_bar=3, hats=2)
    plain = drum_beat(kit, 160, 16, hats=2)
    q = 60 / 160
    rolls = hats_at(plain, kit, [plain["beats"][0] + (b * 4 + 3 + k / 8) * q for b in range(3, 16, 4)
                                 for k in range(8)])
    plain = drum_beat(kit, 140, 16, hats=2)
    q = 60 / 140
    triplets = hats_at(plain, kit, [plain["beats"][0] + (b * 4 + 2 + k * 2 / 3) * q for b in range(1, 16, 2)
                                    for k in range(6)])
    for truth in (echo, rolls, triplets):
        _, info = estimate(truth["audio"], tracked(truth), (100, 180))
        assert info["swing"]["ratio_16"] is None and abs(info["swing"]["ratio_8"] - 0.5) < 0.002


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


def test_no_attack_on_a_ringing_tail(kit):
    """A 55 Hz tail swings through 30 % of the window's peak a few ms into it:
    that 'attack' came 33 ms before the snare that follows the tail."""
    t = np.arange(SR) / SR
    tail = (0.5 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.4)).astype(np.float32)
    snare = np.zeros(SR, np.float32)
    a = int(0.08 * SR)
    snare[a:a + len(kit["snare"])] = 0.8 * kit["snare"]
    for x, want in ((tail + snare, None), (snare, 0.08)):
        peaks, _ = strength_peaks(x, "full")
        attack = _attack(x, peaks[np.argmin(np.abs(peaks - 0.088))], "full")
        assert attack is None if want is None else abs(attack - want) < 0.002


def test_drift_ignores_an_offset_and_stray_beats(kit):
    """A constant offset is no drift (it kept a map on calibration beat 5),
    nor are three tracker beats that sit nearer a swung 16th than their own
    onset (91 ms of 'drift' before)."""
    truth = drum_beat(kit, 136, 16, swing=0.66)
    tracker = tracked(truth)
    grid, _ = estimate(truth["audio"], tracker, (100, 180))
    bpm, numbers, beats = tracker_line(tracker["beats"])
    onsets, _ = strength_peaks(truth["audio"], "full")
    early = beats.copy()
    early[-4:-1] -= 0.04
    for g, b in ((replace(grid, phase=grid.phase - 0.014), beats), (grid, early)):
        drift_ms, tempo_map = drift(truth["audio"], g, numbers, b, grid.bpm / bpm, onsets)
        assert drift_ms < 5 and tempo_map is None


def on_lines(grid: Grid, beats) -> float:
    """Share of the beats within 20 ms of a grid line."""
    k = np.round((np.asarray(beats) - grid.phase) / grid.beat)
    return float(np.mean(np.abs(beats - (grid.phase + k * grid.beat)) < 0.02))


@pytest.mark.parametrize("first, second, bars", [(136, 160, 8), (160, 136, 6), (150, 130, 7)])
def test_a_beat_switch_follows_the_longer_tempo(kit, first, second, bars):
    """A line through both tempi matched neither: 136 then 160 read as 142.3,
    12 % of each part's beats on its lines. fit_grid matched one part. The
    longer part, in seconds, is the slower one here."""
    truth = switch(kit, first, second, bars)
    grid, info = estimate(truth["audio"], tracked(truth), (100, 180))
    change = info["tempo_change"]
    follows, bpm, beats = ("first", first, truth["first"]) if first < second else ("second", second, truth["second"])
    assert change["grid_follows"] == follows and abs(change["ratio"] - second / first) < 0.01
    assert change["until"] <= truth["cut"] + 1.5 and change["from"] >= truth["cut"] - 1.5
    assert abs(grid.bpm - bpm) < 0.05 and on_lines(grid, beats) == 1.0
    downbeats = truth["downbeats"][truth["downbeats"] >= (truth["cut"] - 0.01 if follows == "second" else 0)]
    assert abs(bar_error_ms(grid, {"downbeats": downbeats})) < 3


def test_no_tempo_change_in_a_ramp_or_a_steady_beat(kit):
    for truth in (drum_beat(kit, 136, 16, ramp=0.03), drum_beat(kit, 95, 16)):
        assert tempo_change(tracked(truth)["beats"]) is None


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
