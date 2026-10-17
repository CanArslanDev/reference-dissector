import numpy as np

from refdissect.grid import Grid, fit_grid, offsets


def test_fit_grid_recovers_tempo_and_phase(beat):
    grid, info = fit_grid(beat["audio"], (100, 180))
    assert abs(grid.bpm - beat["bpm"]) < 0.02
    phase_error = (grid.phase - beat["phase"]) % grid.beat
    # the grid follows onset-strength peaks, a few ms after the attack
    assert min(phase_error, grid.beat - phase_error) < 0.010
    assert info["score"] > info["mean_onset_strength"]


def test_the_wrong_tempo_drifts(beat):
    """136.36 instead of 137 BPM: right on average near the start, 40 ms off by bar 20."""
    good = offsets(beat["audio"], Grid(beat["bpm"], beat["phase"]), "snare")["beat 3"]
    bad = offsets(beat["audio"], Grid(136.36, beat["phase"]), "snare", tolerance=0.06)["beat 3"]
    assert abs(good["median_ms"]) < 10 and good["p90_abs_ms"] < 12
    assert bad["p90_abs_ms"] > 30


def test_grid_positions():
    g = Grid(120.0, 0.5)
    assert g.step == 0.125 and g.bar == 2.0
    assert g.step_in_bar(0.5 + 3 * 0.125).item() == 3
    assert g.bar_of(0.5 + 2.0).item() == 1
    assert np.isclose(g.beats(1.0), 2.0)


def test_tuning_reads_detuned_tones_within_a_cent():
    """librosa.estimate_tuning was off by up to 60 cents on tones like these."""
    import numpy as np

    from refdissect.audio import SR
    from refdissect.grid import tuning

    t = np.arange(int(8 * SR)) / SR
    rng = np.random.default_rng(1)
    for cents in (-40, -3.3, 0, 23, 45):
        x = np.zeros_like(t)
        for k, midi in enumerate((36, 43, 48, 55, 60, 63, 67)):  # a held chord with harmonics
            f0 = 440 * 2 ** ((midi - 69 + cents / 100) / 12)
            seg = (t >= k * 0.5) & (t < k * 0.5 + 4)
            for h, a in ((1, 1.0), (2, 0.4), (3, 0.2)):
                x += seg * a * np.sin(2 * np.pi * h * f0 * t)
        x += 0.01 * rng.normal(size=len(t))
        assert abs(100 * tuning(x) - cents) < 1.0, cents
