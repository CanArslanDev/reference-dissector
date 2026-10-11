import numpy as np

from refdissect.audio import SR, shift
from refdissect.xcorr import ncc, sliding_ncc


def test_lag_convention():
    """ncc gives lag d when t is c delayed by d; shift(c, d) lines c up with t."""
    rng = np.random.default_rng(0)
    c = (rng.standard_normal(1000) * np.exp(-np.arange(1000) / 100)).astype(np.float32)
    t = shift(c, 7)
    score, lag = ncc(t, c, 20)
    assert lag == 7 and score > 0.999
    assert np.allclose(shift(c, lag), t)


def test_inverted_copy_scores_minus_one():
    rng = np.random.default_rng(1)
    c = rng.standard_normal(500).astype(np.float32)
    score, lag = ncc(-c, c, 5)
    assert lag == 0 and score < -0.999


def test_sliding_ncc_finds_planted_copies(kit):
    x = np.random.default_rng(2).standard_normal(SR * 3).astype(np.float32) * 0.01
    where = [SR // 2, SR + 1234, 2 * SR + 77]
    for a in where:
        x[a:a + len(kit["snare"])] += kit["snare"]
    score, gain = sliding_ncc(x, kit["snare"], "snare")
    for a in where:
        assert score[a] > 0.99
        assert abs(gain[a] - 1.0) < 0.05
    assert np.sort(score)[-len(where) * 50] < 0.5  # nothing else comes close
