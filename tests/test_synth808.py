import numpy as np

from refdissect.audio import SR
from refdissect.synth808 import Params, compare, fit, laid_out, one_shot, voice


def _zero_crossing_hz(x):
    s = np.signbit(x)
    crossings = np.flatnonzero(s[1:] != s[:-1])
    return (len(crossings) - 1) / 2 / ((crossings[-1] - crossings[0]) / SR)


TRUE = Params(f_end_hz=41.2, f_start_hz=70.0, glide_ms=40, decay_s=0.8, attack_ms=2, drive=2.0, asym=0.05,
              phase=0.3)


def test_voice_pitch_and_drop():
    y = voice(TRUE, 1.0)
    assert abs(_zero_crossing_hz(y[int(0.4 * SR):]) - 41.2) < 0.5
    assert abs(TRUE.glide_semitones - 12 * np.log2(70 / 41.2)) < 1e-9
    moved = voice(TRUE, 1.0, pitch_hz=55.0)
    assert abs(_zero_crossing_hz(moved[int(0.4 * SR):]) - 55.0) < 0.6


def test_one_shot_level_and_fade():
    y = one_shot(TRUE, 41.2, seconds=1.5)
    assert abs(20 * np.log10(np.max(np.abs(y))) + 1.0) < 0.05
    assert abs(y[-1]) < 1e-3


def test_fit_gets_close():
    reference = laid_out(TRUE, int(0.6 * SR))
    p, info = fit(reference, 41.2, maxiter=25, popsize=10, seed=0)
    m = compare(reference, laid_out(p, len(reference)), 41.2)
    assert m["ncc"] > 0.9
    assert abs(p.f_end_hz - 41.2) < 1.5
