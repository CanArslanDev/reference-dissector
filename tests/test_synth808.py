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


def test_envelope_and_harmonics_match_their_direct_forms():
    """The O(n) moving mean and the sliced harmonic bands give what the
    convolution and the boolean masks gave (the 808 fit got twice as fast)."""
    import numpy as np
    from scipy.signal import hilbert

    from refdissect.audio import SR, samples
    from refdissect.describe import envelope_db, harmonics
    from refdissect.synth808 import Params, laid_out

    y = laid_out(Params(34, 55, 30, 1.2, 3, 2.5, 0.05, 0.3), int(0.65 * SR))
    env = np.abs(hilbert(y))
    w = samples(25)
    direct = np.sqrt(np.convolve(env ** 2, np.ones(w) / w, mode="same"))
    direct = 20 * np.log10(direct / direct.max() + 1e-9)
    assert np.max(np.abs(np.maximum(envelope_db(y, 25), -40) - np.maximum(direct, -40))) < 1e-4
    seg = y[int(0.15 * SR):int(0.6 * SR)] * np.hanning(int(0.6 * SR) - int(0.15 * SR))
    p = np.abs(np.fft.rfft(seg, 1 << 16)) ** 2
    f = np.fft.rfftfreq(1 << 16, 1 / SR)
    lv = [p[(f > h * 33.3 * 0.97) & (f < h * 33.3 * 1.03)].max() for h in range(1, 13)]
    masks = np.maximum(10 * np.log10(np.array(lv[1:]) / lv[0] + 1e-12), -60)
    assert np.allclose(harmonics(y, 33.3), masks)
