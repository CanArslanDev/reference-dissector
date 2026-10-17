"""groove, space, rhythm, basicpitch, loudness, listening and mixsim on signals
whose answer is known."""

import numpy as np
import pytest

from refdissect.audio import SR
from refdissect.grid import Grid


def test_lufs_of_a_997_hz_sine():
    from refdissect.loudness import integrated_lufs, to_lufs
    t = np.arange(10 * SR) / SR
    x = 0.1 * np.sin(2 * np.pi * 997 * t)  # -20 dBFS peak: -23.0 LUFS for one channel
    assert integrated_lufs(x, SR) == pytest.approx(-23.0, abs=0.1)
    assert integrated_lufs(to_lufs(x, SR, -14.0), SR) == pytest.approx(-14.0, abs=0.01)
    assert integrated_lufs(np.zeros(SR), SR) == float("-inf")


def test_lufs_matches_ableton_live_mcp():
    am = pytest.importorskip("MCP_Server.audio_analysis")
    from refdissect.loudness import integrated_lufs
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, (5 * SR, 2))
    x[:, 1] = np.sin(2 * np.pi * 997 * np.arange(5 * SR) / SR) * 0.1
    assert integrated_lufs(x, SR) == pytest.approx(am.integrated_loudness(x, SR)[0], abs=0.1)


def ducked(kicks, depth_db=-20.0, back_s=0.3, seconds=8.0):
    """A 60 Hz tone that falls depth_db within 10 ms of each kick (a compressor's
    attack) and comes back linearly in dB by back_s."""
    t = np.arange(int(seconds * SR)) / SR
    g = np.zeros_like(t)
    for k in kicks:
        m = (t >= k) & (t < k + back_s)
        u = t[m] - k
        g[m] = np.minimum(g[m], depth_db * np.minimum(u / 0.01, 1) * (1 - u / back_s))
    return np.sin(2 * np.pi * 60 * t) * 10 ** (g / 20)


def test_kick_curve_reads_depth_and_recovery():
    from refdissect.groove import band_envelope, curve_distance, kick_curve
    kicks = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    c = kick_curve(band_envelope(ducked(kicks), 30, 150), kicks)
    assert c["kicks"] == 6
    assert c["depth_db"] == pytest.approx(-20, abs=2.5)
    assert c["back_to_-3db_s"] == pytest.approx(0.255, abs=0.03)
    flat = kick_curve(band_envelope(ducked(kicks, depth_db=0.0), 30, 150), kicks)
    assert abs(flat["depth_db"]) < 1 and curve_distance(c, flat) > 5


def test_drum_roles_balance():
    from refdissect.groove import balance_moves, cut_one_shots, hit_levels, role_offsets
    rng = np.random.default_rng(0)
    x = np.zeros(4 * SR)
    kick_t, hat_t = [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]
    for k in kick_t:
        i = int(k * SR)
        x[i:i + 2000] += np.sin(2 * np.pi * 60 * np.arange(2000) / SR)
    for h in hat_t:
        i = int(h * SR)
        x[i:i + 500] += 0.1 * rng.normal(size=500)
    levels = {"kick": hit_levels(x, kick_t, 30, 150), "hat": hit_levels(x, hat_t, 5000, 14000)}
    off = role_offsets(levels)
    assert off["kick"] == 0.0 and off["hat"] < -10
    moves = balance_moves({"kick": 0.0, "hat": -19.0}, {"kick": 0.0, "hat": -5.0})
    assert moves == {"kick": 0.0, "hat": -14.0}
    notes = [{"start_time": 0, "duration": 0.25}, {"start_time": 1, "duration": 1.0}]
    assert cut_one_shots(notes, 0.28, 135) == [notes[0]]  # 0.28 s is 0.63 beats at 135


def test_space_reads_width_tail_and_bandpass():
    from refdissect.space import band_correlation, fit_bandpass, median_spectrum_db, tail
    rng = np.random.default_rng(1)
    n = 4 * SR
    dry = rng.normal(size=n)
    wet = rng.normal(size=(n, 2))
    x = np.stack([dry, dry], 1) + wet  # D = R: r = 0.5 in every band
    r = band_correlation(x)
    assert all(v["r"] == pytest.approx(0.5, abs=0.08) for v in r.values())
    # a stop at 2 s followed by a 60 dB/s decay (rt60 1 s)
    t = np.arange(n) / SR
    y = rng.normal(size=n) * np.where(t < 2, 1.0, 0.1 * 10 ** (-3 * (t - 2)))
    tl = tail(y, 2.0)
    assert tl["drop_db"] == pytest.approx(-22.4, abs=2)  # -20 dB, then 40 ms of the -60 dB/s decay
    assert tl["rt60_s"] == pytest.approx(1.0, abs=0.15)
    # white noise through a band-pass, against the same noise unfiltered
    import scipy.signal as ss
    noise = rng.normal(size=8 * SR)
    filtered = noise
    for fc, kind in ((900, "high"), (3000, "low")):  # EQ Eight-style: a high-pass, then a low-pass
        filtered = ss.sosfilt(ss.butter(2, fc, btype=kind, fs=SR, output="sos"), filtered)
    f, a = median_spectrum_db(noise)
    _, b = median_spectrum_db(filtered)
    fit = fit_bandpass(f, b - a)
    assert fit["rms_error_db"] < fit["rms_error_db_without_filter"] / 3
    assert 750 < fit["highpass_hz"] < 1100 and 2500 < fit["lowpass_hz"] < 3600 and fit["order"] == 2


def test_onset_profile_finds_the_pattern():
    from refdissect.rhythm import common_pattern, onset_profile
    g = Grid(120.0, 0.5)
    x = np.zeros(int((0.5 + 4 * g.bar + 1) * SR))
    for b in range(4):
        for step in (0, 4, 7, 10, 14):
            i = int((g.phase + b * g.bar + step * g.step) * SR)
            x[i:i + 3000] += np.sin(2 * np.pi * 440 * np.arange(3000) / SR) * np.exp(-np.arange(3000) / 600)
    p = onset_profile(x, g, [0, 1, 2, 3], 200, 2000)
    assert common_pattern(p, 0.75) == ["0", "4", "7", "10", "14"]
    assert p["triplet_only"] == 0


def test_basic_pitch_frame_times():
    from refdissect.basicpitch import WINDOW_FRAMES, column, frame_at, frame_times
    t = frame_times(20000)
    naive = np.arange(20000) * 256 / 22050
    assert t[WINDOW_FRAMES - 1] == pytest.approx(naive[WINDOW_FRAMES - 1])
    assert naive[-1] - t[-1] == pytest.approx(1.2, abs=0.1)  # the drift a naive index makes
    assert frame_at(t, t[500]) == 500
    assert column(21) == 0 and column(108) == 87
    with pytest.raises(ValueError):
        column(15)


def test_ab_file_and_mix(tmp_path):
    from refdissect.listening import ab_file
    from refdissect.loudness import integrated_lufs
    from refdissect.mixsim import gains_to_targets, mix
    t = np.arange(3 * SR) / SR
    quiet, loud = 0.05 * np.sin(2 * np.pi * 300 * t), 0.5 * np.sin(2 * np.pi * 300 * t)
    info = ab_file(quiet, loud, SR, tmp_path / "ab.wav")
    import soundfile as sf
    y, _ = sf.read(tmp_path / "ab.wav")
    a, b = y[:3 * SR], y[-3 * SR:]
    assert integrated_lufs(a, SR) == pytest.approx(integrated_lufs(b, SR), abs=0.1)
    assert info["switch_at_s"] > 3
    g = gains_to_targets({"pad": quiet, "lead": loud}, {"pad": -20.0, "lead": -20.0}, SR)
    m = mix({"pad": quiet, "lead": loud}, {k: v["gain_db"] for k, v in g.items()})
    assert integrated_lufs(m["audio"], SR) == pytest.approx(-20 + 6.02, abs=0.2)  # two equal sines in phase
    assert mix({"x": loud}, {"x": 12})["clipped_samples"] > 0
