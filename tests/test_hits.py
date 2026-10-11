import numpy as np

from refdissect.audio import PRE, SR, band
from refdissect.hits import group, merge, onset_template, template
from refdissect.xcorr import ncc


def _plant(kit, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(SR * 12).astype(np.float32) * 0.05  # loud bleed
    starts = {"snare": [], "hat": []}
    for k in range(20):
        for name, offset in (("snare", 0.0), ("hat", 0.25)):
            a = int((0.3 + k * 0.5 + offset) * SR)
            x[a:a + len(kit[name])] += kit[name]
            starts[name].append(a)
    return x, starts


def test_two_sounds_two_groups(kit):
    x, starts = _plant(kit)
    all_starts = sorted(starts["snare"] + starts["hat"])
    groups = group(x, starts=all_starts, cluster_ms=60, threshold=0.6)
    big = [g for g in groups if g.count >= 15]
    assert len(big) == 2
    for g in big:
        members = set(g.starts())
        assert members <= set(starts["snare"]) or members <= set(starts["hat"])


def test_averaging_beats_single_hits(kit):
    """Averaging N hits lowers the bleed by about sqrt(N)."""
    x, starts = _plant(kit)
    g = group(x, starts=starts["snare"], threshold=-1.0, cluster_ms=60)[0]
    t, used = template(x, g, 200)
    clean = np.r_[np.zeros(PRE, np.float32), kit["snare"]][: len(t)]
    single = [abs(ncc(x[s - PRE:s - PRE + len(t)], clean, 50)[0]) for s in starts["snare"]]
    assert used == 20
    assert abs(ncc(t, clean, 50)[0]) > max(single)


def test_merge_joins_split_groups(kit):
    x, starts = _plant(kit)
    half = len(starts["snare"]) // 2
    a = group(x, starts=starts["snare"][:half], threshold=-1.0)[0]
    b = group(x, starts=starts["snare"][half:], threshold=-1.0)[0]
    assert len(merge(x, [a, b], threshold=0.8)) == 1


def test_onset_template_puts_the_attack_at_pre(kit):
    x, starts = _plant(kit)
    late = [s + int(0.012 * SR) for s in starts["snare"]]  # hit times 12 ms after the attack
    g = group(band(x, "snare"), starts=late, threshold=-1.0, cluster_ms=60)[0]
    t, _ = onset_template(x, g, 100)
    env = np.abs(t)
    first = int(np.argmax(env > 0.3 * env.max()))
    assert abs(first - PRE) <= int(0.002 * SR)
