import numpy as np

from refdissect.audio import SR, attack_time, samples


def test_attack_time_skips_lead_in(kit):
    """A one-shot with 20 ms of quiet lead-in attacks 20 ms in, in every band."""
    for role in ("kick", "snare", "hat"):
        x = np.r_[np.zeros(samples(20), np.float32), kit[role]]
        assert abs(attack_time(x, role) - 0.020) < 0.002


def test_attack_time_reads_the_roles_band(kit):
    """A hat 30 ms after a loud kick: in the hat band the attack is the hat's."""
    x = kit["kick"].copy()
    a = samples(30)
    x[a:a + len(kit["hat"])] += 0.5 * kit["hat"]
    assert attack_time(x, "kick") < 0.003
    assert abs(attack_time(x, "hat") - 0.030) < 0.002
    assert attack_time(np.zeros(SR, np.float32), "snare") == 0.0
