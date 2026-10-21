"""layers.explain on synthetic beats whose answer is known: hits under other
hits, layers, flams, rolls and pitched hats, and what must not be found."""

import numpy as np
import pytest
from conftest import burst

from refdissect import layers
from refdissect.audio import SR, transpose
from refdissect.detect import MAX_MS, Template, detect
from refdissect.hits import Group, template

BPM = 136.0
STEP = 60 / BPM / 4


@pytest.fixture(scope="module")
def sounds(kit):
    return {**kit, "clap": burst(4, 200, 700, 5000, 45), "tom": burst(5, 400, 60, 400, 150, tone_hz=110.0)}


def render(sounds: dict, hits: list[tuple[str, float, float]], seconds: float) -> np.ndarray:
    """hits: (sound, time s, gain); sounds may hold pitched copies (name+st)."""
    x = np.random.default_rng(9).standard_normal(int(seconds * SR)).astype(np.float32) * 0.003
    for name, t, gain in hits:
        y, a = sounds[name], int(round(t * SR))
        x[a:a + len(y)] += gain * y[: max(0, min(len(y), len(x) - a))]
    return x


def oracle(x: np.ndarray, hits: list[tuple[str, float, float]], name: str, role_band: str,
           alone: float = 0.03) -> Template:
    """The template of `name` from its hits that sound alone (nothing else
    within `alone` s), or from all of them when it never does (a kick on a
    hat every time: its template holds the hat, as discover()'s would)."""
    others = np.array([t for n, t, _ in hits if n != name])
    mine = [t for n, t, _ in hits if n == name]
    times = [t for t in mine if not len(others) or np.min(np.abs(others - t)) > alone] or mine
    g = Group(members=[(int(round(t * SR)), 0, 1.0, 1.0, 1.0) for t in times])
    audio, used = template(x, g, MAX_MS)  # the attack exactly PRE into the window
    return Template(name, "kick" if role_band == "low" else role_band, role_band, audio, used, {}, g)


def found(events: dict, name: str, times, tol: float = 0.003) -> np.ndarray:
    """Which of times have a hit of name within tol (its t is the attack:
    onset_template puts the attack PRE into the window)."""
    t = np.array([e["t"] for e in events[name]])
    return np.array([bool(len(t)) and float(np.min(np.abs(t - s))) <= tol for s in times])


def extra(events: dict, name: str, times, tol: float = 0.003) -> int:
    """Hits of name with no truth time within tol."""
    times = np.asarray(times)
    return sum(1 for e in events[name] if not len(times) or np.min(np.abs(times - e["t"])) > tol)


def backbeat(bars: int = 12, hats_under_snare: bool = True) -> list[tuple[str, float, float]]:
    """Kick on 0 and 10, snare on 4 and 12, hats on every 8th (0.25-0.4),
    under the snare too."""
    rng = np.random.default_rng(5)
    hits = []
    for b in range(bars):
        for s in range(16):
            t = 0.05 + (b * 16 + s) * STEP
            if s in (0, 10):
                hits.append(("kick", t, 0.9))
            if s in (4, 12):
                hits.append(("snare", t, 0.8))
            if s % 2 == 0 and (hats_under_snare or s not in (4, 12)):
                hits.append(("hat", t, float(rng.uniform(0.25, 0.4))))
    return hits


def explain(x, templates):
    return detect(x, templates)


def test_hats_under_every_snare(sounds):
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    snares = np.array([t for n, t, _ in hits if n == "snare"])
    hats = [t for n, t, _ in hits if n == "hat"]
    under = [t for t in hats if np.min(np.abs(snares - t)) < 0.001]
    assert found(events, "hat", under).all()  # 24 of 24; one winner per band kept 0
    assert found(events, "hat", hats).mean() == 1.0 and extra(events, "hat", hats) == 0
    assert found(events, "snare", snares).all() and extra(events, "snare", snares) == 0
    assert all(e["source"] == "layer" and e["under"] == "snare" for e in events["hat"]
               if np.min(np.abs(snares - e["t"])) < 0.003)


def test_a_clap_layered_on_a_snare_0_to_8_ms_late(sounds):
    """Clap alone on 6, snare alone on 12, both on 4 (the clap 0-8 ms late)."""
    rng = np.random.default_rng(7)
    hits = []
    for b in range(16):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("kick", o + 10 * STEP, 0.9), ("clap", o + 6 * STEP, 0.7),
                 ("snare", o + 12 * STEP, 0.8), ("snare", o + 4 * STEP, 0.8),
                 ("clap", o + 4 * STEP + rng.uniform(0, 0.008), 0.7)]
        hits += [("hat", o + s * STEP, 0.3) for s in range(0, 16, 2)]
    x = render(sounds, hits, 0.05 + 16 * 16 * STEP + 1)
    templates = [oracle(x, hits, n, b) for n, b in (("hat", "hat"), ("kick", "low"), ("snare", "snare"),
                                                    ("clap", "snare"))]
    events = explain(x, templates)
    for name in ("clap", "snare", "hat"):
        times = [t for n, t, _ in hits if n == name]
        assert found(events, name, times).all(), name  # the clap on the snare: recall 0.5 before
        assert extra(events, name, times) == 0, name


def test_flams_25_ms(sounds):
    """Every snare has a grace note 25 ms before it at -6 dB."""
    hits = [h for h in backbeat(hats_under_snare=False)]
    grace = [("snare", t - 0.025, 0.4) for n, t, _ in hits if n == "snare"]
    hits += grace
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    main = [t for n, t, g in hits if n == "snare" and g > 0.5]
    alone = [(n, t, g) for n, t, g in hits if (n, t, g) not in grace]
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, alone, "snare", "snare")]
    events = explain(x, templates)
    graces = [t for _, t, _ in grace]
    assert found(events, "snare", main).all() and found(events, "snare", graces).all()
    assert extra(events, "snare", main + graces) == 0
    flams = {e["t"]: e.get("flam_of") for e in events["snare"]}
    assert all(flams[min(flams, key=lambda u: abs(u - t))] is not None for t in graces)
    assert all(flams[min(flams, key=lambda u: abs(u - t))] is None for t in main)


def test_64th_hat_rolls(sounds):
    """A beat of 64ths (27.6 ms apart at 136 BPM) at the end of every bar."""
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("snare", o + 4 * STEP, 0.8), ("kick", o + 8 * STEP, 0.9)]
        hits += [("hat", o + s * STEP, 0.35) for s in range(0, 12, 2)]
        hits += [("hat", o + 12 * STEP + i * STEP / 4, 0.25 + 0.01 * i) for i in range(16)]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    hats = [t for n, t, _ in hits if n == "hat"]
    assert found(events, "hat", hats).mean() >= 0.95  # 35 ms apart at least: 0.25 of a 64th roll
    assert extra(events, "hat", hats) <= 2


def test_a_pitched_hat_roll(sounds):
    """32nds climbing 1 to 5 semitones, the hat as Simpler transposes it."""
    sounds = {**sounds, **{f"hat{st:+d}": transpose(sounds["hat"], st) for st in range(1, 6)}}
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("snare", o + 4 * STEP, 0.8), ("kick", o + 8 * STEP, 0.9)]
        hits += [("hat", o + s * STEP, 0.35) for s in range(0, 12, 2)]
        hits += [(f"hat{1 + i // 2:+d}", o + 12 * STEP + i * STEP / 2, 0.3) for i in range(8) if i // 2 < 5]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    for st in range(1, 5):
        times = [t for n, t, _ in hits if n == f"hat{st:+d}"]
        pitched = {"hat": [e for e in events["hat"] if e.get("semitones") == st]}
        assert found(pitched, "hat", times, tol=0.004).mean() >= 0.9, st
        assert extra(pitched, "hat", times, tol=0.004) == 0, st
    plain = {"hat": [e for e in events["hat"] if not e.get("semitones")]}
    assert extra(plain, "hat", [t for n, t, _ in hits if n == "hat"]) == 0


def test_no_layers_on_plain_hits(beat):
    """The conftest beat (nothing plays under anything but the hats on the
    kicks): every hit found, nothing added under another sound."""
    from refdissect.detect import discover
    from refdissect.grid import fit_grid
    templates = discover(beat["audio"], fit_grid(beat["audio"], (100, 180))[0])
    events = detect(beat["audio"], templates)
    added = [e for evs in events.values() for e in evs if e["source"] == "layer"]
    truth = np.sort(np.concatenate([np.array(v) for v in beat["truth"].values()]))
    assert all(np.min(np.abs(truth - e["t"])) < 0.005 for e in added)


def test_one_hit_per_strike_of_a_ringing_sound(sounds):
    """A tom ringing 400 ms (110 Hz) on every beat, a snare on 2 and 4: one
    hit per strike, none on the decay."""
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("tom", o + s * STEP, 0.8) for s in (0, 4, 8, 12)]
        hits += [("snare", o + s * STEP + 2 * STEP, 0.6) for s in (4, 12)]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "tom", "low", alone=0.1), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    for name in ("tom", "snare"):
        times = [t for n, t, _ in hits if n == name]
        assert found(events, name, times).all() and extra(events, name, times) == 0, name


def test_a_hit_at_zero_seconds(sounds):
    """The stem is padded: a hit attacking at 0 s is found (detect.py
    started its windows PRE before an attack, so it could not)."""
    hits = [("snare", 0.0, 0.8)] + [("snare", 0.3 + 0.4 * k, 0.8) for k in range(12)]
    x = render(sounds, hits, 6.0)
    events = explain(x, [oracle(x, hits[1:], "snare", "snare")])
    assert found(events, "snare", [t for _, t, _ in hits]).all()


def test_velocity_is_the_joint_least_squares_gain(sounds):
    """Gains follow the hits' levels, a hat under a snare included."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    t = np.array([e["t"] for e in events["hat"]])
    g = np.array([e["gain"] for e in events["hat"]])
    truth = [(time, gain) for n, time, gain in hits if n == "hat"]
    pairs = [(g[np.argmin(np.abs(t - time))], gain) for time, gain in truth]
    est, true = np.log(np.array(pairs)).T
    err_db = 20 / np.log(10) * (est - true - np.median(est - true))
    assert np.corrcoef(est, true)[0, 1] > 0.95 and np.percentile(np.abs(err_db), 90) < 1.0


def test_explain_reports_what_it_decided(sounds):
    hits = backbeat(bars=8)
    x = render(sounds, hits, 0.05 + 8 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    from refdissect.detect import explain as detect_explain
    info = detect_explain(x, templates).info
    assert "hat" in info["purified"]["snare"]  # the snare's template holds the hat on every snare
    assert info["tonal"] == ["kick"] and set(info["usual_gain"]) == {"hat", "kick", "snare"}
    assert layers.SEMITONES and info["pitched"]["hat"] < layers.SHARP


def test_silence_and_a_lone_template(sounds):
    """Nothing to find in silence, and one template finds its own hits."""
    hits = [("snare", 0.3 + 0.4 * k, 0.8) for k in range(10)]
    x = render(sounds, hits, 5.0)
    snare = oracle(x, hits, "snare", "snare")
    assert explain(np.zeros(SR * 3, np.float32), [snare]) == {"snare": []}
    assert layers.explain(x, [], {}, {}).events == {}
    events, times = explain(x, [snare]), [t for _, t, _ in hits]
    assert found(events, "snare", times).all() and extra(events, "snare", times) == 0
