"""layers.explain on synthetic beats whose answer is known: hits under other
hits, layers, flams, rolls and pitched hats, and what must not be found."""

import numpy as np
import pytest
from conftest import burst

from refdissect import layers
from refdissect.audio import PRE, SR, fit, samples, transpose
from refdissect.detect import DETECT_MS, MAX_MS, THRESHOLD, Template, detect
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


def averaged(x: np.ndarray, hits: list[tuple[str, float, float]], name: str, role_band: str) -> Template:
    """The template of `name` averaged over all its hits, as discover()
    averages a grid slot's: whatever plays with some of them is in it."""
    g = Group(members=[(int(round(t * SR)), 0, 1.0, 1.0, 1.0) for n, t, _ in hits if n == name])
    audio, used = template(x, g, MAX_MS)
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


@pytest.mark.parametrize("late", ["0-8 ms", "0 ms"])
def test_a_clap_layered_on_a_snare(sounds, late):
    """Clap alone on 6, snare alone on 12, both on 4, the clap 0-8 ms late
    or exactly on the snare. On it, the clap's and the snare's templates
    each hold half the other: the clap lost its moments under the snare as
    one sound found twice, half its hits, was merged into the snare, and its
    solo hits went too (0 of 32). Half no longer merges: every solo clap is
    found; the 16 under the snare are not."""
    rng = np.random.default_rng(7)
    hits = []
    for b in range(16):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("kick", o + 10 * STEP, 0.9), ("clap", o + 6 * STEP, 0.7),
                 ("snare", o + 12 * STEP, 0.8), ("snare", o + 4 * STEP, 0.8),
                 ("clap", o + 4 * STEP + (rng.uniform(0, 0.008) if late == "0-8 ms" else 0.0), 0.7)]
        hits += [("hat", o + s * STEP, 0.3) for s in range(0, 16, 2)]
    x = render(sounds, hits, 0.05 + 16 * 16 * STEP + 1)
    templates = [oracle(x, hits, n, b) for n, b in (("hat", "hat"), ("kick", "low"), ("snare", "snare"),
                                                    ("clap", "snare"))]
    events = explain(x, templates)
    for name in ("clap", "snare", "hat"):
        times = [t for n, t, _ in hits if n == name]
        solo = [t for t in times if name != "clap" or late == "0-8 ms" or abs((t - 0.05) / STEP % 16 - 6) < 0.01]
        assert found(events, name, solo).all(), name  # the clap on the snare: recall 0.5 before, 0 at 0 ms
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
    assert found(events, "hat", hats).mean() >= 0.95  # hits 35 ms apart, as detect() kept them: 0.25
    assert extra(events, "hat", hats) <= 2


def test_a_pitched_hat_roll(sounds):
    """32nds climbing 1 to 4 semitones, two at each, the hat as Simpler
    transposes it."""
    sounds = {**sounds, **{f"hat{st:+d}": transpose(sounds["hat"], st) for st in range(1, 5)}}
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("snare", o + 4 * STEP, 0.8), ("kick", o + 8 * STEP, 0.9)]
        hits += [("hat", o + s * STEP, 0.35) for s in range(0, 12, 2)]
        hits += [(f"hat{1 + i // 2:+d}", o + 12 * STEP + i * STEP / 2, 0.3) for i in range(8)]
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


def identify(events: list[dict], truth: dict) -> tuple[str, float, np.ndarray]:
    """The truth sound a template's hits play, the template's offset from it
    (s, a discovered window need not start PRE before the attack) and which
    hits sit at that offset (3 ms): the sound with the most such hits, then
    the smallest offset."""
    t = np.array([e["t"] for e in events])
    best = None
    for name, times in truth.items():
        times = np.asarray(times)
        d = np.array([times[np.argmin(np.abs(times - u))] - u for u in t])
        off = float(np.median(d))
        at = np.abs(d - off) <= 0.003
        if best is None or (at.sum(), -abs(off)) > (best[2].sum(), -abs(best[1])):
            best = (name, off, at)
    return best


def discovered(beat: dict) -> list[Template]:
    from refdissect.detect import discover
    from refdissect.grid import fit_grid
    return discover(beat["audio"], fit_grid(beat["audio"], (100, 180))[0])


def test_the_conftest_beat_with_discovered_templates(beat):
    """The conftest beat (a hat on every eighth, under the 20 snares and the
    40 kicks too) with discover()'s templates, as the pipeline runs: every
    hit found once, the hat under each snare as a layer under it, and every
    layer hit on a hit of its own template's sound. The hat template starts
    23 ms before its hats' attack, so the hat sits at -23 ms in the snare:
    held lags of -10..30 ms found 0 of the 20."""
    found_ = detect(beat["audio"], discovered(beat))
    heard = {name: identify(evs, beat["truth"]) for name, evs in found_.items() if evs}
    for sound, times in beat["truth"].items():
        mine = [name for name, (s, _, _) in heard.items() if s == sound]
        hits = sum(int(heard[name][2].sum()) for name in mine)
        extra_ = sum(len(found_[name]) - int(heard[name][2].sum()) for name in mine)
        assert hits == len(times) and extra_ == 0, sound  # 160 hats, 40 kicks, 20 snares
    hat = next(name for name, (s, _, _) in heard.items() if s == "hat")
    off = heard[hat][1]
    snares = np.array(beat["truth"]["snare"])
    under = [e for e in found_[hat] if np.min(np.abs(snares - (e["t"] + off))) < 0.003]
    assert len(under) == 20 and all(e["source"] == "layer" for e in under)
    for name, evs in found_.items():
        if any(e["source"] == "layer" for e in evs):
            at = heard[name][2]
            assert all(ok for e, ok in zip(evs, at) if e["source"] == "layer"), name


def test_no_hat_under_a_kick_that_has_none(sounds):
    """Kicks 2 +- 4 ms after the hats of steps 0 and 10, and alone on 7, with
    templates averaged over every hit: the hat's carries the kicks' body,
    the kick's blurs the hats. Read over all bands, the kick template held
    the hat through that body (c 0.41), its purified column held a negative
    hat, and a hat was found on 10 of the 16 lone kicks."""
    rng = np.random.default_rng(3)
    hits = []
    for b in range(16):
        o = 0.05 + b * 16 * STEP
        hits += [("hat", o + s * STEP, float(rng.uniform(0.25, 0.4))) for s in range(0, 16, 2)]
        hits += [("kick", o + s * STEP + (2 + rng.uniform(-4, 4)) / 1000, 0.9) for s in (0, 10)]
        hits += [("kick", o + 7 * STEP, 0.9)] + [("snare", o + s * STEP, 0.8) for s in (4, 12)]
    x = render(sounds, hits, 0.05 + 16 * 16 * STEP + 1)
    events = explain(x, [averaged(x, hits, n, b) for n, b in (("hat", "hat"), ("kick", "low"), ("snare", "snare"))])
    hats = [t for n, t, _ in hits if n == "hat"]
    assert found(events, "hat", hats).all() and extra(events, "hat", hats) == 0


@pytest.mark.parametrize("kick_gain", [0.9, 0.45])
def test_hats_under_the_snare_when_kicks_land_on_hats(sounds, kick_gain):
    """discover()'s templates on a beat whose kicks land on 2 of every 8
    hats. The hat's average carries their body, so its kick band calls it a
    tone (38 % of its energy there at full level, ringing), and tones were
    never held: the 24 hats under the snares were lost. At full level a
    second hat template, found in the snare band 22 ms late, is held at -22
    ms (lags of -10..30 ms missed it); at half level there is none, and the
    hat-band template itself is held."""
    rng = np.random.default_rng(5)
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o + s * STEP, kick_gain) for s in (0, 10)] + [("snare", o + s * STEP, 0.8) for s in (4, 12)]
        hits += [("hat", o + s * STEP, float(rng.uniform(0.25, 0.4))) for s in range(0, 16, 2)]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    truth = {name: [t for n, t, _ in hits if n == name] for name in ("kick", "snare", "hat")}
    events = detect(x, discovered({"audio": x}))
    heard = {name: identify(evs, truth) for name, evs in events.items() if evs}
    hat = max((name for name, (s, _, _) in heard.items() if s == "hat"), key=lambda name: heard[name][2].sum())
    off, snares = heard[hat][1], np.array(truth["snare"])
    under = [t for t in truth["hat"] if np.min(np.abs(snares - t)) < 0.001]
    t = np.array([e["t"] for e in events[hat]]) + off
    assert all(np.min(np.abs(t - u)) <= 0.003 for u in under)
    assert heard[hat][2].all()  # every hit of the hat template on a hat


def test_merges_until_a_round_merges_none(sounds, monkeypatch):
    """explain() merges until a round merges nothing, so the hits it keeps
    index the list it merged last: with three rounds at most, a third round
    that still merged left them indexing the list before it (IndexError on
    this beat). Forced here: the column with the fewest candidates into the
    one with the most, in each of three rounds."""
    hits = []
    for b in range(16):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("kick", o + 10 * STEP, 0.9), ("clap", o + 6 * STEP, 0.7),
                 ("snare", o + 12 * STEP, 0.8), ("snare", o + 4 * STEP, 0.8), ("clap", o + 4 * STEP, 0.7)]
        hits += [("hat", o + s * STEP, 0.3) for s in range(0, 16, 2) if s != 6]
    x = render(sounds, hits, 0.05 + 16 * 16 * STEP + 1)
    templates = [oracle(x, hits, n, b) for n, b in (("hat", "hat"), ("kick", "low"), ("snare", "snare"),
                                                    ("clap", "snare"))]
    rounds = []

    def three_rounds(cands, pairs, first, kind, clean=None, band=None):
        rounds.append(1)
        cols = np.unique(cands.col)
        if len(rounds) > 3 or len(cols) < 2:
            return {}
        count = {int(c): int(np.sum(cands.col == c)) for c in cols}
        return {min(count, key=count.get): (max(count, key=count.get), 0)}

    monkeypatch.setattr(layers, "duplicates", three_rounds)
    events = explain(x, templates)
    assert len(rounds) == 4 and sum(len(v) for v in events.values()) > 0


@pytest.mark.parametrize("found_in", ["low", "snare"])
def test_one_hit_per_strike_of_a_ringing_sound(sounds, found_in):
    """A tom ringing 400 ms (110 Hz) on every beat, a snare on 2 and 4: one
    hit per strike, none on the decay, whichever band discover() found the
    tom in. Found in the snare band, its candidates were 8 ms apart and its
    ringing counted only in the low band: one extra hit per strike."""
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("tom", o + s * STEP, 0.8) for s in (0, 4, 8, 12)]
        hits += [("snare", o + s * STEP + 2 * STEP, 0.6) for s in (4, 12)]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    tom = oracle(x, hits, "tom", "low", alone=0.1)
    tom = Template("tom", "kick", found_in, tom.audio, tom.hits, {}, tom.group)
    templates = [tom, oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    for name in ("tom", "snare"):
        times = [t for n, t, _ in hits if n == name]
        assert found(events, name, times).all() and extra(events, name, times) == 0, name


def test_a_hit_at_zero_seconds(sounds):
    """The stem is padded: a hit attacking at 0 s is found (detect.py
    started its windows PRE before an attack, so it could not), and one
    whose attack the stem cuts off 3 ms in is found at 0 s, not at -3 ms
    (the plan wrote a note with a negative start)."""
    hits = [("snare", 0.0, 0.8)] + [("snare", 0.3 + 0.4 * k, 0.8) for k in range(12)]
    x = render(sounds, hits, 6.0)
    events = explain(x, [oracle(x, hits[1:], "snare", "snare")])
    assert found(events, "snare", [t for _, t, _ in hits]).all()
    cut = x[int(0.003 * SR):]
    events = explain(cut, [oracle(cut, [(n, t - 0.003, g) for n, t, g in hits[1:]], "snare", "snare")])
    assert min(e["t"] for e in events["snare"]) == 0.0


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


def test_hits_carry_where_their_sound_starts(sounds):
    """Each hit carries t_attack: its window start plus the template's
    attack_s, and a pitched copy's that much sooner as Simpler plays it
    faster. Its t (PRE after the window start) puts a hat 4 semitones up
    1 ms late; its t_attack is on the hat."""
    sounds = {**sounds, **{f"hat{st:+d}": transpose(sounds["hat"], st) for st in range(1, 5)}}
    hits = []
    for b in range(12):
        o = 0.05 + b * 16 * STEP
        hits += [("kick", o, 0.9), ("snare", o + 4 * STEP, 0.8), ("kick", o + 8 * STEP, 0.9)]
        hits += [("hat", o + s * STEP, 0.35) for s in range(0, 12, 2)]
        hits += [(f"hat{1 + i // 2:+d}", o + 12 * STEP + i * STEP / 2, 0.3) for i in range(8)]
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    events = explain(x, templates)
    for tp in templates:
        for e in events[tp.name]:
            start = e["t"] - PRE / SR
            assert e["t_attack"] == pytest.approx(start + tp.attack_s * 2 ** (-e.get("semitones", 0) / 12), abs=2e-4)
    up4 = np.array([t for n, t, _ in hits if n == "hat+4"])
    late = [(e["t"] - up4[np.argmin(np.abs(up4 - e["t"]))], e["t_attack"] - up4[np.argmin(np.abs(up4 - e["t"]))])
            for e in events["hat"] if e.get("semitones") == 4]
    t_err, attack_err = np.abs(np.array(late)).T
    assert len(late) >= 20 and np.median(t_err) > 0.0008 and np.median(attack_err) < 0.0003


def test_a_residual_template_keeps_its_own_hits(sounds):
    """A template of the hat's sound and another one (as a residual
    template of discover() can be, on loop transients htdemucs left in the
    drum stem): held under the hat, it found all 72 hats away from the snare
    a second time. Found by the residual pass, it keeps its matched-filter
    hits only; found on the grid from fewer hits than the hat, it is a
    variant of the hat (NEAR_MS) and not held either. The hats stay found
    once each."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    hat = templates[0].audio
    other = fit(np.r_[np.zeros(PRE, np.float32), burst(11, 120, 4000, 12000, 20)], len(hat))
    mixed = (0.6 * hat + 0.5 * np.max(np.abs(hat)) * other).astype(np.float32)
    hats = [t for n, t, _ in hits if n == "hat"]
    from refdissect.detect import explain as detect_explain
    for kind in ("residual", "grid"):
        got = detect_explain(x, templates + [Template("mixed", "hat", "hat", mixed, 3, {}, None, found=kind)])
        assert not [e for e in got.events["mixed"] if e["source"] == "layer" or e.get("semitones")], kind
        assert "mixed" not in got.info["purified"].get("hat", {}), kind
        assert found(got.events, "hat", hats).all() and extra(got.events, "hat", hats) == 0, kind


def _snare_holding_a_hat(seed: int, hat_gain: float, hat_low: float, top_gain: float, top_ms: float, hat_ms: float):
    """A snare template's fit bands (kick, snare, hat rows) with a hat at
    hat_ms (its hat-band part times hat_gain, its low part times hat_low)
    and at top_ms a top end of the snare's own, partly like the hat; and
    the hat's."""
    rng = np.random.default_rng(seed)
    n = samples(40)
    decay = np.exp(-np.arange(n) / samples(10))
    hat = np.zeros((3, n))
    hat[2], hat[1] = rng.standard_normal(n) * decay, rng.standard_normal(n) * decay
    top = rng.standard_normal(n) * decay
    snare = 0.8 * rng.standard_normal((3, samples(120)))
    snare[2] = 0.0
    at, later = samples(hat_ms), samples(top_ms)
    snare[2, at:at + n] += hat_gain * hat[2]
    snare[1, at:at + n] += hat_low * hat[1]
    snare[2, later:later + n] += top_gain[0] * hat[2] + top_gain[1] * top
    return snare, hat


@pytest.mark.parametrize("case", ["body", "top"])
def test_a_held_sound_sits_where_it_plays_with_its_holder(case):
    """Where a held template sits: of where it fits best in its own band and
    over all bands, the one nearer the holder's start. Over all bands a
    snare's body like the hat's low part put the hat 21 ms after the hat it
    plays with (tempo95-1: 23.8 ms over all bands, 2.6 in the hat band);
    in the hat band alone a snare's own top 28 ms on beat it (producer-1:
    28.7 ms, 2.4 over all bands)."""
    if case == "body":  # the snare's body at 23.8 ms like the hat's low part
        snare, hat = _snare_holding_a_hat(0, 1.0, 0.0, (0.6, 0.8), 23.8, 2.6)
        snare[1, samples(23.8):samples(23.8) + hat.shape[1]] += 2.0 * hat[1]
        want = 2.6
    else:  # the snare's own top at 28.7 ms, more like the hat in its band than the hat under it
        snare, hat = _snare_holding_a_hat(1, 0.8, 1.0, (1.0, 0.5), 28.7, 2.4)
        want = 2.4
    held = layers._held([snare, hat], [samples(60), hat.shape[1]], 0, [1], [1, 2], [16, 100])
    assert [h for h, *_ in held] == [1] and held[0][2] == pytest.approx(samples(want), abs=samples(0.5))


def test_a_view_its_main_explains_is_left_out(sounds):
    """A kick template and a view of it that discover() found in the snare
    band from more hits (same_sound_as: the kick), its window 3 ms later
    on the sound. Taken by hits, the view came first, the kick was left
    out as a composite of it and every kick was the view's, 3 ms late. The
    main comes first now and explains the view (relations() looks within
    +-10 ms), so the view is left out: the kicks are the main's, timed by
    its window (here the oracle's, on the attack; melodic-1's starts on
    the body), and the view adds none."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    kick, late = templates[1], samples(3)
    view = Template("view", "kick", "snare", np.r_[kick.audio[late:], np.zeros(late, np.float32)], kick.hits + 3,
                    {}, None, same_sound_as="kick", found="residual")
    events = explain(x, templates + [view])
    kicks = [t for n, t, _ in hits if n == "kick"]
    assert found(events, "kick", kicks, tol=0.001).all() and extra(events, "kick", kicks) == 0
    assert events["view"] == []


@pytest.mark.xfail(strict=True, reason="B9: a view and its main merge by hits, not by where the sound starts")
@pytest.mark.parametrize("late_ms", [12, 20])
def test_a_view_later_on_the_sound_leaves_the_kicks_on_their_attack(sounds, late_ms):
    """A known limitation (B9), pinned. A view 12-20 ms later on the kick
    than its main is not explained by it (relations() looks within +-10
    ms) and stays; the two keep taking each other's hits, and duplicates()
    merges the main into the view, averaged from more hits: every kick
    comes out late (0 of 24 within 1 ms). Merging into the template
    relations() takes first fixes this beat but not the suite: on
    htdemucs_ft producer-2's snare F1@20 0.587 -> 0.569 (plan 0.401 ->
    0.262), kick808-2's snare median error 0.31 -> 0.70 ms, and PIRAMIT's
    kick pad 93 -> 78 notes (no truth). The merge should keep the one whose
    window starts where their shared hits start."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    kick, late = templates[1], samples(late_ms)
    view = Template("view", "kick", "snare", np.r_[kick.audio[late:], np.zeros(late, np.float32)], kick.hits + 3,
                    {}, None, same_sound_as="kick", found="residual")
    events = explain(x, templates + [view])
    kicks = [t for n, t, _ in hits if n == "kick"]
    assert found(events, "kick", kicks, tol=0.001).all() and events["view"] == []


def test_a_merge_keeps_the_cleaner_template(sounds):
    """Two snare-band templates of the snare that keep taking each other's
    hits: its own, and one that also holds a sound the snare never plays,
    averaged from more hits. Merged into the one from more hits, every
    snare came out under NCC 0.8 (make_plan gives no pad under 0.5 clean).
    Read in one band, the one with more clean hits keeps them."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    snare = templates[2].audio
    other = fit(np.r_[np.zeros(PRE, np.float32), burst(11, 150, 300, 8000, 40)], len(snare))
    held = Template("held", "snare", "snare", (snare + 0.9 * np.max(np.abs(snare)) * other).astype(np.float32),
                    templates[2].hits + 8, {}, None)
    from refdissect.detect import explain as detect_explain
    got = detect_explain(x, templates + [held])
    snares = [t for n, t, _ in hits if n == "snare"]
    assert got.info["merged"] == {"held": {"into": "snare", "offset_ms": 0.0}}
    assert found(got.events, "snare", snares).all() and all(e["ncc"] >= 0.8 for e in got.events["snare"])


def test_clean_hits_choose_a_merge_only_within_one_band():
    """One sound found twice both ways (14 shared moments of 18 and 16
    hits) in two bands: a snare-band template with 7 clean hits and a
    low-band one with 16 (layers-2's snare). The low band reads high
    whatever the template: the snare-band one read 16 of 18 there, and the
    low-band one 1 of 16 in the snare band. Across bands the merge goes by
    first (the template averaged from more hits); in one band, into the
    one with more clean hits."""
    col = np.r_[np.zeros(18, int), np.ones(16, int)]
    p = np.r_[np.arange(18) * 4000, np.arange(16) * 4000 + 10]
    cands = layers.Candidates(col, p, np.ones(34), ["matched"] * 34)
    pairs = [(18 + k, k) if k % 2 else (k, 18 + k) for k in range(14)]  # (lost, kept), each keeps half
    clean, kind = np.array([7, 16]), [False, False]
    assert layers.duplicates(cands, pairs, [0, 1], kind, clean, ["snare", "low"]) == {1: (0, -10)}
    assert layers.duplicates(cands, pairs, [0, 1], kind, clean, ["snare", "snare"]) == {0: (1, 10)}


def test_a_left_out_view_gives_its_hits_to_its_main_not_to_a_hat_before_it(sounds):
    """A view of the kick (same_sound_as) holding a little of the hat too:
    the kick's template explains it and leaves it out. The hat, averaged
    from more hits (48) than the kick's template (24), is taken before both
    and explains the view at 0.07 of its level. A left-out template's hits
    went to every template that explained it: on melodic-1 with its hat
    averaged from 16 hits to the kick's 15, the hat got the kick view's
    hits at 0.08 and lost 32 of its 92 hits. Only one at HAND of its level
    or more gets them: here the kick."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    hat, kick = templates[0], templates[1]
    audio = kick.audio + 0.05 * np.max(np.abs(kick.audio)) / np.max(np.abs(hat.audio)) * hat.audio
    view = Template("view", "kick", "snare", audio.astype(np.float32), kick.hits + 3, {}, None,
                    same_sound_as="kick", found="residual")
    from refdissect.detect import explain as detect_explain
    got = detect_explain(x, templates + [view])
    kicks, hats = ([t for n, t, _ in hits if n == name] for name in ("kick", "hat"))
    assert hat.hits > kick.hits + 3 and list(got.info["composite"]["view"]["by"]) == ["kick"]
    assert found(got.events, "kick", kicks, tol=0.001).all() and extra(got.events, "kick", kicks) == 0
    assert found(got.events, "hat", hats).all() and extra(got.events, "hat", hats) == 0


def test_a_silent_template_is_left_out_with_its_hits_to_no_one():
    """A template whose weighted bands are all zero (a window of digital
    silence) is explained whole through explained()'s energy floor, with no
    coefficient over 0: it is left out, and its hits go to no template. The
    HAND rule took the largest of none and raised (ValueError)."""
    a = np.random.default_rng(1).standard_normal((4, 400)).astype(np.float32)
    zero = np.zeros_like(a)
    rel = layers.relations([a, zero], [[a, zero], [a, zero]], [400, 400], [20, 10], {0, 1}, [0, 0])
    assert rel["composite"] == {1: {"share": 1.0, "by": {}}} and rel["kept"] == [0]


def test_a_merge_survivor_that_is_an_echo_is_never_probed(sounds):
    """The kick and the snare hold the hat. When the hat's hits went into
    another column (here a hat template whose window starts 12 ms on), the
    holders' probes went to that column, even where relations() had found
    it an echo, whose matched filter repeats an earlier template's and is
    never probed for: on PIRAMIT 7 kick notes came in that way. Probes
    follow the column the hits went to, unless it is an echo."""
    hits = backbeat()
    x = render(sounds, hits, 0.05 + 12 * 16 * STEP + 1)
    templates = [oracle(x, hits, "hat", "hat"), oracle(x, hits, "kick", "low"), oracle(x, hits, "snare", "snare")]
    late = samples(12)
    templates.append(Template("later", "hat", "hat", np.r_[templates[0].audio[late:], np.zeros(late, np.float32)],
                              templates[0].hits - 10, {}, None))
    xp = np.r_[np.zeros(layers.LEAD, np.float32), x, np.zeros(max(len(tp.audio) for tp in templates), np.float32)]
    sc, _ = layers._scene(xp, templates, DETECT_MS, THRESHOLD, True)
    cands, weak = layers._matched(sc, xp, THRESHOLD, layers.CLOSEST_MS)
    cands, _ = layers._one_each(sc, cands)
    cands, gains = layers._significant(sc, cands, layers.MATCHED_GAIN)
    assert {h for held in sc.rel["holds"].values() for h, *_ in held} == {0} and 3 not in sc.rel["echo"]
    merged = {sc.col_of[0]: (sc.col_of[3], late)}  # the hat's hits into the later template
    got, _ = layers._probes(sc, cands, gains, merged, weak, xp, layers.MATCHED_GAIN, True)
    assert np.sum(got.col == sc.col_of[3]) > 0
    sc.rel["echo"][3] = 0.9
    got, _ = layers._probes(sc, cands, gains, merged, weak, xp, layers.MATCHED_GAIN, True)
    assert not np.any(got.col == sc.col_of[3])
