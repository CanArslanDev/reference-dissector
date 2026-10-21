"""Where each drum sound plays, and the 808's notes.

Drums, in two passes:

1. discover(): in each band (low, snare, hat), the onsets, at where they
   start, that share a place in the beat (a 16th, swung or not, a triplet,
   a 32nd) are grouped by waveform; groups with enough hits become
   candidates. A sound found in several bands, or a template of two sounds
   that play together, is kept once (one_per_sound); then the onsets no
   template explains are grouped wherever they sit (rare sounds, fills).
   Templates are aligned on where the sound starts in its role's band
   (hits.RISE), so a hit's time is its attack: t_attack.
2. detect(): each template slides over the whole stem (matched filter); its
   peaks above a threshold are its hits, with a level. Where two templates of
   one band claim the same moment, the better match keeps it. count_once()
   then keeps each hit once: not a view's hits again, nor a template's own
   ring.

The templates come from the song itself, so the second pass finds a sound
even where the first could not see it (under other hits, off the grid).
On PIRAMIT the snare templates separate cleanly: 28 hits at NCC >= 0.8 and
nothing between 0.3 and 0.8.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.signal import find_peaks

from .audio import PRE, SR, band, samples
from .grid import Grid
from .hits import RISE, Group, aligned_template, group, merge, onsets, rise, starts, strength_peaks
from .xcorr import ncc, sliding_ncc

CLUSTER_MS = {"kick": 80, "snare": 60, "hat": 40, "low": 80}
# How much of a template is compared. Patterns repeat, so whatever always
# follows a hit (the kick two sixteenths after a snare) is in its average
# too; long windows let that later, louder sound decide the match. Detection
# looks at the attack only, library matching at a little more.
DETECT_MS = {"kick": 80, "snare": 60, "hat": 40, "low": 80}
LENGTH_MS = {"kick": 200, "snare": 150, "hat": 80, "low": 200}
MAX_MS = 350  # templates are kept this long and cut where used
THRESHOLD = {"kick": 0.6, "snare": 0.6, "hat": 0.65, "low": 0.6}
# A low tone correlates with itself one period later; under 200 Hz one hit
# would otherwise be found every 35 ms while it rings
MIN_GAP_MS = {"kick": 35, "snare": 35, "hat": 35, "low": 100}


SAME_MS = 15  # hits of two templates this close are one moment
# A view finds its main's hits again at one lag, 36-39 ms before or after
# them on 4 suite beats (full-1, glides-1, layers-1, rolls-1): its hits
# this close to the main's are those hits
VIEW_MS = 50
RING_MS = 250  # how long after a hit its template's own ring is looked for (ringing())
ROLE_BAND = {"kick": "low", "snare": "snare", "hat": "hat"}  # where a guessed role's sound is compared
RESIDUAL_MS = 10  # an onset this close to a template's hit is explained by it
# How far before an onset-strength peak its start is looked for: a kick's
# body peaks in the low band 30-32 ms after its click when it comes late
LOOK_BACK_MS = {"low": 45, "kick": 45, "snare": 30, "hat": 30}


@dataclass
class Template:
    """A sound that repeats: name (sound_1 has the most hits), role (a guess
    from the energy its hits add; matching against a library decides), band
    (where it was found, and where it is detected), its hits lined up
    (MAX_MS long; averaged, the median in the hat band), the grid slots of
    those hits, and attack_s: seconds from the start of `audio` to where the
    sound starts (hits.RISE of its role), PRE unless the window is cut.
    same_sound_as names the template that holds the same sound in another
    band, found says which pass of discover() found it (grid, residual)."""
    name: str
    role: str
    band: str
    audio: np.ndarray
    hits: int
    slots: dict[int, int] = field(default_factory=dict)
    group: Group | None = None
    attack_s: float = PRE / SR
    same_sound_as: str | None = None
    found: str = "grid"


@dataclass
class Candidate:
    """A group on its way to becoming a template: its band, its detection
    template (DETECT_MS, the sound's start `attack` samples in) and what
    that finds over the whole stem: start times (s), gains and NCCs."""
    band: str
    group: Group
    audio: np.ndarray
    attack: int
    times: np.ndarray
    gains: np.ndarray
    ncc: np.ndarray
    short: np.ndarray
    role: str = "snare"
    found: str = "grid"
    view_of: Candidate | None = None


def new_energy_role(x: np.ndarray, g: Group, gap_ms: float = 10, span_ms: float = 60) -> tuple[str, dict]:
    """A first guess at kick / snare / hat from the energy the hits add: the
    spectrum of the span after each onset minus the span before it (an 808
    ringing under a snare is there before the snare too, so it cancels out),
    as the median over the hits of the share each adds below 150 Hz and
    above 5 kHz. Summed over the hits instead, the kicks under a third of
    the hats decided: a hat template of a v1 beat read as a kick (0.87 of
    the energy below 150 Hz, 0.00 for the median hit). Only a guess: bright
    snares and dull hats overlap; the library decides."""
    span, gap = samples(span_ms), samples(gap_ms)
    n = 1 << (span - 1).bit_length()
    f = np.fft.rfftfreq(n, 1 / SR)
    win = np.hanning(span)
    low, high = [], []
    for s, lag, _sign, _gain, _ in g.members:
        a = s - lag
        if a - gap - span < 0 or a + span > len(x):
            continue
        after = np.abs(np.fft.rfft(x[a:a + span] * win, n)) ** 2
        before = np.abs(np.fft.rfft(x[a - gap - span:a - gap] * win, n)) ** 2
        added = np.maximum(after - before, 0)
        total = added.sum() or 1.0
        low.append(added[f < 150].sum() / total)
        high.append(added[f > 5000].sum() / total)
    shares = {"low": float(np.median(low)) if low else 0.0, "high": float(np.median(high)) if high else 0.0}
    if shares["low"] > 0.35:
        role = "kick"
    elif shares["high"] > 0.35:
        role = "hat"
    else:
        role = "snare"
    return role, {k: round(v, 3) for k, v in shares.items()}


def min_hits_for(n_samples: int, grid: Grid) -> int:
    """How many hits a sound needs at one place of the beat to be seeded: 3,
    or one per 4 bars in a longer song."""
    bars = max(1.0, (n_samples / SR - max(grid.phase, 0.0)) / grid.bar)
    return max(3, int(np.ceil(bars / 4)))


def seeds(x: np.ndarray, band_name: str) -> np.ndarray:
    """Where the band's onsets start (s): hits.starts of its onset-strength
    peaks, in RISE[band_name], up to LOOK_BACK_MS before them; a peak without
    a clean rise is left out, and two peaks of one hit (a click and the body
    after it) count once."""
    t, _ = strength_peaks(x, band_name)
    s = starts(x, np.round(t * SR).astype(int), RISE[band_name], before_ms=LOOK_BACK_MS[band_name], after_ms=20)
    s = np.unique(s[s >= 0])
    if len(s) > 1:
        s = s[np.r_[True, np.diff(s) > samples(5)]]
    return s / SR


def places(t: np.ndarray, grid: Grid, width_ms: float = 12.0, min_hits: int = 3) -> np.ndarray:
    """Which place in the beat each time falls on: clusters of the times
    modulo a quarter note, each up to width_ms either side of its median,
    taken densest first; -1 for times in clusters of fewer than min_hits.
    Swung 16ths, triplets and 32nds are places of their own wherever the
    song puts them, and the meter does not matter."""
    beat, w = grid.beat, width_ms / 1000
    pos = (np.asarray(t, dtype=float) - grid.phase) % beat

    def dist(a, b):
        return np.abs((a - b + beat / 2) % beat - beat / 2)
    near = dist(pos[:, None], pos[None, :]) <= w
    out = np.full(len(pos), -1)
    seen = np.zeros(len(pos), bool)
    k = 0
    for i in np.argsort(-near.sum(axis=1), kind="stable"):
        if out[i] >= 0 or seen[i]:
            continue
        seen[i] = True
        free = near[i] & (out < 0)
        centre = pos[i] + float(np.median((pos[free] - pos[i] + beat / 2) % beat - beat / 2))
        mine = (dist(pos, centre) <= w) & (out < 0)
        if mine.sum() >= min_hits:
            out[mine] = k
            k += 1
    return out


def found_by(x: np.ndarray, audio: np.ndarray, band_name: str, attack: int, min_gain: float = 0.25,
             tail_ms: float = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a detection template finds over the whole stem, as detect()
    finds it: (start times in s, gains, NCC), hits far below the usual gain
    left out, and those within tail_ms after a hit of twice their level:
    the same hit ringing on, or found again where the template's own later
    part lines up with it."""
    score, gain = sliding_ncc(x, audio, band_name)
    p, props = find_peaks(score, height=THRESHOLD[band_name], distance=samples(MIN_GAP_MS[band_name]))
    g, h = gain[p], props["peak_heights"]
    keep = g >= min_gain * np.median(g) if len(g) else np.zeros(0, bool)
    t, g, h = (p[keep] + attack) / SR, g[keep], h[keep]
    tail = np.zeros(len(t), bool)
    last = 0
    for i in range(1, len(t)):
        tail[i] = t[i] - t[last] < tail_ms / 1000 and g[i] < 0.5 * g[last]
        if not tail[i]:
            last = i
    return t[~tail], g[~tail], h[~tail]


def candidate(x: np.ndarray, g: Group, band_name: str, found: str = "grid", median: bool = False) -> Candidate:
    """A group's detection template, its start in the band of the role its
    hits guess (new_energy_role, hits.RISE), and what it finds."""
    role = new_energy_role(x, g)[0]
    short, _, at = aligned_template(x, g, 100, LOOK_BACK_MS[role], bands=RISE[role], median=median)
    audio = short[:samples(DETECT_MS[band_name])]
    return Candidate(band_name, g, audio, at, *found_by(x, audio, band_name, at), short, role, found)


def refined(x: np.ndarray, c: Candidate, median: bool | None = None, more: float = 2.0) -> Candidate:
    """A hat-band candidate rebuilt from the hits it finds instead of the
    onsets that seeded it, when it finds `more` times as many: each brought
    to the template's level (1 / its gain), their median. A template is then
    what it detects: seeded on the kick's places, a group in the hat band
    held the kick under its hats, found every hat, and guessed itself a
    kick; rebuilt, it is the hat. In the other bands rebuilding made things
    worse: a few seeds that found the kick's body and more grew into a
    template that held the kick (suite kick F1 0.82 -> 0.75)."""
    if c.band != "hat" or len(c.times) < max(3, more * c.group.count):
        return c
    hits = np.round(c.times * SR).astype(int) - c.attack + PRE
    g = Group(members=[(int(s), 0, 1.0, float(1 / max(gain, 1e-3)), float(h))
                       for s, gain, h in zip(hits, c.gains, c.ncc)])
    out = candidate(x, g, c.band, c.found, c.band == "hat" if median is None else median)
    return out if len(out.times) >= 3 else c


def pairs(a: np.ndarray, b: np.ndarray, ms: float = SAME_MS) -> tuple[np.ndarray, np.ndarray]:
    """Indices (i, j) of the times a[i] whose nearest time in b, b[j], is
    within ms."""
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int)
    j = np.clip(np.searchsorted(b, a), 1, len(b) - 1) if len(b) > 1 else np.zeros(len(a), int)
    j = np.where(np.abs(b[j - 1] - a) < np.abs(b[j] - a), j - 1, j) if len(b) > 1 else j
    ok = np.abs(b[j] - a) < ms / 1000
    return np.flatnonzero(ok), j[ok]


def likeness(a: Candidate, b: Candidate) -> dict:
    """How alike two candidates are: the share of each one's hits the other
    finds within SAME_MS (cover_a, cover_b), the correlation of their gains
    (in dB) on the shared hits (r; None with fewer than 6, or gains that
    hardly vary), and in the band of a's role (ROLE_BAND) their templates'
    NCC (alike) and b's level against a's (level_db); the same in the band a
    was found in (alike_found, level_db_found)."""
    i, j = pairs(a.times, b.times)
    out = {"shared": len(i), "cover_a": len(i) / max(1, len(a.times)), "cover_b": len(set(j)) / max(1, len(b.times)),
           "r": None, "same_role": a.role == b.role}
    if len(i) >= 6:
        ga, gb = 20 * np.log10(np.maximum(a.gains[i], 1e-6)), 20 * np.log10(np.maximum(b.gains[j], 1e-6))
        if min(np.std(ga), np.std(gb)) >= 0.5:
            out["r"] = float(np.corrcoef(ga, gb)[0, 1])
    for key, b_name in (("", ROLE_BAND[a.role]), ("_found", a.band)):
        sa, sb = band(a.short, b_name), band(b.short, b_name)
        out["alike" + key] = abs(ncc(sa, sb, samples(15))[0])
        out["level_db" + key] = float(20 * np.log10((np.linalg.norm(sb) + 1e-12) / (np.linalg.norm(sa) + 1e-12)))
    return out


def explains(L: dict) -> bool:
    """Whether likeness() says b holds a's sound on the hits they share: the
    gains move together (r >= 0.8, or 0.6 when the two guess the same
    role); or in a's role's band b's template has a's shape (NCC >= 0.6) at
    a's level or louder (within 6 dB), and either the two find each other's
    hits (cover >= 0.8 both ways), or they guess the same role and b holds
    a's level in the band a was found in too. A kick under some of the hats
    is not held by the hat: the hat templates of v1 beats held it 5.9-10.9
    dB down in the low band (2.9 dB on a slow beat where htdemucs had left
    little of the kick's body in the drum stem), each guessed its own role,
    and the kick was on a third of the hats or less; their gains on the 15
    hits they shared on a 95 BPM beat still went together at r = 0.66. A
    crash on every kick it plays with guesses kick too, and the kick holds
    its low band, but not the crash in the hat band it was found in. Views
    of one hit found in the low band hold its top 14 dB down (low-band
    onsets line the hits up too loosely for it)."""
    if L["r"] is not None and L["r"] >= (0.6 if L["same_role"] else 0.8):
        return True
    if L["alike"] < 0.6 or L["level_db"] < -6.0:
        return False
    if L["cover_a"] >= 0.8 and L["cover_b"] >= 0.8:
        return True
    return L["same_role"] and L["level_db_found"] >= -6.0


def grid_candidates(x: np.ndarray, grid: Grid, bands, min_hits: int, width_ms: float,
                    seeded: dict[str, np.ndarray] | None = None) -> list[Candidate]:
    """Groups of the onsets that share a place in the beat (places), per
    band, merged across places when their averages match. seeded: each
    band's seeds(), when known already."""
    out: list[Candidate] = []
    for band_name in bands:
        xb = band(x, band_name)
        t = seeded[band_name] if seeded else seeds(x, band_name)
        where = places(t, grid, width_ms, min_hits)
        groups: list[Group] = []
        for k in sorted(set(where[where >= 0].tolist())):
            starts_k = np.round(t[where == k] * SR).astype(int)
            groups += [g for g in group(xb, starts=starts_k, threshold=0.6, cluster_ms=CLUSTER_MS[band_name])
                       if g.count >= min_hits]
        out += [refined(x, candidate(x, g, band_name)) for g in merge(xb, groups, threshold=0.8, length_ms=100,
                                                                      min_count=min_hits)]
    return [c for c in out if len(c.times)]


def onset_share(c: Candidate, onsets: np.ndarray, top: np.ndarray,
                near_ms: float = RESIDUAL_MS) -> tuple[float, float, float]:
    """(share of c's hits that sit on an onset, one of `onsets` within
    near_ms; share whose hat band (`top`) rises there, within 15 ms; share
    with an onset 15-45 ms before them)."""
    if not len(c.times):
        return 0.0, 0.0, 0.0
    on = len(pairs(c.times, onsets, near_ms)[0]) / len(c.times)
    at = np.round(c.times * SR).astype(int)
    up = np.mean([rise(top, int(t), samples(15), samples(15)) is not None for t in at])
    before = np.mean(np.searchsorted(onsets, c.times - 0.015) > np.searchsorted(onsets, c.times - 0.045))
    return on, float(up), float(before)


def starts_something(c: Candidate, onsets: np.ndarray, top: np.ndarray, min_onsets: float = 0.3,
                     min_top: float = 0.05, max_before: float = 0.3) -> bool:
    """Whether a candidate's hits start a sound (onset_share): min_onsets of
    them on an onset, and either min_top with an attack that reaches the hat
    band or under max_before with an onset 15-45 ms before them. What rings
    does not: an 808 matched once a period in the snare band, or the body of
    a kick, which htdemucs leaves in the drum stem 20-40 ms after its click.
    On 8 suite beats the candidates that played no true hit (10 % or fewer
    within 20 ms) and rang had 4-28 % of their hits on an onset, or 0-3 %
    rising in the hat band; those whose hits were true (60 % or more)
    27-100 % and 27-100 %. A clean candidate (NCC >= 0.8 on 80 % of its
    hits) with nothing starting just before it is let 15 ms from the
    onsets: the template starts where its role's band climbs from the floor,
    an onset where its band reaches 30 % of its peak, and on leadin-1 the
    snare's template started 13 ms before its hat-band onsets (0.19 within
    10 ms): no snare was found. A kick without a click has no top either,
    but nothing starts just before it, as a click does before a body."""
    on, up, before = onset_share(c, onsets, top)
    if on < min_onsets and before < max_before and np.mean(c.ncc >= 0.8) >= 0.8:
        on = onset_share(c, onsets, top, 15)[0]
    return on >= min_onsets and (up >= min_top or before < max_before)


def holder(a: Candidate, kept: list[Candidate]) -> tuple[Candidate | None, float, float, int]:
    """(the kept candidate that explains the most of a's hits (explains()),
    the share of a's hits it holds, the share any kept one finds, and how
    many of a's hits are new: none of the kept ones finds them, nor the
    holder within VIEW_MS, where a view finds its main's hits again)."""
    covered = np.zeros(len(a.times), bool)
    best, held = None, 0.0
    for k in kept:
        i, _ = pairs(a.times, k.times)
        if not len(i):
            continue
        covered[i] = True
        if len(i) / len(a.times) > held and len(i) >= 3 and explains(likeness(a, k)):
            best, held = k, len(i) / len(a.times)
    new = ~covered
    if best is not None:
        new[pairs(a.times, best.times, VIEW_MS)[0]] = False
    return best, held, float(covered.mean()) if len(a.times) else 0.0, int(new.sum())


def one_per_sound(cands: list[Candidate], kept: list[Candidate] | None = None, alone: float = 0.6,
                  share: float = 0.8, view: float = 0.3, min_new: int = 3) -> list[Candidate]:
    """The candidates that are sounds of their own, those with more clean
    hits (NCC >= 0.8), at a larger share of their hits, first (clean hits
    squared over hits: 17 clean of 32 come after 16 of 16, where the 16
    other hits were a low band's guesses). One the kept ones already hold is
    left out: a kept one that explains it (holder) holds `alone` of its
    hits, or kept ones find `share` of them and one of those explains it (a
    template of a kick and the hat on it, found in the snare band, has the
    kick's hits and some of the hat's). One that its holder explains on
    `view` of its hits but that finds min_new hits nobody else does is kept
    as a view of the same sound (view_of: the kick in the hat band, where
    the low band missed it under an 808); with fewer new hits it is left
    out too. A hit it finds again within VIEW_MS of its holder's is not
    new. Such candidates used to stay as sounds of their own and play their
    holder's hits a second time: dropped, kick F1@20 over 22 suite scenarios
    went from 0.839 to 0.869 (analysis, guessed roles)."""
    kept = list(kept or [])
    for c in sorted(cands, key=lambda c: (-np.sum(c.ncc >= 0.8) ** 2 / max(1, len(c.ncc)), -float(np.median(c.ncc)))):
        k, held, covered, new = holder(c, kept)
        if held >= alone or (k is not None and covered >= share):
            continue
        if k is not None and held >= view:
            if new < min_new:
                continue
            c.view_of = k.view_of or k
        kept.append(c)
    return kept


def locked(a: Candidate, b: Candidate, share: float = 0.8, spread_ms: float = 2.0) -> bool:
    """Whether a's hits come at one lag, 15-60 ms, after b's: share of a's
    hits within spread_ms of it, and share of b's hits followed there."""
    if len(a.times) < 3 or len(b.times) < 3:
        return False
    j = np.searchsorted(b.times, a.times - SAME_MS / 1000) - 1
    lag = np.where(j >= 0, a.times - b.times[np.maximum(j, 0)], np.inf)
    near = (lag >= SAME_MS / 1000) & (lag <= 0.06)
    if near.mean() < share:
        return False
    at = float(np.median(lag[near]))
    return (np.mean(np.abs(lag - at) <= spread_ms / 1000) >= share
            and len(pairs(b.times + at, a.times, spread_ms)[0]) >= share * len(b.times))


def loudness_db(x: np.ndarray, c: Candidate, ms: float = 30) -> float:
    """The median level (dB, full band) of the first ms of c's hits."""
    n = samples(ms)
    rms = [np.sqrt(np.mean(np.square(x[i:i + n], dtype=np.float64))) for i in np.round(c.times * SR).astype(int)]
    return float(20 * np.log10(np.median(rms) + 1e-12)) if rms else -np.inf


def unlocked(x: np.ndarray, kept: list[Candidate]) -> list[Candidate]:
    """kept without the quieter of two candidates of different roles that
    always come one fixed lag apart (locked): what always comes 40 ms
    before a snare is part of it, or of the sound before it. On two suite
    beats a candidate played no true hit 41.6 and 42.7 ms before every
    snare, 1.6 and 15.4 dB under it; on 28 others no candidates were
    locked. A kick's body after its click guesses kick too, and stays. A
    view of one that goes is the same sound, and goes with it."""
    out = list(kept)
    for a in kept:
        for b in kept:
            if a is not b and a in out and b in out and a.role != b.role and locked(a, b):
                out.remove(a if loudness_db(x, a) < loudness_db(x, b) else b)
    return [c for c in out if c.view_of is None or c.view_of in out]


def residual_candidates(x: np.ndarray, kept: list[Candidate], bands, min_hits: int = 3,
                        seeded: dict[str, np.ndarray] | None = None) -> list[Candidate]:
    """Groups of the onsets in each band that no kept template of that band
    explains (none of its hits within RESIDUAL_MS), at any place: rare
    sounds, fills, sounds off the grid. A crash on a kick is explained by
    the kick in the low band, not in the hat band."""
    out: list[Candidate] = []
    for band_name in bands:
        mine = [c.times for c in kept if c.band == band_name]
        found = np.sort(np.concatenate(mine)) if mine else np.zeros(0)
        t = seeded[band_name] if seeded else seeds(x, band_name)
        if len(found):
            i, _ = pairs(t, found, RESIDUAL_MS)
            t = np.delete(t, i)
        if len(t) < min_hits:
            continue
        xb = band(x, band_name)
        groups = [g for g in group(xb, starts=np.round(t * SR).astype(int), threshold=0.6,
                                   cluster_ms=CLUSTER_MS[band_name]) if g.count >= min_hits]
        out += [candidate(x, g, band_name, "residual") for g in merge(xb, groups, threshold=0.8, length_ms=100,
                                                                      min_count=min_hits)]
    return [c for c in out if len(c.times) >= min_hits]


def discover(drums: np.ndarray, grid: Grid, bands=("snare", "hat", "low"), min_hits: int | None = None,
             width_ms: float = 12.0, residual: bool = True, min_residual: int = 3) -> list[Template]:
    """Templates of the sounds that repeat. grid: on the attacks' time base
    (grid.estimate's; only its tempo and phase are used).

    In each band, the onsets, at where they start (seeds), that share a
    place in the beat (places: 16ths swung or not, triplets, 32nds, any
    meter) are grouped by waveform; groups of min_hits or more (min_hits_for:
    3, or one per 4 bars) are merged when their averages match. Candidates
    whose hits do not start a sound (starts_something: an 808 ringing, the
    body of a kick after its click) are left out, and a sound found in more
    than one band, or a template of two sounds that play together, is kept
    once (one_per_sound; a partial view that finds hits the others miss is
    kept as same_sound_as, its shared hits counted once: count_once). Then
    the onsets none of them explains are grouped wherever they sit
    (residual_candidates: min_residual hits) and go through the same tests.
    Each template's role is guessed from the energy its hits add
    (new_energy_role); its audio is the hits' mean (the median in the hat
    band), the sound's start in its role's band PRE into it (attack_s).

    Against the code before (16th slots +-12 ms, 8 hits a slot, a cross-band
    duplicate test, the full-band attack) on 32 suite beats of 14 scenarios
    (htdemucs stems, the library's roles), strict F1@20 per role, mean over
    scenarios, analysis: kick 0.26 -> 0.87, snare 0.85 -> 0.96, hat 0.45 ->
    0.75, open hat 0.23 -> 0.97, crash 0 -> 0.47; plan: kick 0.17 -> 0.80,
    snare 0.58 -> 0.79. Kicks were found 22.6 ms late (median), now 0.9."""
    min_hits = min_hits or min_hits_for(len(drums), grid)
    seeded = {b: seeds(drums, b) for b in bands}
    onsets = np.sort(np.concatenate(list(seeded.values())))
    top = band(drums, "hat")
    kept = one_per_sound([c for c in grid_candidates(drums, grid, bands, min_hits, width_ms, seeded)
                          if starts_something(c, onsets, top)])
    if residual:
        kept = one_per_sound([c for c in residual_candidates(drums, kept, bands, min_residual, seeded)
                              if starts_something(c, onsets, top)], kept)
    kept = unlocked(drums, kept)
    kept.sort(key=lambda c: -len(c.times))
    names = {id(c): f"sound_{k + 1}" for k, c in enumerate(kept)}
    out: list[Template] = []
    for c in kept:
        main = c.view_of or c
        audio, used, at = aligned_template(drums, c.group, MAX_MS, LOOK_BACK_MS[main.role], bands=RISE[main.role],
                                           median=c.band == "hat")
        slot_counts = Counter(grid.step_in_bar(np.array(c.group.starts()) / SR).tolist())
        out.append(Template(names[id(c)], main.role, c.band, audio, used, dict(sorted(slot_counts.items())), c.group,
                            round(at / SR, 5), names[id(main)] if c.view_of else None, c.found))
    return out


def ring(tp: Template) -> np.ndarray:
    """How loud the template's own ring can be at each lag (samples) after
    its start, as a share of its start: its level in its band over its
    detection length there, never more than at a lag before (a ring only
    decays; a hat template's level rises again 220 ms on, where the next 8th
    hat always plays, and that is no ring)."""
    y = band(tp.audio, tp.band).astype(np.float64)
    e = np.convolve(y ** 2, np.ones(samples(DETECT_MS[tp.band])), mode="valid")
    return np.minimum.accumulate(np.sqrt(e / (e[0] or 1e-12)))


def ringing(y: np.ndarray, times: np.ndarray, gains: np.ndarray, own: np.ndarray, ms: float | None = None,
            margin: float = 1.25) -> np.ndarray:
    """Which of a template's hits (start times in s, sorted, and gains) are a
    louder hit of its own ringing on: within ms after a kept hit, no louder
    than that hit's ring there (its gain times own, ring(), at the lag; the
    matched filter finds a ring at NCC times that), and nothing starts in
    the template's band y (no hits.rise to half the window's peak within 15
    ms). NCC ignores level, so a tonal snare's template matched its own ring
    37-158 ms after each snare at 0.10-0.59 of its gain: 101 extra hits for
    32 snares, and the scorer gave the template to no sound. A hat's ring is
    gone within the 110 ms to the next 16th, so a soft hat after an
    accented one is never taken for it."""
    out = np.zeros(len(times), bool)
    last = None
    ms = RING_MS if ms is None else ms
    for k, t in enumerate(times):
        lag = int(round((t - times[last]) * SR)) if last is not None else len(own)
        if (lag < min(len(own), samples(ms)) and gains[k] <= margin * gains[last] * own[lag]
                and rise(y, int(round(t * SR)), samples(15), samples(15), level=0.5) is None):
            out[k] = True
            continue
        last = k
    return out


def count_once(events: dict[str, list[dict]], templates: list[Template],
               drums: np.ndarray | None = None) -> dict[str, list[dict]]:
    """Each hit once. A template that holds the same sound as another one
    (same_sound_as) loses its events within VIEW_MS of that one's: a hit
    found in two bands is one hit, and so is a hit a view finds again at a
    lag. Within 15 ms instead, full-1's snare-band view of the kick kept 43
    hits 37 ms after a kick, none of them on a true one. Given the stem
    (drums), every template also loses its own ringing (ringing())."""
    def at(name):
        return np.array([e.get("t_attack", e["t"]) for e in out[name]], dtype=float)

    out = {k: sorted(v, key=lambda e: e.get("t_attack", e["t"])) for k, v in events.items()}
    if drums is not None:
        ys = {b: band(drums, b) for b in {tp.band for tp in templates}}
        for tp in templates:
            if out.get(tp.name):
                rung = ringing(ys[tp.band], at(tp.name), np.array([e["gain"] for e in out[tp.name]]), ring(tp))
                out[tp.name] = [e for e, r in zip(out[tp.name], rung) if not r]
    for tp in templates:
        if tp.same_sound_as and tp.same_sound_as in out:
            shared = set(pairs(at(tp.name), np.sort(at(tp.same_sound_as)), VIEW_MS)[0].tolist())
            out[tp.name] = [e for k, e in enumerate(out[tp.name]) if k not in shared]
    return out


def detect(drums: np.ndarray, templates: list[Template], thresholds: dict | None = None,
           same_hit_ms: float = 15, min_gain: float = 0.25) -> dict[str, list[dict]]:
    """Every hit of every template: {name: [{t, t_attack, ncc, gain}]} (t in
    seconds, PRE after the template's window starts; t_attack where the
    sound starts, attack_s in)."""
    thresholds = {**THRESHOLD, **(thresholds or {})}
    raw: dict[str, list[dict]] = {}
    for tp in templates:
        score, gain = sliding_ncc(drums, tp.audio[: samples(DETECT_MS[tp.band])], tp.band)
        peaks, props = find_peaks(score, height=thresholds[tp.band], distance=samples(MIN_GAP_MS[tp.band]))
        found = [{"t": round((p + PRE) / SR, 4), "t_attack": round(p / SR + tp.attack_s, 4), "ncc": round(float(h), 3),
                  "gain": round(float(gain[p]), 4)} for p, h in zip(peaks, props["peak_heights"])]
        # NCC ignores level: the quiet ringing tail of a low hit matches its own
        # template too. Hits far below the template's usual level are dropped.
        if found:
            usual = float(np.median([e["gain"] for e in found]))
            found = [e for e in found if e["gain"] >= min_gain * usual]
        if tp.band == "low":  # and a ringing tail behind a louder hit of the same sound
            kept: list[dict] = []
            for e in found:
                if kept and e["t"] - kept[-1]["t"] < 0.25 and e["gain"] < 0.5 * kept[-1]["gain"]:
                    continue
                kept.append(e)
            found = kept
        raw[tp.name] = found
    # within a band, one moment belongs to the template that matches it best
    out = {k: [] for k in raw}
    by_band: dict[str, list[tuple[str, dict]]] = {}
    for tp in templates:
        by_band.setdefault(tp.band, []).extend((tp.name, e) for e in raw[tp.name])
    for events in by_band.values():
        events.sort(key=lambda ne: ne[1]["t"])
        i = 0
        while i < len(events):
            j = i
            while j + 1 < len(events) and events[j + 1][1]["t"] - events[i][1]["t"] <= same_hit_ms / 1000:
                j += 1
            name, e = max(events[i:j + 1], key=lambda ne: ne[1]["ncc"])
            out[name].append(e)
            i = j + 1
    return out


def peak_score_histogram(drums: np.ndarray, tp: Template) -> list[int]:
    """How many matched-filter peaks fall in each NCC band 0.3-0.4 ... 0.9-1.0:
    a clean template shows a gap between its hits and everything else."""
    score, _ = sliding_ncc(drums, tp.audio[: samples(DETECT_MS[tp.band])], tp.band)
    _, props = find_peaks(score, height=0.3, distance=samples(MIN_GAP_MS[tp.band]))
    return np.histogram(props["peak_heights"], bins=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[0].tolist()


def pitch_offset(midi: list[float] | np.ndarray) -> tuple[float, float]:
    """Common offset from equal temperament (semitones, -0.5..0.5) of pitches,
    averaged on the circle, and how much they agree on it (0..1)."""
    m = np.asarray(midi, dtype=float)
    if not len(m):
        return 0.0, 0.0
    z = np.mean(np.exp(2j * np.pi * (m - np.round(m))))
    return round(float(np.angle(z) / (2 * np.pi)), 3), round(float(abs(z)), 3)


def notes_808(bass: np.ndarray, tuning: float, fade_db: float = -35.0, own_tuning: bool = True,
              min_agreement: float = 0.5) -> list[dict]:
    """808 notes from the bass stem: onset, end (where it falls fade_db below
    its peak, or the next note), pitch rounded after removing a tuning offset,
    and the pitch where it ends (a slide when they differ).

    The offset is the 808's own when its notes agree on one (own_tuning, at
    least min_agreement): an 808 sample is often detuned on its own. On
    PIRAMIT the melodic stems sit at A440 (-1..+3 cents by spectral peaks and by
    pyin) while its 808 notes sit at +33 cents. Otherwise `tuning` (the
    track's) is used."""
    lo = band(bass, "808")
    on = onsets(lo, delta=0.05)
    w = samples(25)
    env = np.sqrt(np.convolve(lo ** 2, np.ones(w) / w, mode="same"))
    notes = []
    for k, s in enumerate(on):
        nxt = on[k + 1] if k + 1 < len(on) else len(lo)
        a, b = s + samples(60), min(s + samples(450), nxt - samples(20))
        if b - a < samples(60):
            continue
        f0, voiced, _ = librosa.pyin(lo[a:b], fmin=25, fmax=160, sr=SR, frame_length=4096, hop_length=512)
        f = f0[voiced] if np.any(voiced) else np.array([])
        if not len(f):
            continue
        midi = float(librosa.hz_to_midi(np.median(f)))
        tail = f[-3:] if len(f) >= 3 else f
        peak = float(env[s:s + samples(200)].max())
        seg = env[s:nxt]
        below = np.where(seg[samples(100):] < peak * 10 ** (fade_db / 20))[0]
        end = s + samples(100) + int(below[0]) if len(below) else nxt
        notes.append({"t": round(s / SR, 4), "end": round(min(end, nxt) / SR, 4),
                      "midi_float": round(midi, 2),
                      "slide": round(float(librosa.hz_to_midi(np.median(tail))) - midi, 1),
                      "peak": round(peak, 5)})
    offset, agreement = pitch_offset([n["midi_float"] for n in notes])
    use = offset if own_tuning and agreement >= min_agreement else tuning
    for n in notes:
        n["pitch"] = int(round(n["midi_float"] - use))
    return notes


def isolated(notes: list[dict], pitch: int | None = None, ring_s: float = 0.6) -> list[dict]:
    """Notes of one pitch (default: the most common) that ring at least ring_s
    before the next note: what the 808 sounds like on its own."""
    if not notes:
        return []
    pitch = pitch if pitch is not None else Counter(n["pitch"] for n in notes).most_common(1)[0][0]
    out = []
    for k, n in enumerate(notes):
        nxt = notes[k + 1]["t"] if k + 1 < len(notes) else float("inf")
        if n["pitch"] == pitch and nxt - n["t"] >= ring_s:
            out.append(n)
    return out
