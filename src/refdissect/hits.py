"""Hits in a stem: finding them, grouping the ones that are the same sound,
and averaging each group into a template.

A produced track plays the same sample many times. Lined up and averaged,
N hits keep the sample and push everything that plays alongside it (hats
over a snare, vocal bleed) down by about 10*log10(N) dB, as long as that
other material differs from hit to hit. What always plays with the hit (a
hat on every snare) stays in the template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy.ndimage import maximum_filter1d

from .audio import PRE, SR, attack, band, samples, shift
from .xcorr import ncc

HOP = 128
# The bands a sound's start is read in (rise), by its role (or the band its
# onsets were found in): the role's band, where what rang before (a kick or
# 808 under a snare) is filtered out. A kick starts where its kick or snare
# band rises, whichever comes first: on htdemucs drums its body comes 26 ms
# after its click (a v1 beat: 30 % of the kick band's peak at +25.9 ms, of
# the snare band's at -1.7 ms, the clean kick's at 0.0); the rest went to
# the bass stem. Not the earliest band of all: a template of the kick's
# places found in the hat band then started on the kick, and its hits on
# the hats came 14.4 ms early.
RISE = {"low": ("low", "snare"), "kick": ("kick", "snare"), "snare": ("snare",), "hat": ("hat",)}


def onsets(x: np.ndarray, delta: float = 0.07) -> np.ndarray:
    """Onset positions (samples), moved back to where each attack starts."""
    env = librosa.onset.onset_strength(y=x, sr=SR, hop_length=HOP)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, delta=delta,
                                        backtrack=True, units="frames")
    return librosa.frames_to_samples(frames, hop_length=HOP)


def strength_peaks(x: np.ndarray, role: str = "full", delta: float = 0.3,
                   wait_frames: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """(times in s, strength) of onset-strength peaks in the role's band:
    where attacks happen, without backtracking (for grids and matching)."""
    env = librosa.onset.onset_strength(y=band(x, role), sr=SR, hop_length=HOP)
    p = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10, post_avg=10, delta=delta,
                               wait=wait_frames)
    return p * HOP / SR, env[p]


@dataclass
class Group:
    """Hits judged to be one sound. members: (start, lag, sign, gain, score);
    the hit's audio lines up with the group when read from start - PRE - lag."""
    members: list[tuple[int, int, float, float, float]] = field(default_factory=list)
    total: np.ndarray | None = None
    n: int = 0

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def consistency(self) -> float:
        """Median similarity of the members to the group when they joined."""
        scores = [m[4] for m in self.members[1:]]
        return float(np.median(scores)) if scores else 1.0

    def starts(self) -> list[int]:
        return sorted(m[0] for m in self.members)


def group(x: np.ndarray, starts=None, cluster_ms: float = 80, threshold: float = 0.6,
          max_lag_ms: float = 4, min_db: float = -40.0) -> list[Group]:
    """Hits that are the same sound, grouped greedily from the loudest: each
    hit joins the group whose running average it matches best (NCC of the
    first cluster_ms) if that reaches threshold, else starts a new group.
    threshold=-1 puts every hit in one group (when the times are known)."""
    starts = onsets(x) if starts is None else np.asarray(starts, dtype=int)
    length, max_lag = samples(cluster_ms), samples(max_lag_ms)
    loud = np.array([np.sqrt(np.mean(x[max(0, s - PRE):s - PRE + length] ** 2)) for s in starts])
    floor = (loud.max() if len(loud) else 0.0) * 10 ** (min_db / 20)
    keep = sorted(((int(s), l) for s, l in zip(starts, loud)
                   if s - PRE >= 0 and s - PRE + length <= len(x) and l > floor), key=lambda sl: -sl[1])
    groups: list[Group] = []
    for s, _ in keep:
        w = x[s - PRE:s - PRE + length]
        best: tuple[float, Group | None, int] = (0.0, None, 0)
        for g in groups:
            score, lag = ncc(g.total / g.n, w, max_lag)
            if abs(score) > abs(best[0]):
                best = (score, g, lag)
        score, g, lag = best
        if g is not None and abs(score) >= threshold:
            aligned = shift(w, lag) * np.sign(score)  # the group's average is w delayed by lag
            centroid = g.total / g.n
            gain = float(np.dot(aligned, centroid) / (np.dot(aligned, aligned) or 1.0))
            g.total = g.total + aligned * gain
            g.n += 1
            g.members.append((s, int(lag), float(np.sign(score)), gain, abs(score)))
        else:
            groups.append(Group(members=[(s, 0, 1.0, 1.0, 1.0)], total=w.astype(np.float64), n=1))
    return sorted(groups, key=lambda g: -g.count)


def template(x: np.ndarray, g: Group, length_ms: float, pre: int = PRE,
             median: bool = False) -> tuple[np.ndarray, int]:
    """The group's hits lined up and averaged over length_ms (starting `pre`
    samples before the hit times): (template, hits used). median=True takes
    the median of the hits (each at its group gain) instead: what plays
    alongside a hit only some of the time drops out instead of averaging
    in. On 30 calibration beats it made the oracle hat templates 6.5 dB
    cleaner against their true files (median null test -14.0 against -7.5
    dB) and the library found 27 of 30 hats instead of 24, but 17 kicks
    instead of 18 and 29 of 30 808s either way."""
    length = samples(length_ms)
    rows = [x[a:a + length] * sign * gain for s, lag, sign, gain, _ in g.members
            if (a := s - pre - lag) >= 0 and a + length <= len(x)]
    if not rows:
        return np.zeros(length, np.float32), 0
    out = np.median(rows, axis=0) if median else np.mean(rows, axis=0, dtype=np.float64)
    return out.astype(np.float32), len(rows)


def rise(y: np.ndarray, at: int, before: int, after: int, level: float = 0.3, floor: bool = False) -> int | None:
    """Where one band of a sound starts, near sample `at` (a hit time): from
    the quietest point (the 1 ms envelope held over 10 ms) in the `before`
    samples up to `at`, the first sample whose 1 ms envelope reaches `level`
    of its peak in the window (up to `after` past `at`), as audio.attack
    reads a sound on its own. What still rings from the sound before is left
    out, and a click that reaches `level` before a louder body counts. None
    when even the quietest point holds `level` of the peak: nothing starts
    here. floor=True measures the climb from the quietest point instead
    (`level` of the way from it to the peak, which must be 6 dB above it),
    for a sound known to be there: on a suite beat an 808 held the snare
    band at 45 % of the snare's peak, and no snare rose from quiet."""
    a, b = max(0, at - before), min(len(y), at + after)
    if b - a < samples(2) or at <= a:
        return None
    w = samples(1)
    env = np.convolve(np.abs(y[a:b]), np.ones(w) / w, mode="same")
    held = maximum_filter1d(env, samples(10))
    peak = env.max()
    v = int(np.argmin(held[:at - a + 1]))
    base = held[v] if floor else 0.0
    if peak == 0 or (peak < 2 * base if floor else held[v] >= level * peak):
        return None
    k = np.flatnonzero(env[v:] >= base + level * (peak - base))
    return a + v + int(k[0]) if len(k) else None


def _start(ys: list[np.ndarray], at: int, before_ms: float, after_ms: float, floor: bool = False) -> int | None:
    found = [r for y in ys if (r := rise(y, at, samples(before_ms), samples(after_ms), floor=floor)) is not None]
    return min(found) if found else None


def start_of(x: np.ndarray, at: int, bands=RISE["low"], before_ms: float = 30.0, after_ms: float = 30.0,
             floor: bool = False) -> int | None:
    """The earliest rise() of x's bands around sample `at` (a hit time, which
    can sit a little before or after the attack): where the sound starts.
    None when no band rises."""
    return _start([band(x, b) for b in bands], at, before_ms, after_ms, floor)


def starts(x: np.ndarray, at, bands=RISE["low"], before_ms: float = 30.0, after_ms: float = 30.0) -> np.ndarray:
    """start_of() for many hit times (samples) of a whole stem, its bands
    filtered once: the starts, -1 where no band rises."""
    ys = [band(x, b) for b in bands]
    return np.array([-1 if (s := _start(ys, int(t), before_ms, after_ms)) is None else s for t in at], dtype=int)


def aligned_template(x: np.ndarray, g: Group, length_ms: float, search_ms: float = 30, bands=None,
                     median: bool = False) -> tuple[np.ndarray, int, int]:
    """(template, hits used, attack): template() over length_ms with the
    sound's start (start_of in `bands`, the climb from the floor: discover
    passes RISE of its role) `attack` samples in: PRE, unless the hits sit
    too close to the start of x. Without bands, or a band that rises, the
    full-band attack (audio.attack). The library is compared over +-10 ms of
    lag only: a template 12 ms off matched its true sample at NCC 0.15
    instead of 0.85. In the full band the attack sat on the tail of the kick
    or 808 before a snare: 10 of 31 calibration templates (all six snares)
    were 13-24 ms early."""
    pre = PRE + samples(search_ms)
    wide, used = template(x, g, length_ms + search_ms * 2, pre=pre, median=median)
    onset = start_of(wide, pre, bands, before_ms=search_ms, floor=True) if bands else None
    if onset is None:
        onset = attack(wide, search=pre + samples(60))
    start = max(0, onset - PRE)
    out = wide[start:start + samples(length_ms)]
    return np.pad(out, (0, samples(length_ms) - len(out))).astype(np.float32), used, onset - start


def onset_template(x: np.ndarray, g: Group, length_ms: float, search_ms: float = 30, bands=None,
                   median: bool = False) -> tuple[np.ndarray, int]:
    """aligned_template() without the attack: (template, hits used), the
    sound's start PRE into the window."""
    return aligned_template(x, g, length_ms, search_ms, bands, median)[:2]


def merge(x: np.ndarray, groups: list[Group], threshold: float = 0.8, length_ms: float = 100,
          min_count: int = 2) -> list[Group]:
    """Join groups whose averages match: a sample heard over different
    overlapping sounds first lands in several groups."""
    groups = [g for g in groups if g.count >= min_count]
    merged = True
    while merged:
        merged = False
        templates = [template(x, g, length_ms)[0] for g in groups]
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                score, d = ncc(templates[a], templates[b], samples(4))
                if abs(score) >= threshold:
                    sign = float(np.sign(score))
                    for s, lag, sg, gain, sc in groups[b].members:
                        groups[a].members.append((s, lag + d, sg * sign, gain, sc))
                    del groups[b]
                    merged = True
                    break
            if merged:
                break
    return sorted(groups, key=lambda g: -g.count)


def classify(t: np.ndarray) -> str:
    """kick / snare / hat from where the energy of the first 100 ms is. Crude:
    an 808 under a snare makes it look like a kick; prefer the band that
    matches best when a library is at hand."""
    head = t[: samples(100)]
    spec = np.abs(np.fft.rfft(head)) ** 2
    f = np.fft.rfftfreq(len(head), 1 / SR)
    total = spec.sum() or 1.0
    low, high = spec[f < 150].sum() / total, spec[f > 3500].sum() / total
    if high > 0.55:
        return "hat"
    if low > 0.5:
        return "kick"
    return "snare"
