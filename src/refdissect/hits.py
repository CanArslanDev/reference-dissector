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

from .audio import PRE, SR, attack, band, samples, shift
from .xcorr import ncc

HOP = 128


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


def template(x: np.ndarray, g: Group, length_ms: float, pre: int = PRE) -> tuple[np.ndarray, int]:
    """The group's hits lined up and averaged over length_ms (starting `pre`
    samples before the hit times): (template, hits used)."""
    length = samples(length_ms)
    total = np.zeros(length, np.float64)
    used = 0
    for s, lag, sign, gain, _ in g.members:
        a = s - pre - lag
        if a < 0 or a + length > len(x):
            continue
        total += x[a:a + length] * sign * gain
        used += 1
    return (total / max(used, 1)).astype(np.float32), used


def onset_template(x: np.ndarray, g: Group, length_ms: float, search_ms: float = 30) -> tuple[np.ndarray, int]:
    """template() with the attack (audio.attack) exactly PRE into the window.
    Hit times from an onset-strength peak can sit several ms off the attack
    (more in one band than another), and the library is compared over +-10 ms
    of lag only: a template 12 ms off matched its true sample at NCC 0.15
    instead of 0.85."""
    pre = PRE + samples(search_ms)
    wide, used = template(x, g, length_ms + search_ms * 2, pre=pre)
    onset = attack(wide, search=pre + samples(60))
    start = max(0, onset - PRE)
    out = wide[start:start + samples(length_ms)]
    return np.pad(out, (0, samples(length_ms) - len(out))).astype(np.float32), used


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
