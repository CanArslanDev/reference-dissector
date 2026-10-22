"""Every drum moment explained by every template at once: layered and
simultaneous hits, flams, rolls and pitched hits.

detect() used to keep one template per moment and band (the best NCC within
15 ms) and hits at least 35 ms apart. That loses, by design, a hat under a
snare, a clap layered on a snare, the grace note of a flam and the hits of a
64th roll. explain() instead:

1. takes each template's matched-filter hits (its own band and attack, NCC
   at its threshold) 8 ms apart (tones 35 ms, the low band 100 ms), and
   those of pitched copies of the noisy templates (1-5 semitones up and
   down, as Simpler plays them);
2. fits the neighbourhood of every hit (+-150 ms) as a non-negative sum of
   all the hits in it (NNLS), on the kick, snare and hat bands concatenated,
   each band weighted by 1 / RMS of the stem in it. Two hits whose placed
   templates correlate 0.6 or more are one sound found twice (a duplicate
   template, a pitched copy, a kick ringing on): the one that explains more
   keeps the moment, and templates that keep losing to each other are
   merged;
3. keeps a hit whose gain reaches a quarter of its template's usual gain and
   whose share of the fit (what the fit loses without it) is above what the
   template takes at shifted, empty windows (the null);
4. looks where the kept hits leave something out: a template at its offset
   under the hits of a template that holds it, where its matched filter
   peaks below its threshold, and at the onsets of its band (not for tones:
   their matched filter has a broad maximum). A hit found there needs half
   its usual gain, or a quarter where what the others leave matches the
   template as well as a matched-filter hit does, and the null (a backstop
   there: on 48 suite beats it alone rejected 44 of the 1,724 probe hits
   over those floors).

Before the fit, a template that holds another (a snare template holds the
hat that plays on every snare) has it subtracted, else it takes that
sound's place at every hit (purification), and a template the others
explain (a duplicate, a kick+hat found where they always coincide) is left
out, its hits handed to the templates that explain it. Two kinds of
template are never held or probed for, as they only find another
template's sound a second time: one of discover()'s residual pass (found
where no template of its band plays: it keeps its matched-filter hits, and
gets no pitched copies), and one that a bigger template of its fit band
holds within NEAR_MS (a variant of that sound). Re-fit on discover()'s
templates of the 48 suite beats (htdemucs), the drums' F1 at 20 ms went
0.847 -> 0.879 (mean over sounds; 0.801 -> 0.844 over hits) and the false
hits from 68.6 to 43.1 a beat; open hats 0.901 -> 0.979, snares 0.933 ->
0.961; claps 0.580 -> 0.500, as the layers scenario's clap had been found
only by a residual template held under the snare.

On 22 calibration-suite beats (layers, rolls, flams, dynamics, v1, v1orig,
producer, full, kick808), against one template per moment and band: hats
within 20 ms of a snare or clap 16 -> 324 of 432 found, 64th hat rolls 19 ->
50 of 64, drum F1 at 20 ms over hits 0.613 -> 0.781 (the mean over sounds
0.421 -> 0.584; docs/01-pipeline.md).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls
from scipy.signal import fftconvolve, find_peaks

from .audio import PRE, SR, band, samples, transpose
from .hits import onsets as band_onsets
from .xcorr import ncc, sliding_ncc

FIT_BANDS = ("kick", "snare", "hat")
ROW = {"low": 0, "kick": 0, "snare": 1, "hat": 2}  # the fit band a template found in a detection band is heard in
LEAD = samples(10)  # zeros before the stem: a hit at 0 s has its window, PRE before the attack
CLOSEST_MS = {"kick": 8, "snare": 8, "hat": 8, "low": 100}  # flams, 64th rolls; a low tone echoes itself
CANDIDATE_NCC = 0.3  # matched-filter peaks below the threshold that are probed
CONTEXT_MS = 150
MATCHED_GAIN, LAYER_GAIN = 0.25, 0.5  # x the template's usual gain
NULL_SHIFTS_MS = (-47, -29, 29, 47)
NULL_QUANTILE = 0.95
HOLDS = ((0.3, 2.0), (0.015, 0.3))  # S holds H: LS coefficient c, share of S's energy over H's span
COMPOSITE = 0.8  # share of a template the earlier ones explain: not a sound of its own
ECHO = 0.5  # share of a template's own band an earlier one explains: its matched filter repeats that one
SAME_SOUND = 0.6  # placed templates this alike are one sound found twice
# Share of a template's hits that are another's found twice: one sound. A
# sound layered on another in half its hits (a clap on every other snare)
# loses half; 13 of the 15 merges on the suite lost 0.67-1.0 of the fewer's
# hits, and the other two (0.57, 0.63) were sounds of their own
DUPLICATE = 2 / 3
TONAL = (0.5, 0.25, 0.5)  # a tone: this share in the kick band, or this one ringing this much (tonal)
SHARP = 0.5  # NCC with its copy 1 semitone off under which a template is noisy enough for pitched copies
SEMITONES = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
# Where a held template may sit in the one holding it. Either window may start
# first: on the conftest beat discover()'s hat template starts 23 ms before
# its hats' attack, so the hat sits at -23 ms in the snare; suite holds run
# from -8 to +29 ms
LAG_MS = (-30, 30)
PAIR_MS, PAIR_MIN = 40, 5  # co-detections that give a pair offset
SEARCH_MS, ONSET_MS = 3, 6  # how far a probe looks, around a predicted window start and around an onset
SAME_MS = 15  # a probe where the template already plays is not a new hit
FLAM_MS = 35  # a quieter hit this close before another of the same sound is its grace note
# A tone's matched filter peaks again a period or two on: its candidates are
# 35 ms apart (as detect() kept every template's) and its ringing is no hit
# (TAIL), wherever it was found. With 8 ms, a kick found in the snare band
# was found again 15-23 ms after every strike (v1-1)
TONE_MS = 35
TAIL = (0.25, 0.5)  # a low or tonal template's hit this soon (s) after one this much louder is its ringing (_tails)
NEAR_MS = 2  # a template held this close by a bigger one of its fit band is a variant of that sound (_held)


@dataclass
class Column:
    """A template as the fit sees it: its first n samples in each FIT_BAND,
    weighted (and purified), with what it came from."""
    name: str
    template: int
    data: np.ndarray  # (len(FIT_BANDS), n)
    semitones: int = 0

    @property
    def n(self) -> int:
        return self.data.shape[1]


@dataclass
class Candidates:
    """Placed columns: column, window start (samples into the padded stem),
    matched-filter NCC, and where each came from."""
    col: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    p: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    ncc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    source: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.p)

    def add(self, col, p, ncc_, source: str) -> None:
        p = np.atleast_1d(np.asarray(p, dtype=int))
        self.col = np.r_[self.col, np.broadcast_to(np.asarray(col, dtype=int), p.shape)]
        self.p = np.r_[self.p, p]
        self.ncc = np.r_[self.ncc, np.broadcast_to(np.asarray(ncc_, dtype=float), p.shape)]
        self.source += [source] * len(p)

    def take(self, keep) -> Candidates:
        keep = np.asarray(keep)
        keep = np.flatnonzero(keep) if keep.dtype == bool else keep.astype(int)
        return Candidates(self.col[keep], self.p[keep], self.ncc[keep], [self.source[i] for i in keep])

    def sorted(self) -> Candidates:
        return self.take(np.lexsort((self.col, self.p)))

    def join(self, other: Candidates) -> Candidates:
        out = self.take(np.arange(len(self)))
        out.col, out.p, out.ncc = np.r_[out.col, other.col], np.r_[out.p, other.p], np.r_[out.ncc, other.ncc]
        out.source += other.source
        return out.sorted()


def _weights(x: np.ndarray) -> np.ndarray:
    """1 / RMS of the stem in each FIT_BAND: each band counts alike in the
    fit, so the hat band is not drowned by the kick's."""
    return np.array([1.0 / (np.sqrt(np.mean(band(x, b).astype(np.float64) ** 2)) or 1.0) for b in FIT_BANDS])


def _bands(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.stack([band(x, b) * np.float32(wb) for b, wb in zip(FIT_BANDS, w)]).astype(np.float32)


def _xcorr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """r[d + b.n - 1] = sum over bands and m of a[m] b[m - d]: the inner
    product of a with b placed d samples later."""
    return sum(fftconvolve(a[i].astype(np.float64), b[i, ::-1].astype(np.float64)) for i in range(len(a)))


class Kit:
    """The columns, their cross-correlations (computed when first needed)
    and the stem's weighted bands: what a fit needs."""

    def __init__(self, y: np.ndarray, columns: list[Column]):
        self.y = y
        self.columns = columns
        self._r: dict[tuple[int, int], np.ndarray] = {}

    def r(self, i: int, j: int) -> np.ndarray:
        if (i, j) not in self._r:
            self._r[(i, j)] = _xcorr(self.columns[i].data, self.columns[j].data)
        return self._r[(i, j)]

    def inner(self, ci: int, pi: int, cj: int, pj: int) -> float:
        """Inner product of column ci at pi with column cj at pj."""
        d, nj, ni = pj - pi, self.columns[cj].n, self.columns[ci].n
        return float(self.r(ci, cj)[d + nj - 1]) if -nj < d < ni else 0.0

    def gram(self, col: np.ndarray, p: np.ndarray) -> np.ndarray:
        n = len(p)
        g = np.zeros((n, n))
        for a in range(n):
            for b in range(a, n):
                g[a, b] = g[b, a] = self.inner(col[a], p[a], col[b], p[b])
        return g

    def rhs(self, col: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Inner product of the stem with each placed column."""
        out = np.zeros(len(p))
        for c in np.unique(col):
            data = self.columns[c].data
            every = np.flatnonzero(col == c)
            for idx in np.array_split(every, max(1, len(every) // 512)):  # (bands, hits, n) windows: 512 at a time
                win = self.y[:, p[idx, None] + np.arange(data.shape[1])]
                out[idx] = np.einsum("bhn,bn->h", win, data, dtype=np.float64)
        return out

    def energy(self, c: int) -> float:
        return float(self.r(c, c)[self.columns[c].n - 1])


def solve(g: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """NNLS from the normal equations: gains >= 0 minimising
    |y - A gains|^2 - |y|^2 = gains.G.gains - 2 gains.b (returned too)."""
    n = len(b)
    if not n:
        return np.zeros(0), 0.0
    ridge = 1e-9 * (np.trace(g) / n or 1.0)
    low = np.linalg.cholesky(g + ridge * np.eye(n))
    gains, _ = nnls(low.T, solve_triangular(low, b, lower=True), maxiter=50 * n)
    return gains, float(gains @ g @ gains - 2 * gains @ b)


def fit(kit: Kit, cands: Candidates, drops: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate's gain, and its drop (gain^2 units: what the fit of its
    neighbourhood loses without it, over its column's energy), from the NNLS
    of every candidate within CONTEXT_MS of it. cands sorted by p."""
    context = samples(CONTEXT_MS)
    lo, hi = np.searchsorted(cands.p, cands.p - context, "left"), np.searchsorted(cands.p, cands.p + context, "right")
    b_all = kit.rhs(cands.col, cands.p)
    gains, drop = np.zeros(len(cands)), np.zeros(len(cands))
    last = None
    for i in range(len(cands)):
        if last is None or last[0] != (lo[i], hi[i]):
            idx = np.arange(lo[i], hi[i])
            g = kit.gram(cands.col[idx], cands.p[idx])
            last = ((lo[i], hi[i]), g, b_all[idx], *solve(g, b_all[idx]))
        _, g, b, sol, e = last
        k = i - lo[i]
        gains[i] = sol[k]
        if drops and sol[k] > 0:
            rest = np.r_[0:k, k + 1:len(b)]
            drop[i] = (solve(g[np.ix_(rest, rest)], b[rest])[1] - e) / (g[k, k] or 1.0)
    return gains, drop


# ── How the templates overlap ──

def _placed(data: np.ndarray, lag: int, n: int) -> np.ndarray:
    """data moved lag samples later, over [0, n)."""
    out = np.zeros((data.shape[0], n), dtype=np.float64)
    a, b = max(0, lag), min(n, lag + data.shape[1])
    if b > a:
        out[:, a:b] = data[:, a - lag:b - lag]
    return out


def containment(s: np.ndarray, h: np.ndarray, lags: tuple[int, int]) -> tuple[float, float, int]:
    """(share, c, lag): h's least-squares coefficient c inside s at the lag
    in lags where c*h explains the largest share of s's energy over h's
    span (bands as rows)."""
    n = h.shape[1]
    lo, hi = lags
    seg = _placed(s, -lo, hi - lo + n)  # seg[:, k + m] = s[:, lo + k + m]
    num = sum(np.correlate(seg[i], h[i].astype(np.float64), "valid") for i in range(len(seg)))
    cum = np.cumsum(np.r_[0.0, (seg ** 2).sum(axis=0)])
    ss = np.maximum(cum[n:] - cum[:-n], 1e-12)
    hh = float((h.astype(np.float64) ** 2).sum()) or 1e-12
    c = np.maximum(num / hh, 0.0)
    share = c ** 2 * hh / ss
    k = int(np.argmax(share))
    return float(share[k]), float(c[k]), lo + k


def explained(t: np.ndarray, others: list[np.ndarray], lags: tuple[int, int]) -> tuple[float, np.ndarray, list[int]]:
    """Share of t that a non-negative sum of the others explains, each at
    its own best lag; their coefficients and lags."""
    n = t.shape[1]
    if not others:
        return 0.0, np.zeros(0), []
    best = [containment(t, o[:, :n], lags)[2] for o in others]
    a = np.stack([_placed(o, lag, n).ravel() for o, lag in zip(others, best)], axis=1)
    coef, res = nnls(a, t.astype(np.float64).ravel())
    return 1.0 - res ** 2 / (float((t.astype(np.float64) ** 2).sum()) or 1e-12), coef, best


def relations(full: list[np.ndarray], own: list[list[np.ndarray]], n: list[int], hits: list[int],
              holdable: set[int], rows: list[int]) -> dict:
    """How the templates overlap, from their weighted bands (full; n: column
    lengths; rows: the fit band each is heard in) and their own detection
    bands (own[k][j]: template j in k's band), taking those averaged from
    more hits first:
    - composite: the earlier ones explain COMPOSITE of it in all bands (a
      duplicate, or a kick+hat where both have templates): left out;
    - echo: an earlier one explains ECHO of it in its own band, so its
      matched filter repeats that one's (a kick+hat found in the hat band
      finds every hat): never probed for;
    - holds: s holds h, another sound that may be held (holdable, _held)."""
    order = sorted(range(len(full)), key=lambda k: -hits[k])
    near = (-samples(10), samples(10))
    composite: dict[int, dict] = {}
    echo: dict[int, float] = {}
    kept: list[int] = []
    for k in order:
        share, coef, lags = explained(full[k][:, :n[k]], [full[j] for j in kept], near)
        if kept and share >= COMPOSITE:
            composite[k] = {"share": round(share, 3),
                            "by": {kept[j]: (round(float(c), 3), lag) for j, (c, lag) in enumerate(zip(coef, lags))
                                   if c > 0}}
            continue
        mine, _, _ = explained(own[k][k][:, :n[k]], [own[k][j] for j in kept], near)
        if kept and mine >= ECHO:
            echo[k] = round(mine, 3)
        kept.append(k)
    holds = {s: held for s in kept if (held := _held(full, n, s, [h for h in kept if h != s and h not in echo
                                                                      and h in holdable], rows, hits))}
    return {"composite": composite, "echo": echo, "kept": sorted(kept), "holds": holds}


def _held(full: list[np.ndarray], n: list[int], s: int, others: list[int], rows: list[int],
          hits: list[int]) -> list[tuple[int, float, int, float]]:
    """The templates s holds, (h, c, lag, share) each: where h sits (lag),
    read in h's own band (rows[h]); h's least-squares coefficient c there
    within HOLDS[0] (an accented hat under every snare is 1.4 times the
    average hat) and the share of s over h's span it explains within
    HOLDS[1] (a near duplicate explains far more), both over all bands, and
    h there in its own band too, at HOLDS[0][0] or
    more: a hat's average carries the body of the kicks some hats sit on,
    and over all bands that body made a kick template hold a hat it never
    plays (full-2: c 0.61 over all bands, 3.2 in the kick band, 0.06 in the
    hat band); its purified column held a negative hat, and a hat was found
    under every kick. c is the smaller of the two readings, so no band loses
    more of h than it holds. Taken greedily, each in what the earlier ones
    leave, a second one needing twice the first's least share: what the
    first leaves is mostly s itself, and a duplicate of the first or a
    noisy sound matches some of it (an open hat 2.9 % in a snare on a
    calibration beat, where it never plays). A template of s's own fit band
    that s holds within NEAR_MS, averaged from fewer hits (hits), is a
    variant of s's sound, not a layer: on tempo160-1 the main hat held a
    second hat-band template at 0.4 ms, which then got 37 hits under the
    hats, 33 of them hats the main template had already. Read over all
    bands, the lag followed a snare's body: on three seed-1 beats the snare
    held the hat 21 ms after the hat it plays with (tempo95-1: 23.8 ms over
    all bands, 2.6 ms in the hat band), and the hats found under the snares
    came 20.5 ms late."""
    rest = full[s].astype(np.float64)
    lags = (samples(LAG_MS[0]), samples(LAG_MS[1]))
    out: list[tuple[int, float, int, float]] = []
    while others:
        least = HOLDS[1][0] * (2 if out else 1)
        found = []
        for h in others:
            d, r = full[h][:, :n[h]].astype(np.float64), rows[h]
            _, _, lag = containment(rest[r:r + 1], d[r:r + 1], lags)
            share, c, _ = containment(rest, d, (lag, lag))
            if r == rows[s] and abs(lag) <= samples(NEAR_MS) and hits[s] > hits[h]:
                continue
            _, mine, _ = containment(rest[r:r + 1], d[r:r + 1], (lag - samples(1), lag + samples(1)))
            if HOLDS[0][0] <= min(c, mine) and c <= HOLDS[0][1] and least <= share <= HOLDS[1][1]:
                found.append((share, min(c, mine), lag, h))
        if not found:
            break
        share, c, lag, h = max(found)
        out.append((h, c, lag, share))
        rest = rest - c * _placed(full[h][:, :n[h]], lag, rest.shape[1])
        others = [o for o in others if o != h]
    return out


def purified(s: np.ndarray, held: list[tuple[np.ndarray, int, float]]) -> np.ndarray:
    """s less each template it holds, (data, lag, c) each: c as _held took
    it, the smaller of its readings over all bands and in h's own band."""
    out = s.astype(np.float64).copy()
    for d, lag, c in held:
        out -= c * _placed(d, lag, s.shape[1])
    return out.astype(np.float32)


def ring(x: np.ndarray, n: int) -> float:
    """How alike x's first n samples are to x 8-35 ms later (best NCC): a
    tone repeats itself, a burst of noise does not."""
    x = np.asarray(x, dtype=np.float64)
    t = x[:n]
    best = 0.0
    for lag in range(samples(8), samples(35), samples(0.5)):
        seg = x[lag:lag + n]
        if len(seg) == n:
            best = max(best, float(seg @ t / ((np.linalg.norm(seg) * np.linalg.norm(t)) or 1.0)))
    return best


def low_share(full: np.ndarray, n: int) -> float:
    """The share of a template's first n samples' energy (weighted bands) in
    the kick band."""
    e = (full[:, :n].astype(np.float64) ** 2).sum(axis=1)
    return float(e[0] / (e.sum() or 1.0))


def tonal(full: np.ndarray, n: int) -> bool:
    """Whether a template (weighted bands) is a tone: TONAL[0] of its first n
    samples' energy in the kick band, or TONAL[1] of it there and that band
    as alike to itself 8-35 ms later as TONAL[2] (NCC). A kick may ring
    little and still be one (0.29 with 59 % there, leadin-2: probed as a
    noisy sound, it was found 36 times where no drum plays)."""
    share = low_share(full, n)
    return share >= TONAL[0] or (share >= TONAL[1] and ring(full[0], n) >= TONAL[2])


def pitch_selectivity(audio: np.ndarray, role_band: str, n: int) -> float:
    """NCC of a template with itself a semitone up and down (in its band,
    the attack searched +-1 ms): a hat drops to 0.1-0.3, a kick stays
    above 0.6."""
    t = band(audio, role_band)[:n]
    return max(abs(ncc(t, band(transpose(audio, st), role_band)[:n], samples(1))[0]) for st in (-1, 1))


# ── Probes: a template where the kept hits leave something out ──

def probe(kit: Kit, base: Candidates, gains: np.ndarray, col: int, centers, search: int) -> dict[str, np.ndarray]:
    """The column near each center: where (within +-search) it best matches
    what the base fit leaves, and the gain, drop and NCC with that residual it
    gets fit with the base candidates within CONTEXT_MS."""
    data = kit.columns[col].data
    n = data.shape[1]
    context = samples(CONTEXT_MS)
    out: dict[str, list] = {k: [] for k in ("p", "gain", "drop", "ncc")}
    for q in np.asarray(centers, dtype=int):
        if q - search < 0 or q + search + n > kit.y.shape[1]:
            continue
        idx = np.arange(*np.searchsorted(base.p, [q - context, q + context]))
        seg = kit.y[:, q - search:q + search + n].astype(np.float64)
        mf = sum(np.correlate(seg[b], data[b].astype(np.float64), "valid") for b in range(len(seg)))
        s = np.arange(-search, search + 1)
        for d in idx[gains[idx] > 0]:
            nd = kit.columns[base.col[d]].n
            lag = base.p[d] - (q + s)
            ok = (lag > -nd) & (lag < n)
            mf[ok] -= gains[d] * kit.r(col, base.col[d])[lag[ok] + nd - 1]
        p = int(q + s[int(np.argmax(mf))])
        cols, ps = np.r_[base.col[idx], col], np.r_[base.p[idx], p]
        g, b = kit.gram(cols, ps), kit.rhs(cols, ps)
        sol, e = solve(g, b)
        resid = kit.y[:, p:p + n].astype(np.float64)
        for k in np.flatnonzero(sol[:-1] > 0):
            resid -= sol[k] * _placed(kit.columns[cols[k]].data, ps[k] - p, n)
        out["p"].append(p)
        out["gain"].append(sol[-1])
        out["drop"].append((solve(g[:-1, :-1], b[:-1])[1] - e) / (g[-1, -1] or 1.0))
        out["ncc"].append(float((resid * data).sum()) / (float(np.linalg.norm(resid) * np.linalg.norm(data)) or 1.0))
    return {k: np.array(v) for k, v in out.items()}


def pair_offset(s_p: np.ndarray, h_p: np.ndarray) -> tuple[int | None, int]:
    """Where h plays relative to s where both were found: the densest 1 ms
    bin of their offsets within PAIR_MS (+-2 ms), if PAIR_MIN agree.
    Returns (offset in samples or None, how many agree)."""
    if not len(s_p) or not len(h_p):
        return None, 0
    d = (h_p[None, :] - s_p[:, None]).ravel()
    d = d[np.abs(d) <= samples(PAIR_MS)]
    if not len(d):
        return None, 0
    bins = np.round(d / samples(1)).astype(int)
    vals, counts = np.unique(bins, return_counts=True)
    near = d[np.abs(bins - vals[np.argmax(counts)]) <= 2]
    return (int(np.median(near)), len(near)) if len(near) >= PAIR_MIN else (None, len(near))


def _far(q: np.ndarray, taken: np.ndarray, ms: float) -> np.ndarray:
    """Which of q have nothing in taken within ms."""
    q = np.asarray(q, dtype=int)
    if not len(taken) or not len(q):
        return np.ones(len(q), bool)
    return np.min(np.abs(q[:, None] - np.asarray(taken)[None, :]), axis=1) > samples(ms)


def _free(centers: np.ndarray, taken: np.ndarray) -> np.ndarray:
    """The centers with no hit in taken within SAME_MS."""
    return np.asarray(centers, dtype=int)[_far(centers, taken, SAME_MS)]


# ── Deciding ──

def distinct(kit: Kit, cands: Candidates, gains: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The candidates that are not a louder candidate found again: two
    placed columns that correlate SAME_SOUND or more explain one sound two
    ways (a duplicate template, a pitched copy, a kick ringing one period
    on), and the one that explains more of the stem keeps it. A hat under a
    snare, or a flam's grace note, correlate far less. Returns the kept
    indices and (lost, kept) pairs."""
    energy = np.array([kit.energy(c) for c in range(len(kit.columns))])
    explains = gains ** 2 * energy[cands.col]
    longest = max(c.n for c in kit.columns)
    kept: list[int] = []
    at: list[int] = []  # the kept ones' window starts, sorted
    pairs: list[tuple[int, int]] = []
    for i in np.argsort(-explains, kind="stable"):
        near = kept[bisect.bisect_left(at, cands.p[i] - longest + 1):bisect.bisect_right(at, cands.p[i] + longest - 1)]
        same = [j for j in near if abs(kit.inner(cands.col[i], cands.p[i], cands.col[j], cands.p[j]))
                >= SAME_SOUND * np.sqrt(energy[cands.col[i]] * energy[cands.col[j]])]
        rival = max(same, key=lambda j: explains[j]) if same else None
        if rival is not None:  # also a hit the fit gave nothing: its template repeats the other's
            pairs.append((int(i), rival))
        elif gains[i] > 0:
            k = bisect.bisect_right(at, cands.p[i])
            at.insert(k, int(cands.p[i]))
            kept.insert(k, int(i))
    return np.array(sorted(kept), dtype=int), pairs


def duplicates(cands: Candidates, pairs: list[tuple[int, int]], first: list[int],
               kind: list[bool]) -> dict[int, tuple[int, int]]:
    """Columns of one kind (kind: tonal or not; None, never merged) that
    keep losing their hits to each other (DUPLICATE of the fewer's hits
    found twice; a kick+hat template and a hat template share the hat, not
    the kind): {merged: (into, offset)} with p_into = p_merged + offset,
    into the one earlier in first. At half, a clap on every other snare at
    a constant offset (its template and the snare's each holding half the
    other) lost half its hits to the snare, was merged into it, and its
    solo hits went too: 0 of 32 claps. Matching as waveforms instead (NCC
    0.85 in both bands, as discover() asks) kept that clap apart but also
    one kick found twice 17 ms apart (0.77 in the snare band; v1orig-5 kick
    F1@20 0.98 -> 0.22)."""
    count: dict[tuple[int, int], list[int]] = {}
    for i, j in pairs:
        a, b = int(cands.col[i]), int(cands.col[j])
        if a != b and kind[a] is not None and kind[a] == kind[b]:
            key = (min(a, b), max(a, b))
            count.setdefault(key, []).append(int(cands.p[j] - cands.p[i]) if a == key[0] else
                                             int(cands.p[i] - cands.p[j]))
    hits = np.bincount(cands.col, minlength=len(first))
    out: dict[int, tuple[int, int]] = {}
    for (a, b), d in sorted(count.items(), key=lambda kv: -len(kv[1])):
        if len(d) >= DUPLICATE * min(hits[a], hits[b]) and a not in out and b not in out:
            off = int(np.median(d))  # p_b - p_a
            if first.index(a) < first.index(b):
                out[b] = (a, -off)
            else:
                out[a] = (b, off)
    return out


def _null(kit: Kit, base: Candidates, gains: np.ndarray, col: int) -> float:
    """The drop the column gets where it does not play: its own hits shifted
    by NULL_SHIFTS_MS (none of its hits within SAME_MS), searched as a layer
    is. Its NULL_QUANTILE."""
    own = base.p[base.col == col]
    if not len(own):
        return 0.0
    centers = (own[:, None] + np.array([samples(s) for s in NULL_SHIFTS_MS])[None, :]).ravel()
    found = probe(kit, base, gains, col, _free(centers, own), samples(SEARCH_MS))
    return float(np.quantile(found["drop"], NULL_QUANTILE)) if len(found["drop"]) else 0.0


def passes(got: dict, usual: float, null: float, threshold: float, min_gain: float) -> np.ndarray:
    """Which probe hits are hits: above the null, and at LAYER_GAIN of the
    usual gain, or at min_gain where what the others leave looks like the
    template as much as a matched-filter hit does (NCC with the residual at
    the threshold: a flam's grace note at -6 dB, its main hit taken away)."""
    loud = got["gain"] >= LAYER_GAIN * usual
    clear = (got["gain"] >= min_gain * usual) & (got["ncc"] >= threshold)
    return (got["drop"] >= null) & (loud | clear)


def decay(full: np.ndarray, n: int) -> np.ndarray:
    """How much of a template's first n samples (weighted bands) its own
    continuation holds d samples on, for d up to TAIL[0]: the gain its
    ringing gets where the column is placed again. A tom's (110 Hz, 150 ms)
    found in the snare band: 0.61 at 54 ms; a kick's on v1-1: 0.70 at 23
    ms."""
    col = full[:, :n].astype(np.float64)
    span = min(full.shape[1], n + samples(TAIL[0] * 1000))
    r = sum(np.correlate(full[b, :span].astype(np.float64), col[b], "valid") for b in range(len(col)))
    return r / (float((col ** 2).sum()) or 1.0)


def _tails(cands: Candidates, gains: np.ndarray, rings: dict[int, np.ndarray]) -> np.ndarray:
    """Hits of the low and tonal columns (rings: their decay) that are not
    the ringing of a louder hit of the same column: within TAIL[0], below
    TAIL[1] of its gain or below what its own decay holds there (within 1
    ms), whichever is more."""
    keep = np.ones(len(cands), bool)
    for c, r in rings.items():
        idx = np.flatnonzero(cands.col == c)
        last = None
        for i in idx[np.argsort(cands.p[idx])]:
            d = cands.p[i] - cands.p[last] if last is not None else len(r)
            if d < len(r) and gains[i] < max(TAIL[1], r[max(0, d - samples(1)):d + samples(1) + 1].max()) * gains[last]:
                keep[i] = False
                continue
            last = i
    return keep


# ── Explaining a stem ──

@dataclass
class Explained:
    """Hits per template ({name: [{t, ncc, gain, source, ...}]}) and what
    decided them."""
    events: dict[str, list[dict]]
    info: dict


@dataclass
class Scene:
    """What the steps of explain() share: the templates, their attack lengths
    (n) and how they overlap (relations); the columns (kit.columns: kept
    templates purified, then pitched copies) with the padded stem's bands;
    the column of each kept template; per column its threshold; the decay of
    the low and tonal columns; and per column its usual gain and null, as
    the steps set them."""
    templates: list
    n: list[int]
    rel: dict
    tones: set[int]
    noisy: list[int]
    col_of: dict[int, int]
    kit: Kit
    thr: np.ndarray
    rings: dict[int, np.ndarray]
    usual: np.ndarray | None = None
    null: np.ndarray | None = None


def explain(x: np.ndarray, templates: list, detect_ms: dict, thresholds: dict, closest_ms: dict | None = None,
            min_gain: float = MATCHED_GAIN, onsets: bool = True, pitched: bool = True) -> Explained:
    """Every hit of every template (detect.Template), as the module
    docstring says. detect_ms: how much of a template's attack its matched
    filter uses, per band; thresholds: the NCC at which that alone is a
    hit."""
    if not templates:
        return Explained({}, {})
    closest_ms = {**CLOSEST_MS, **(closest_ms or {})}
    xp = np.r_[np.zeros(LEAD, np.float32), np.asarray(x, np.float32),
               np.zeros(max(len(tp.audio) for tp in templates), np.float32)]
    sc, decided = _scene(xp, templates, detect_ms, thresholds, pitched)
    cands, weak = _matched(sc, xp, thresholds, closest_ms)
    cands, merged = _one_each(sc, cands)
    cands, gains = _significant(sc, cands, min_gain)
    layers, offsets = _probes(sc, cands, gains, merged, weak, xp, min_gain, onsets)
    cands, gains = _settle(sc, cands, layers, min_gain)
    columns = sc.kit.columns
    info = {"composite": decided["composite"],
            "merged": {columns[a].name: {"into": columns[b].name, "offset_ms": round(off / SR * 1000, 2)}
                       for a, (b, off) in merged.items()},
            "echo": decided["echo"], "tonal": decided["tonal"], "purified": decided["purified"],
            "pair_offsets": offsets, "pitched": decided["pitched"],
            "usual_gain": {c.name: round(float(u), 4) for c, u in zip(columns, sc.usual) if not c.semitones},
            "null_drop": {c.name: round(float(v), 5) for c, v in zip(columns, sc.null) if not c.semitones}}
    return Explained(_events(templates, sc.kit, cands, gains), info)


def _scene(xp: np.ndarray, templates: list, detect_ms: dict, thresholds: dict, pitched: bool) -> tuple[Scene, dict]:
    """The columns: kept templates (relations) purified of what they hold,
    then pitched copies of the noisy ones a semitone changes (SHARP); and
    what decided them (composites, echoes, tones, purities, sharpness)."""
    w = _weights(xp)
    names = [tp.name for tp in templates]
    full = [_bands(tp.audio, w) for tp in templates]
    n = [samples(detect_ms[tp.band]) for tp in templates]
    inband = {b: [band(tp.audio, b)[None, :] for tp in templates] for b in {tp.band for tp in templates}}
    tones = {k for k in range(len(templates)) if tonal(full[k], n[k])}
    rows = [ROW[tp.band] for tp in templates]
    # A tone found in the hat band with less than TONAL[0] of its energy in
    # the kick band is a hat whose average carries the body of the kicks
    # under some of its hits: held as a hat (a tone is never held: a kick
    # template's low ring explains a little of almost anything), its hits as
    # close as a hat's and no hit a ringing (kicks at half level on 2 of
    # every 8 hats: 24 of 24 hats under the snares, against 0); still not
    # probed (a kick found in the hat band, 31 % in the kick band, probed:
    # kick808-1 drums F1@20 0.636 -> 0.415)
    hatlike = {k for k in tones if rows[k] == ROW["hat"] and low_share(full[k], n[k]) < TONAL[0]}
    # A residual template (discover's second pass) was found where no template
    # of its band plays. Held under another, looked for at its band's onsets
    # or copied at other pitches, it found that template's sound a second
    # time: on v1-1, 40 of the 50 hits one got under other templates were the
    # main hat's own hits again, 9 were on no hit. It keeps its matched-filter
    # hits
    residual = {k for k, tp in enumerate(templates) if getattr(tp, "found", "grid") == "residual"}
    holdable = {k for k in range(len(templates)) if (k not in tones or k in hatlike) and k not in residual}
    rel = relations(full, [inband[tp.band] for tp in templates], n, [tp.hits for tp in templates], holdable, rows)
    tones &= set(rel["kept"])
    noisy = [k for k in rel["kept"] if k not in tones | residual and k not in rel["echo"]]
    columns, purity = [], {}
    for k in rel["kept"]:
        held = rel["holds"].get(k, [])
        data = purified(full[k], [(full[h][:, :n[h]], lag, c) for h, c, lag, _ in held])
        purity[names[k]] = {names[h]: {"c": round(float(c), 3), "lag_ms": round(lag / SR * 1000, 2),
                                       "share": round(share, 3)} for h, c, lag, share in held}
        columns.append(Column(names[k], k, data[:, :n[k]]))
    sharp = {k: round(pitch_selectivity(templates[k].audio, templates[k].band, n[k]), 3) for k in noisy} \
        if pitched else {}
    for k in [k for k, v in sharp.items() if v < SHARP]:
        for st in SEMITONES:
            columns.append(Column(names[k], k, _bands(transpose(templates[k].audio, st), w)[:, :n[k]], st))
    kit = Kit(_bands(xp, w), columns)
    thr = np.array([thresholds[templates[col.template].band] for col in columns])
    rings = {c: decay(full[col.template], n[col.template]) for c, col in enumerate(columns)
             if templates[col.template].band == "low" or col.template in tones - hatlike}
    sc = Scene(templates, n, rel, tones, noisy, {k: c for c, k in enumerate(rel["kept"])}, kit, thr, rings)
    decided = {"composite": {names[k]: {"share": v["share"], "by": {names[j]: c for j, (c, _) in v["by"].items()}}
                             for k, v in rel["composite"].items()},
               "echo": {names[k]: v for k, v in rel["echo"].items()}, "tonal": sorted(names[k] for k in tones),
               "purified": {k: v for k, v in purity.items() if v}, "pitched": {names[k]: v for k, v in sharp.items()}}
    return sc, decided


def _matched(sc: Scene, xp: np.ndarray, thresholds: dict, closest_ms: dict) -> tuple[Candidates, dict]:
    """1. Each column's matched-filter hits (its own band and attack, NCC at
    the threshold, closest_ms apart) and its weak peaks (CANDIDATE_NCC up to
    the threshold); a left-out template's hits go to the templates that
    explain it. Sets the usual gains: the median of a template's hits."""
    templates, n, columns = sc.templates, sc.n, sc.kit.columns
    cands, weak, usual = Candidates(), {}, np.ones(len(columns))
    for c, col in enumerate(columns):
        tp = templates[col.template]
        audio = tp.audio if not col.semitones else transpose(tp.audio, col.semitones)
        score, gain = sliding_ncc(xp, audio[:n[col.template]], tp.band)
        gap = max(closest_ms[tp.band], TONE_MS) if c in sc.rings else closest_ms[tp.band]
        p, props = find_peaks(score, height=CANDIDATE_NCC, distance=samples(gap))
        sure = props["peak_heights"] >= sc.thr[c]
        cands.add(c, p[sure], props["peak_heights"][sure], "matched")
        weak[c] = p[~sure]
        if not col.semitones and np.any(sure):
            usual[c] = float(np.median(gain[p[sure]]))
    sc.usual = np.array([usual[sc.col_of[col.template]] for col in columns])
    for k, v in sc.rel["composite"].items():
        tp = templates[k]
        score, _ = sliding_ncc(xp, tp.audio[:n[k]], tp.band)
        p, props = find_peaks(score, height=thresholds[tp.band], distance=samples(closest_ms[tp.band]))
        for j, (_, lag) in v["by"].items():
            new = _far(p + lag, cands.p[cands.col == sc.col_of[j]], 2)
            cands.add(sc.col_of[j], (p + lag)[new], props["peak_heights"][new], "matched")
    return cands.sorted(), weak


def _one_each(sc: Scene, cands: Candidates) -> tuple[Candidates, dict[int, tuple[int, int]]]:
    """2. One column per sound and moment (distinct); columns that keep
    losing to another one are merged into it (duplicates), their hits moved
    to it. Returns the hits and {merged column: (into, offset)}."""
    kit, columns = sc.kit, sc.kit.columns
    order = [sc.col_of[k] for k in sorted(sc.rel["kept"], key=lambda k: -sc.templates[k].hits)]
    order += [c for c in range(len(columns)) if c not in order]
    kind = [None if c.semitones else c.template in sc.tones for c in columns]  # pitched copies are never merged
    merged: dict[int, tuple[int, int]] = {}
    for _ in range(len(columns)):  # each round merges a column away: the last one merges none
        gains, _ = fit(kit, cands)
        kept, pairs = distinct(kit, cands, gains)
        dup = duplicates(cands, pairs, order, kind)
        if not dup:
            break
        for a, (b, off) in dup.items():
            merged[a] = (b, off)
            for ca, cb in _with_copies(columns, a, b):
                mine = cands.take(cands.col == ca)
                cands = cands.take(cands.col != ca)
                if cb is not None:
                    new = _far(mine.p + off, cands.p[cands.col == cb], 2)
                    cands.add(cb, (mine.p + off)[new], mine.ncc[new], "matched")
        cands = cands.sorted()
    else:
        kept, _ = distinct(kit, cands, fit(kit, cands)[0])
    return cands.take(kept), merged


def _with_copies(columns: list[Column], a: int, b: int) -> list[tuple[int, int | None]]:
    """Column a merged into b, and a's pitched copies into b's of the same
    semitones (or dropped): left behind, they reported hits under the
    merged template's name (full-1, one hit)."""
    theirs = {col.semitones: c for c, col in enumerate(columns) if col.template == columns[b].template}
    return [(a, b)] + [(c, theirs.get(col.semitones)) for c, col in enumerate(columns)
                       if col.template == columns[a].template and col.semitones]


def _significant(sc: Scene, cands: Candidates, min_gain: float) -> tuple[Candidates, np.ndarray]:
    """3. The hits at min_gain of their template's usual gain whose drop
    beats the null (what the template takes where it does not play). Sets
    the nulls; returns the hits and their gains."""
    kit, usual = sc.kit, sc.usual
    gains, _ = fit(kit, cands)
    cands = cands.take(gains >= min_gain * usual[cands.col])
    gains, drops = fit(kit, cands, drops=True)
    null = np.array([_null(kit, cands, gains, c) if not col.semitones else 0.0 for c, col in enumerate(kit.columns)])
    sc.null = np.array([null[sc.col_of[col.template]] for col in kit.columns])
    cands = cands.take((gains >= min_gain * usual[cands.col]) & (drops >= sc.null[cands.col]))
    gains, _ = fit(kit, cands)
    return cands, gains


def _root(merged: dict[int, tuple[int, int]], c: int) -> tuple[int, int]:
    """The column c's hits went to, and the offset they moved by."""
    off = 0
    while c in merged:
        c, o = merged[c]
        off += o
    return c, off


def _probes(sc: Scene, cands: Candidates, gains: np.ndarray, merged: dict, weak: dict, xp: np.ndarray,
            min_gain: float, onsets: bool) -> tuple[Candidates, dict]:
    """4. Where the kept hits leave something out: a held template at its
    offset under the hits of the one holding it (the co-detections' offset,
    else the containment lag), and each noisy template at its weak
    matched-filter peaks and the onsets of its band. Returns the probe hits
    that pass and the offsets used."""
    kit, columns, col_of = sc.kit, sc.kit.columns, sc.col_of
    layers, offsets = Candidates(), {}
    for s_tp, held in sc.rel["holds"].items():
        for h_tp, _, held_lag, _ in held:
            (s, s_off), (h, h_off) = _root(merged, col_of[s_tp]), _root(merged, col_of[h_tp])
            sp, hp = cands.p[cands.col == s], cands.p[cands.col == h]
            if s == h or not len(sp):
                continue
            lag, agree = pair_offset(sp, hp)
            lag = held_lag + h_off - s_off if lag is None else lag
            offsets[f"{columns[s].name}>{columns[h].name}"] = {"ms": round(lag / SR * 1000, 2), "pairs": agree}
            got = probe(kit, cands, gains, h, _free(sp + lag, hp), samples(SEARCH_MS))
            ok = passes(got, sc.usual[h], sc.null[h], sc.thr[h], min_gain)
            layers.add(h, got["p"][ok], got["ncc"][ok], "layer")
    for k in sc.noisy:
        c = col_of[k]
        if c in merged:
            continue
        centers = [(weak[c], SEARCH_MS)]
        if onsets:
            centers.append((band_onsets(band(xp, sc.templates[k].band)) - PRE, ONSET_MS))
        for q, ms in centers:
            got = probe(kit, cands, gains, c, _free(q, cands.p[cands.col == c]), samples(ms))
            ok = passes(got, sc.usual[c], sc.null[c], sc.thr[c], min_gain)
            layers.add(c, got["p"][ok], got["ncc"][ok], "layer")
    return layers, offsets


def _settle(sc: Scene, cands: Candidates, layers: Candidates, min_gain: float) -> tuple[Candidates, np.ndarray]:
    """5. The probe hits joined to the others (one sound per moment again,
    the hits already kept win), a low template's ringing dropped (_tails),
    and the floors checked again once everything is fit together. Returns
    the hits and their gains."""
    kit = sc.kit
    if len(layers):
        joined = cands.join(layers)
        lg, _ = fit(kit, joined)
        keep, _ = distinct(kit, joined, lg)
        mask = np.zeros(len(joined), bool)
        mask[keep] = True
        cands = joined.take(mask | np.array([s != "layer" for s in joined.source], dtype=bool))
    gains, _ = fit(kit, cands)
    cands = cands.take(_tails(cands, gains, sc.rings) & (gains > 0))
    for _ in range(2):
        gains, _ = fit(kit, cands)
        layer = np.array([s == "layer" for s in cands.source], dtype=bool)
        clear = cands.ncc >= sc.thr[cands.col]
        floor = np.where(layer & ~clear, LAYER_GAIN, min_gain) * sc.usual[cands.col]
        if np.all(gains >= floor):
            break
        cands = cands.take(gains >= floor)
    return cands, gains


def _events(templates: list, kit: Kit, cands: Candidates, gains: np.ndarray) -> dict[str, list[dict]]:
    """{name: [{t, t_attack, ncc, gain, source, under?, semitones?,
    flam_of?}]}: t is PRE after the window start (detect()'s convention),
    t_attack where the sound starts (the template's attack_s after the window
    start, sooner by as much as Simpler plays a pitched copy faster), gain
    the least-squares level in the joint fit (the velocity), ncc the matched
    filter's (a layer's: with what the other hits leave), under the template
    whose hit a layer was found under (the one overlapping it with the most
    energy)."""
    columns = kit.columns
    longest = max(c.n for c in columns)
    out: dict[str, list[dict]] = {tp.name: [] for tp in templates}
    t = np.maximum((cands.p - LEAD + PRE) / SR, 0.0)  # an attack the stem cuts off (1-5 ms) plays from 0 s
    attack = np.array([getattr(templates[c.template], "attack_s", PRE / SR) * 2 ** (-c.semitones / 12)
                       for c in columns])
    t_attack = np.maximum((cands.p - LEAD) / SR + attack[cands.col], 0.0)
    owner = np.array([columns[c].template for c in cands.col])
    for i in np.argsort(cands.p, kind="stable"):
        col = columns[cands.col[i]]
        e = {"t": round(float(t[i]), 4), "t_attack": round(float(t_attack[i]), 4), "ncc": round(float(cands.ncc[i]), 3),
             "gain": round(float(gains[i]), 4), "source": cands.source[i]}
        if cands.source[i] == "layer":
            near = np.flatnonzero((np.abs(cands.p - cands.p[i]) < longest) & (owner != col.template))
            masks = [abs(kit.inner(cands.col[i], cands.p[i], cands.col[j], cands.p[j])) * gains[j] for j in near]
            if len(near) and max(masks) > 0:
                e["under"] = columns[cands.col[near[int(np.argmax(masks))]]].name
        if col.semitones:
            e["semitones"] = col.semitones
        out[col.name].append(e)
    for evs in out.values():
        _flams(evs)
    return out


def _flams(evs: list[dict]) -> None:
    """Mark a grace note: a quieter hit of the same sound (and pitch) at
    most FLAM_MS before another, with no third hit within 2 x FLAM_MS (that
    is a roll): flam_of is the main hit's t."""
    for pitch in {e.get("semitones", 0) for e in evs}:
        mine = [e for e in evs if e.get("semitones", 0) == pitch]
        for k in range(len(mine) - 1):
            a, b = mine[k], mine[k + 1]
            before = mine[k - 1]["t"] if k else -np.inf
            after = mine[k + 2]["t"] if k + 2 < len(mine) else np.inf
            if (b["t"] - a["t"] <= FLAM_MS / 1000 and a["gain"] < b["gain"]
                    and a["t"] - before > 2 * FLAM_MS / 1000 and after - b["t"] > 2 * FLAM_MS / 1000):
                a["flam_of"] = b["t"]
