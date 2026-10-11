"""Your sample library as a searchable index, and the search.

inventory() walks folders and labels each file with a drum role from its
path (names are all a kit gives you; good enough to count, not to trust).
Library caches the first `seconds` of every one-shot (onset-trimmed,
peak-normalized) in one .npz, so a search over ~9,000 sounds takes 1-3 s.

match() ranks the library against a template by normalized cross-
correlation after filtering both to the role's band, over small lags and
(on the best `prefilter`) other transpositions: the waveform match that
tells "the same recording" from "a similar one". Margin over the next
*different* sound is what makes a verdict; see verdict() and
docs/04-identifying-samples.md.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from .audio import PRE, SR, band, fit, read, response, samples, shift, transpose, trim_onset
from .xcorr import energy, ncc

AUDIO = {".wav", ".aif", ".aiff", ".flac", ".mp3"}
# First match wins; role words before "808" so an "808 snare" is a snare
ROLES = [
    ("open_hat", r"open ?h|\boh\b|o-?hat|openhat|open hi"),
    ("hat", r"hi ?-?hat|\bhh\b|\bhats?\b|closed|\bch\b|hihat"),
    ("clap", r"clap"),
    ("snare", r"snare|\bsnr\b|\bsd\b|rimshot"),
    ("rim", r"\brim"),
    ("kick", r"kick|\bkck\b|\bbd\b|bassdrum"),
    ("crash/ride", r"crash|ride|cymbal|splash"),
    ("perc", r"perc|shaker|tamb|conga|bongo|tom\b|toms|cowbell|block|bell"),
    ("fx", r"\bfx\b|riser|sweep|impact|transition|reverse|uplifter|downlifter|noise"),
    ("vox", r"vox|vocal|acapella|adlib|chant|tag"),
    ("808", r"808|sub ?bass|\bsubs?\b"),
    ("melodic/loop", r"loop|melod|stem|bpm|guitar|piano|keys|pad|synth|chord|bass"),
]


def role_of(path: str) -> str:
    p = path.lower()
    for name, pattern in ROLES:
        if re.search(pattern, p):
            return name
    return "other"


# Folders left out: generated sounds (ours, they would match themselves) and Live's backups
SKIP_DIRS = {"AbletonMCP Generated", "Backup"}


def inventory(folders: list[str | Path], skip: set[str] = SKIP_DIRS) -> list[dict]:
    """Every audio file under the folders: path, role (from the path below the
    folder), duration in seconds (None if unreadable)."""
    import soundfile as sf
    rows = []
    for folder in folders:
        folder = Path(folder).expanduser()
        for root, dirs, files in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in skip)
            for name in sorted(files):
                if name.startswith(".") or Path(name).suffix.lower() not in AUDIO:
                    continue
                path = Path(root) / name
                try:
                    info = sf.info(str(path))
                    duration = info.frames / info.samplerate
                except Exception:
                    duration = None
                rows.append({"path": str(path), "role": role_of(str(path.relative_to(folder))),
                             "duration": duration, "source": folder.name})
    return rows


def summarize(rows: list[dict], one_shot_seconds: float = 4.0) -> dict:
    """Counts per source and role: one-shots and longer files."""
    out: dict[str, dict[str, list[int]]] = {}
    for r in rows:
        counts = out.setdefault(r["source"], {}).setdefault(r["role"], [0, 0])
        counts[0 if r["duration"] is not None and r["duration"] <= one_shot_seconds else 1] += 1
    return out


SEMITONE_EDGES = 40.0 * 2 ** (np.arange(0, 110) / 12)  # 40 Hz .. 22 kHz, one band per semitone


def semitone_spectrum(x: np.ndarray, ms: float = 100, n_fft: int = 16384) -> np.ndarray:
    """Log energy per semitone band of the first `ms` (z-scored). Transposing
    a sound by k semitones moves this by k bands, so sounds can be compared
    at every transposition at once (see Library.shortlist)."""
    seg = x[: samples(ms)]
    seg = seg * np.hanning(len(seg))
    p = np.abs(np.fft.rfft(seg, n_fft)) ** 2
    f = np.fft.rfftfreq(n_fft, 1 / SR)
    idx = np.searchsorted(SEMITONE_EDGES, f) - 1
    ok = (idx >= 0) & (idx < len(SEMITONE_EDGES) - 1)
    bands = np.bincount(idx[ok], weights=p[ok], minlength=len(SEMITONE_EDGES) - 1)
    v = np.log10(bands + 1e-10)
    return ((v - v.mean()) / (v.std() or 1.0)).astype(np.float32)


class Library:
    """The first `seconds` of each file, onset-trimmed and peak-normalized,
    plus each file's semitone-band spectrum for the pitch-invariant shortlist."""

    def __init__(self, paths: list[str], roles: list[str], audio: np.ndarray, seconds: float,
                 spectra: np.ndarray | None = None):
        self.paths, self.roles, self.audio, self.seconds = paths, roles, audio, seconds
        self.index = {p: i for i, p in enumerate(paths)}
        self.spectrum_ms = 400 if seconds >= 1.0 else 100  # 808s: long enough to resolve a 30 Hz tone
        if spectra is None or len(spectra) != len(paths):
            spectra = np.stack([semitone_spectrum(a.astype(np.float32), self.spectrum_ms) for a in audio]) \
                if len(audio) else np.zeros((0, len(SEMITONE_EDGES) - 1), np.float32)
        self.spectra = spectra

    @classmethod
    def build(cls, rows: list[dict], seconds: float = 0.6, max_duration: float = 4.0,
              roles: set[str] | None = None, exclude: set[str] = frozenset({"vox"})) -> Library:
        n = int(seconds * SR)
        paths, kinds, audio = [], [], []
        for r in rows:
            if r["duration"] is None or r["duration"] > max_duration or r["role"] in exclude:
                continue
            if roles and r["role"] not in roles:
                continue
            try:
                x = trim_onset(read(r["path"], seconds=seconds + 0.5))
            except Exception:
                continue
            if not len(x) or not np.any(x):
                continue
            x = fit(x, n) / (np.max(np.abs(x)) or 1.0)
            paths.append(r["path"])
            kinds.append(r["role"])
            audio.append(x.astype(np.float16))
        return cls(paths, kinds, np.stack(audio) if audio else np.zeros((0, n), np.float16), seconds)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), paths=np.array(self.paths), roles=np.array(self.roles), audio=self.audio,
                 seconds=np.array(self.seconds), spectra=self.spectra)

    @classmethod
    def load(cls, path: str | Path) -> Library:
        with np.load(str(path), allow_pickle=False) as data:
            spectra = np.array(data["spectra"]) if "spectra" in data.files else None
            return cls([str(p) for p in data["paths"]], [str(r) for r in data["roles"]],
                       np.array(data["audio"]), float(data["seconds"]), spectra)

    def shortlist(self, t: np.ndarray, semitones, chosen: list[int], keep: int) -> dict[int, list[float]]:
        """The `keep` entries whose spectrum is most like t's at some
        transposition, with the transpositions to try (best and neighbours)."""
        semis = [int(round(s)) for s in semitones]
        T = semitone_spectrum(t, self.spectrum_ms)
        C = self.spectra[chosen]
        n = len(T)
        best = np.full(len(chosen), -np.inf, np.float32)
        best_s = np.zeros(len(chosen), np.int64)
        for k in semis:
            # a sound transposed up by k has its band b at b + k
            lo, hi = max(0, k), min(n, n + k)
            a, c = T[lo:hi], C[:, lo - k:hi - k]
            sim = (c @ a) / (np.linalg.norm(c, axis=1) * (np.linalg.norm(a) or 1.0) + 1e-9)
            better = sim > best
            best[better], best_s[better] = sim[better], k
        top = np.argsort(-best)[:keep]
        return {chosen[j]: sorted({float(s) for s in (best_s[j] - 1, best_s[j], best_s[j] + 1) if s in semis})
                for j in top}

    def __len__(self) -> int:
        return len(self.paths)

    def candidate(self, i: int, length: int, semitones: float = 0.0) -> np.ndarray:
        """Entry i laid out like a template: its onset PRE into the window."""
        raw = transpose(self.audio[i].astype(np.float32), semitones)
        return fit(np.r_[np.zeros(PRE - samples(2), np.float32), raw], length)

    def match(self, t: np.ndarray, role: str, semitones=(0,), base: float = 0.0, top: int = 10,
              prefilter: int = 200, roles: set[str] | None = None, max_lag_ms: float = 30,
              chunk: int = 256) -> list[dict]:
        """Entries ranked by |NCC| with template t, both filtered to the role's
        band, over +-max_lag_ms (hit times found in a mix can sit 20 ms or
        more off where a library file starts: a quiet lead-in, an onset
        detector's peak). Everything is tried at `base` semitones; the best
        `prefilter` also at base + each of `semitones`."""
        n_t = len(t)
        n = 1 << (2 * n_t - 1).bit_length()
        H = response(role, n)
        T = np.fft.rfft(t, n) * H
        t_energy = float(energy(T, n))
        max_lag = samples(max_lag_ms)
        lags = np.r_[np.arange(0, max_lag + 1), np.arange(n - max_lag, n)]
        chosen = [i for i, r in enumerate(self.roles) if not roles or r in roles]
        scores = np.zeros(len(chosen), np.float32)
        best_lag = np.zeros(len(chosen), np.int64)
        for a in range(0, len(chosen), chunk):
            idx = chosen[a:a + chunk]
            block = np.stack([self.candidate(i, n_t, base) for i in idx])
            C = np.fft.rfft(block, n, axis=1) * H
            corr = np.fft.irfft(T[None, :] * np.conj(C), n, axis=1)[:, lags]
            v = corr / np.sqrt(t_energy * energy(C, n) + 1e-20)[:, None]
            k = np.argmax(np.abs(v), axis=1)
            scores[a:a + chunk] = np.abs(v[np.arange(len(idx)), k])
            best_lag[a:a + chunk] = np.where(k <= max_lag, k, k - len(lags))
        results = [{"i": i, "ncc": float(scores[j]), "lag": int(best_lag[j]), "semitones": float(base)}
                   for j, i in enumerate(chosen)]
        results.sort(key=lambda r: -r["ncc"])
        if any(semitones):
            tries = {r["i"]: [float(s) for s in semitones if s] for r in results[:prefilter // 2]}
            for i, shifts in self.shortlist(t, semitones, chosen, prefilter).items():
                tries.setdefault(i, [])
                tries[i] = sorted(set(tries[i]) | {s for s in shifts if s})
        else:
            tries = {}
        by_i = {r["i"]: r for r in results}
        for i, shifts in tries.items():
            r = by_i[i]
            for s in shifts:
                C = np.fft.rfft(self.candidate(r["i"], n_t, base + s), n) * H
                v = np.fft.irfft(T * np.conj(C), n)[lags] / np.sqrt(t_energy * float(energy(C, n)) + 1e-20)
                k = int(np.argmax(np.abs(v)))
                if abs(v[k]) > r["ncc"]:
                    r.update(ncc=float(abs(v[k])), lag=k if k <= max_lag else k - len(lags),
                             semitones=float(base + s))
        results.sort(key=lambda r: -r["ncc"])
        out = results[:top] if top else results
        for r in out:
            r["path"], r["role"] = self.paths[r["i"]], self.roles[r["i"]]
            r["ncc"] = round(r["ncc"], 4)
            r["lag_ms"] = round(r.pop("lag") * 1000 / SR, 2)
        return out

    def same(self, i: int, j: int, length: int, threshold: float = 0.95) -> bool:
        """Two entries that are one recording (kits copy each other's sounds)."""
        if i == j:
            return True
        a, b = self.candidate(i, length), self.candidate(j, length)
        return abs(ncc(a, b, samples(2))[0]) > threshold

    def distinct(self, ranked: list[dict], length: int, keep: int = 5) -> list[dict]:
        """The ranking with copies of a higher-ranked sound removed."""
        out: list[dict] = []
        for r in ranked:
            if any(self.same(r["i"], o["i"], length) for o in out):
                continue
            out.append(r)
            if len(out) == keep:
                break
        return out


def per_hit_votes(x: np.ndarray, members, candidates: dict[str, np.ndarray], role: str,
                  head_ms: float = 70, max_lag_ms: float = 30) -> dict:
    """For every hit on its own: which candidate matches it best (over the
    first head_ms). A real match wins nearly every hit, not just the average."""
    head = PRE + samples(head_ms)
    cands = {k: band(c, role)[:head] for k, c in candidates.items()}
    wins = dict.fromkeys(candidates, 0)
    scores: dict[str, list[float]] = {k: [] for k in candidates}
    for s, lag, _sign, _gain, _ in members:
        a = s - PRE - lag
        if a - 200 < 0 or a + head + 200 > len(x):
            continue
        w = band(x[a - 200:a + head + 200], role)[200:-200]
        per = {k: abs(ncc(w, c, samples(max_lag_ms))[0]) for k, c in cands.items()}
        for k, v in per.items():
            scores[k].append(v)
        wins[max(per, key=per.get)] += 1
    return {"wins": wins, "median_ncc": {k: round(float(np.median(v)), 3) if v else None
                                          for k, v in scores.items()}}


def null_test(t: np.ndarray, c: np.ndarray, role: str, max_lag_ms: float = 30) -> float:
    """What is left of t after subtracting c (aligned, best gain), in dB of t.
    -inf for the same file; around -6 dB is what NCC 0.86 leaves."""
    tb, cb = band(t, role), band(c, role)
    _, lag = ncc(tb, cb, samples(max_lag_ms))
    aligned = shift(cb, lag)
    gain = float(np.dot(tb, aligned) / (np.dot(aligned, aligned) or 1.0))
    residual = tb - gain * aligned
    return round(float(10 * np.log10(np.sum(residual ** 2) / (np.sum(tb ** 2) or 1.0) + 1e-12)), 1)


# Verdict thresholds, from the calibration in docs/03-calibration.md (6 synthetic
# beats, 30 sounds): noisy sounds (snare, clap, hat) separate clearly; tonal ones
# (kick, 808) correlate highly with any similar sound, so their margins stay small.
TONAL = {"kick", "808"}  # roles; perc and others follow the noisy rules


def verdict(role: str, ncc_top: float, margin: float, wins_share: float | None = None) -> str:
    if role in TONAL:
        if ncc_top >= 0.97 and margin >= 0.05:
            return "same sample (likely)"
        return "same family" if ncc_top >= 0.9 else "no close match"
    if ncc_top >= 0.95 and margin >= 0.03:
        return "same sample"
    if margin >= 0.05 and (wins_share is None or wins_share >= 0.8) and ncc_top >= 0.8:
        return "candidate: maybe the same sample, processed (check by ear)"
    if ncc_top >= 0.85:
        return "same family"
    return "no close match"


def load_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
