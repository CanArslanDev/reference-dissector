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

match_fine() finds noisy sounds (snare, clap, hat, ...) at any pitch: their
spectral "speckle" (speckle()) gives the file and its transposition to a
cent, and NCC at quarter-cent steps around it gives the verdict. Its index
is built once (build_speckle(): 4.6 s for 6,718 sounds, 102 MB) and kept
next to the library's .npz. pair_fit() is the layer test: two library files
fitted to a template together.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d

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


# ── Speckle: which noisy sample, and at what pitch ──
#
# The fine structure of a noisy one-shot's magnitude spectrum (the speckle of
# its noise) is fixed by the recording, and played faster or slower the whole
# spectrum moves along a log-frequency axis. On a 1-cent grid, correlating two
# speckles over +-12 semitones finds the file and its transposition at once,
# with no time alignment and no resampling. NCC needs the pitch to a cent: the
# true file's NCC halves within 1-5 cents (open hat 0.97 at 0, 0.59 at 1 cent;
# snare 0.98, 0.56 at 5 cents), so a search over whole semitones misses a
# detuned snare or hat. Calibration v2, 57 noisy sounds at the true hit times,
# each template played 0, +32.5, -47.5, -100 and +240 cents off: match_fine
# with union finds 191 of 285 (snares 65/70, open hats 70/80, hats 31/80), the
# semitone search 65; when right, the transposition is within 1.5 cents (median
# 0, p90 0.5).

SPECKLE_VERSION = 1
# per band pipeline.match searches noisy sounds in: the range whose fine structure is
# compared (Hz) and the template length (ms, detect.LENGTH_MS) both sides are read over
SPECKLE = {"snare": (800.0, 14000.0, 150), "hat": (3500.0, 16000.0, 80)}
SPECKLE_N = 1 << 16  # FFT size: 0.67 Hz bins, finer than a cent above 1.2 kHz
SPECKLE_SMOOTH = 150  # cents: coarser structure (the sound's colour, the band filter) is taken out
SPECKLE_WIDE = 1200  # cents searched either way
NOT_SPECKLED = {"kick", "808", "melodic/loop"}  # tonal library roles: no noise to compare
TRANSPOSE_MARGIN = 4096  # samples resampled past what a transposed candidate's window holds (Library.candidate)
# a file whose start peaks under this (dBFS) plays nothing a pad could use: Library.build leaves it out, and
# an index built before it lists none (Library.audible). NCC is scale-free and every entry is normalized to
# full scale, so a float WAV holding a recording 430 dB down (an fx file of the user's library) matched
# v1orig-3's hat template at NCC 0.961, and the plan put it on the hat pad: 144 hats played silence
SILENT_DB = -60.0


def speckle(x: np.ndarray, band_name: str) -> np.ndarray:
    """The speckle of a template-length window (or of each row of a block):
    dB magnitude on a 1-cent grid over SPECKLE[band_name]'s range, minus its
    150-cent moving average, z-scored. Read through a Hann window over the
    whole window, after the band's filter."""
    lo, hi, _ = SPECKLE[band_name]
    x = np.atleast_2d(np.asarray(x, np.float32))
    seg = band(x, band_name) * np.hanning(x.shape[1]).astype(np.float32)
    mag = np.abs(np.fft.rfft(seg, SPECKLE_N, axis=1))
    pos = lo * 2 ** (np.arange(int(1200 * np.log2(hi / lo)) + 1) / 1200) * SPECKLE_N / SR
    k, w = pos.astype(int), (pos - pos.astype(int)).astype(np.float32)
    db = 20 * np.log10(mag[:, k] * (1 - w) + mag[:, k + 1] * w + 1e-9)
    fine = db - uniform_filter1d(db, SPECKLE_SMOOTH, axis=1)
    fine -= fine.mean(axis=1, keepdims=True)
    return (fine / np.maximum(fine.std(axis=1, keepdims=True), 1e-9)).astype(np.float32)


def correlate(q: np.ndarray, bank: np.ndarray, wide: int = SPECKLE_WIDE,
              chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(peak, shift in cents) of speckle q against every row of bank over
    +-wide cents, through the FFT. The peak is the mean product at the best
    shift (1 for the same speckle); a positive shift means q is the row's
    sound transposed up."""
    return correlate_many([q], bank, wide, chunk)[0]


def correlate_many(qs: list[np.ndarray], bank: np.ndarray, wide: int = SPECKLE_WIDE,
                   chunk: int = 512) -> list[tuple[np.ndarray, np.ndarray]]:
    """correlate() of each speckle in qs, each block of the bank transformed
    once for all of them (the same numbers as one at a time)."""
    n_bins = bank.shape[1]
    nfft = next_fast_len(n_bins + wide, real=True)
    Qs = [np.fft.rfft(q.astype(np.float32), nfft) for q in qs]
    lags = np.r_[np.arange(0, wide + 1), np.arange(nfft - wide, nfft)]
    out = [(np.zeros(len(bank), np.float32), np.zeros(len(bank), np.int64)) for _ in qs]
    for a in range(0, len(bank), chunk):
        S = np.conj(np.fft.rfft(bank[a:a + chunk].astype(np.float32), nfft, axis=1))
        for Q, (peak, shift_cents) in zip(Qs, out):
            c = np.fft.irfft(Q[None, :] * S, nfft, axis=1)[:, lags]
            k = np.argmax(c, axis=1)
            peak[a:a + chunk] = c[np.arange(len(c)), k] / n_bins
            shift_cents[a:a + chunk] = np.where(k <= wide, k, k - len(lags))
    return out


def speckle_path(library_path: str | Path) -> Path:
    """Where a library's speckle index is kept: library.npz -> library_speckle.npz."""
    p = Path(library_path)
    return p.with_name(f"{p.stem}_speckle.npz")


def _digest(paths: list[str]) -> str:
    return hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()[:16]


def copies(a: np.ndarray, b: np.ndarray, pitched: bool, threshold: float = 0.95) -> bool:
    """Whether two library entries laid out as played (candidate()) are one
    recording: |NCC| over threshold in the full band (+-2 ms), and when
    they are played at different transpositions in the snare band too.
    Two different kicks brought to one pitch agree where their tone is, not
    above it: in a 20-beat suite run a wrong kick 2 semitones down hid the
    true one as its "copy" (0.952 full band, 0.83 in the snare band), while
    a kit's copy a semitone up agreed in both (0.997 and 0.992)."""
    if abs(ncc(a, b, samples(2))[0]) <= threshold:
        return False
    return not pitched or abs(ncc(band(a, "snare"), band(b, "snare"), samples(2))[0]) > threshold


class _Ncc:
    """|NCC| of a template against blocks of candidates of its length, both
    filtered to the role's band (band()'s response applied to the spectra),
    with the best lag within +-max_lag_ms."""

    def __init__(self, t: np.ndarray, role: str, max_lag_ms: float = 30):
        self.n = 1 << (2 * len(t) - 1).bit_length()
        self.H = response(role, self.n)
        self.T = np.fft.rfft(t, self.n) * self.H
        self.t_energy = float(energy(self.T, self.n))
        self.max_lag = samples(max_lag_ms)
        self.lags = np.r_[np.arange(0, self.max_lag + 1), np.arange(self.n - self.max_lag, self.n)]

    def __call__(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(|NCC|, lag in samples) per row of block."""
        C = self.spectra(block)
        return self.scores(np.conj(C), energy(C, self.n))

    def spectra(self, block: np.ndarray) -> np.ndarray:
        """The rows' spectra in the band: the same for every template of this
        length and role (Library._sweep shares them)."""
        return np.fft.rfft(np.atleast_2d(block), self.n, axis=1) * self.H

    def scores(self, conj: np.ndarray, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """__call__ from the rows' spectra, conjugated, and their energy e."""
        corr = np.fft.irfft(self.T[None, :] * conj, self.n, axis=1)[:, self.lags]
        v = np.abs(corr / np.sqrt(self.t_energy * e + 1e-20)[:, None])
        k = np.argmax(v, axis=1)
        return v[np.arange(len(v)), k], np.where(k <= self.max_lag, k, k - len(self.lags))


class Library:
    """The first `seconds` of each file, onset-trimmed and peak-normalized
    (a file that peaks under SILENT_DB is left out, or never listed: audible),
    plus each file's semitone-band spectrum for the pitch-invariant shortlist,
    and (once built or loaded) the speckle index: per SPECKLE band, the
    entries of noisy roles and their speckles (float16)."""

    def __init__(self, paths: list[str], roles: list[str], audio: np.ndarray, seconds: float,
                 spectra: np.ndarray | None = None, floor_db: float | None = None):
        self.paths, self.roles, self.audio, self.seconds = paths, roles, audio, seconds
        self.floor_db = floor_db  # the level floor build() kept entries over (None: an index built before it)
        self._audible: dict[int, bool] = {}
        self.index = {p: i for i, p in enumerate(paths)}
        self.spectrum_ms = 400 if seconds >= 1.0 else 100  # 808s: long enough to resolve a 30 Hz tone
        if spectra is None or len(spectra) != len(paths):
            spectra = np.stack([semitone_spectrum(a.astype(np.float32), self.spectrum_ms) for a in audio]) \
                if len(audio) else np.zeros((0, len(SEMITONE_EDGES) - 1), np.float32)
        self.spectra = spectra
        self.speckles: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # band -> (entries, speckles)
        self.speckle_info: dict = {}
        self._ready: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}  # prepare()'s searches, by template

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
            if not len(x) or np.max(np.abs(x)) < 10 ** (SILENT_DB / 20):  # silent: nothing a pad could play
                continue
            x = fit(x, n) / (np.max(np.abs(x)) or 1.0)
            paths.append(r["path"])
            kinds.append(r["role"])
            audio.append(x.astype(np.float16))
        return cls(paths, kinds, np.stack(audio) if audio else np.zeros((0, n), np.float16), seconds,
                   floor_db=SILENT_DB)

    def save(self, path: str | Path) -> None:
        np.savez(str(path), paths=np.array(self.paths), roles=np.array(self.roles), audio=self.audio,
                 seconds=np.array(self.seconds), spectra=self.spectra,
                 floor_db=np.array(np.nan if self.floor_db is None else self.floor_db))

    @classmethod
    def load(cls, path: str | Path) -> Library:
        """The library, with its speckle index when one is kept next to it
        (speckle_path) and was built from these entries by this version."""
        with np.load(str(path), allow_pickle=False) as data:  # each array read once: not copied again (0.5 GB)
            spectra = data["spectra"] if "spectra" in data.files else None
            floor = float(data["floor_db"]) if "floor_db" in data.files else np.nan
            lib = cls([str(p) for p in data["paths"]], [str(r) for r in data["roles"]],
                      data["audio"], float(data["seconds"]), spectra, None if np.isnan(floor) else floor)
        if speckle_path(path).is_file():
            lib.load_speckle(speckle_path(path))
        return lib

    def build_speckle(self, chunk: int = 256) -> dict:
        """The speckle index: every entry not of a tonal role (NOT_SPECKLED),
        laid out as a template is (candidate() at 0 semitones), per SPECKLE
        band. Returns speckle_info."""
        start = time.perf_counter()
        rows = np.array([i for i, r in enumerate(self.roles) if r not in NOT_SPECKLED], np.int32)
        self.speckles = {}
        for name, (_lo, _hi, ms) in SPECKLE.items():
            n = samples(ms)
            parts = [speckle(np.stack([self.candidate(int(i), n) for i in rows[a:a + chunk]]), name)
                     for a in range(0, len(rows), chunk)]
            bins = int(1200 * np.log2(SPECKLE[name][1] / SPECKLE[name][0])) + 1
            self.speckles[name] = (rows, np.concatenate(parts).astype(np.float16) if parts
                                   else np.zeros((0, bins), np.float16))
        self.speckle_info = {"version": SPECKLE_VERSION, "entries": int(len(rows)),
                             "build_s": round(time.perf_counter() - start, 1)}
        return self.speckle_info

    def save_speckle(self, path: str | Path) -> None:
        """The speckle index, stamped with its version, its settings and the
        library's entries it was built from."""
        np.savez(str(path), version=np.array(SPECKLE_VERSION), digest=np.array(_digest(self.paths)),
                 settings=np.array(json.dumps({"bands": SPECKLE, "n_fft": SPECKLE_N, "smooth": SPECKLE_SMOOTH})),
                 build_s=np.array(self.speckle_info.get("build_s", 0.0)),
                 rows=next(iter(self.speckles.values()))[0], **{name: s for name, (_, s) in self.speckles.items()})

    def load_speckle(self, path: str | Path) -> dict:
        """A saved speckle index, if it is this version's, with these settings,
        of these entries; otherwise none, and speckle_info says why."""
        settings = json.dumps({"bands": SPECKLE, "n_fft": SPECKLE_N, "smooth": SPECKLE_SMOOTH})
        with np.load(str(path), allow_pickle=False) as data:
            why = ("another version" if int(data["version"]) != SPECKLE_VERSION
                   else "other settings" if str(data["settings"]) != settings
                   else "another library's entries" if str(data["digest"]) != _digest(self.paths) else None)
            if why:
                self.speckles, self.speckle_info = {}, {"ignored": f"{Path(path).name}: built for {why}"}
                return self.speckle_info
            rows = data["rows"]
            self.speckles = {name: (rows, data[name]) for name in SPECKLE}
            self.speckle_info = {"version": SPECKLE_VERSION, "entries": int(len(rows)),
                                 "build_s": float(data["build_s"]), "file": Path(path).name}
        return self.speckle_info

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
        """Entry i laid out like a template: its onset PRE into the window.
        Transposed, only as much of the entry is resampled as the window
        holds, and TRANSPOSE_MARGIN samples more, after which soxr gives the
        same samples as from the whole entry (on 5,400 transpositions of 300
        entries: all the same from 1,024 on, not at 256)."""
        x = self.audio[i]
        if semitones:
            x = x[:int(np.ceil((length - PRE + samples(2)) * 2 ** (semitones / 12))) + TRANSPOSE_MARGIN]
        raw = transpose(x.astype(np.float32), semitones)
        return fit(np.r_[np.zeros(PRE - samples(2), np.float32), raw], length)

    def block(self, rows, length: int, semitones: float = 0.0) -> np.ndarray:
        """candidate() of each entry in rows, stacked; untransposed, read
        straight from the cache (the same samples, without a copy per entry)."""
        if semitones:
            return np.stack([self.candidate(i, length, semitones) for i in rows])
        pre = PRE - samples(2)
        out = np.zeros((len(rows), length), np.float32)
        m = max(0, min(length - pre, self.audio.shape[1]))
        out[:, pre:pre + m] = self.audio[rows, :m]
        return out

    def _key(self, t: np.ndarray, *settings) -> tuple:
        return (np.ascontiguousarray(t).tobytes(), np.asarray(t).dtype.str, *settings)

    def _sweep(self, ts: list[np.ndarray], role: str, base: float, roles: set[str] | None, max_lag_ms: float,
               chunk: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """_match()'s first pass for templates of one length: (|NCC|, lag) of
        every chosen entry at base semitones, each block of entries laid out
        and transformed once for all the templates."""
        chosen = [i for i, r in enumerate(self.roles) if not roles or r in roles]
        scorers = [_Ncc(t, role, max_lag_ms) for t in ts]
        out = [(np.zeros(len(chosen), np.float32), np.zeros(len(chosen), np.int64)) for _ in ts]
        for a in range(0, len(chosen), chunk):
            C = scorers[0].spectra(self.block(chosen[a:a + chunk], len(ts[0]), base))
            e, conj = energy(C, scorers[0].n), np.conj(C)
            for (scores, lags), score in zip(out, scorers):
                scores[a:a + chunk], lags[a:a + chunk] = score.scores(conj, e)
        return out

    def prepare(self, windows: dict[str, list[np.ndarray]], max_lag_ms: float = 30, chunk: int = 256) -> None:
        """match() and match_fine() made ready for several templates at once
        ({band: windows of one length}, as pipeline.match searches them): the
        first pass over every entry (_sweep) and, in a band with a speckle
        index, the speckle search (correlate_many) share each block of the
        library, which a search one template at a time transformed again for
        each one (v1-1: 34 passes over 9,000 entries for 6 templates). The
        numbers are the same; they are kept, by template, until forget()."""
        self._ready = {}
        for role, ts in windows.items():
            ts = [np.asarray(t) for t in ts if len(t)]
            if not ts:
                continue
            for length in sorted({len(t) for t in ts}):
                same = [t for t in ts if len(t) == length]
                for t, found in zip(same, self._sweep(same, role, 0.0, None, max_lag_ms, chunk)):
                    self._ready[self._key(t, "pass", role, 0.0, None, max_lag_ms, chunk)] = found
                if role in self.speckles and length == samples(SPECKLE[role][2]):
                    peaks = correlate_many([speckle(t, role)[0] for t in same], self.speckles[role][1])
                    for t, found in zip(same, peaks):
                        self._ready[self._key(t, "speckle", role, SPECKLE_WIDE)] = found

    def forget(self) -> None:
        """prepare()'s searches dropped."""
        self._ready = {}

    def match(self, t: np.ndarray, role: str, semitones=(0,), base: float = 0.0, top: int = 10,
              prefilter: int = 200, roles: set[str] | None = None, max_lag_ms: float = 30,
              chunk: int = 256) -> list[dict]:
        """Entries ranked by |NCC| with template t, both filtered to the role's
        band, over +-max_lag_ms (hit times found in a mix can sit 20 ms or
        more off where a library file starts: a quiet lead-in, an onset
        detector's peak). Everything is tried at `base` semitones; the best
        `prefilter` also at base + each of `semitones`."""
        return self._listed(self._match(t, role, semitones, base, prefilter, roles, max_lag_ms, chunk), top)

    def _match(self, t: np.ndarray, role: str, semitones, base: float, prefilter: int, roles: set[str] | None,
               max_lag_ms: float, chunk: int) -> list[dict]:
        """match()'s ranking of every chosen entry: i, ncc, lag (samples), semitones."""
        n_t = len(t)
        score = _Ncc(t, role, max_lag_ms)
        n, H, T, t_energy, max_lag, lags = score.n, score.H, score.T, score.t_energy, score.max_lag, score.lags
        chosen = [i for i, r in enumerate(self.roles) if not roles or r in roles]
        key = self._key(t, "pass", role, float(base), tuple(sorted(roles)) if roles else None, max_lag_ms, chunk)
        scores, best_lag = self._ready.get(key) or self._sweep([t], role, base, roles, max_lag_ms, chunk)[0]
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
        return results

    def audible(self, i: int) -> bool:
        """Whether entry i's file peaks at SILENT_DB or more over what build()
        read of it (its first seconds + 0.5): so for every entry of an index
        built with that floor; read once per entry listed from one built
        before it (9 ms a file: one of the 8,854 entries of the suite's
        library peaks under the floor, at -432 dBFS). A file that cannot be
        read is kept (an index searched without its files)."""
        if self.floor_db is not None and self.floor_db >= SILENT_DB:
            return True
        if i not in self._audible:
            try:
                x = read(self.paths[i], seconds=self.seconds + 0.5)
                self._audible[i] = bool(len(x)) and float(np.max(np.abs(x))) >= 10 ** (SILENT_DB / 20)
            except Exception:
                self._audible[i] = True
        return self._audible[i]

    def _listed(self, ranked: list[dict], top: int) -> list[dict]:
        """The ranking's first `top` audible entries (all of them with top 0),
        with path, role, NCC to 4 places and lag in ms."""
        out: list[dict] = []
        for r in ranked:
            if top and len(out) == top:
                break
            if self.audible(r["i"]):
                out.append(r)
        for r in out:
            r["path"], r["role"] = self.paths[r["i"]], self.roles[r["i"]]
            r["ncc"] = round(r["ncc"], 4)
            r["lag_ms"] = round(r.pop("lag") * 1000 / SR, 2)
        return out

    def match_fine(self, t: np.ndarray, role: str, keep: int = 20, top: int = 10, wide: int = SPECKLE_WIDE,
                   cents: float = 1.5, step: float = 0.25, semitones=(0,), union: int = 0,
                   max_lag_ms: float = 30) -> list[dict]:
        """Noisy sounds at any pitch (needs the speckle index): the `keep`
        entries whose speckle correlates best with t's within +-wide cents,
        and with union, the best `union` of match(t, role, semitones), are
        each tried by NCC (as match() does) at their speckle transposition
        +-cents in `step`-cent steps, and ranked by their best NCC (match()'s,
        for those it found). semitones is fractional (0.0025: a quarter cent);
        speckle is the peak. t is a template of the band's SPECKLE length
        (snare 150 ms, hat 80 ms). Not for tonal sounds: kicks and 808s
        correlate broadly over pitch, and other kicks at fractional
        transpositions fit them better."""
        if role not in self.speckles:
            raise ValueError(f"No speckle index for the {role} band (build_speckle, or index-speckle)")
        if len(t) != samples(SPECKLE[role][2]):
            raise ValueError(f"A {role}-band template is {SPECKLE[role][2]} ms here, not {len(t) * 1000 / SR:.0f}")
        rows, bank = self.speckles[role]
        peak, cents_off = self._ready.get(self._key(t, "speckle", role, wide)) or correlate(speckle(t, role)[0],
                                                                                           bank, wide)
        tried = {int(rows[j]): int(j) for j in np.argsort(-peak)[:keep]}
        found = {}
        if union:
            at = {int(i): j for j, i in enumerate(rows)}
            for r in self._match(t, role, semitones, 0.0, 200, None, max_lag_ms, 256)[:union]:
                found[r["i"]] = r
                if r["i"] in at:
                    tried.setdefault(r["i"], at[r["i"]])
        score = _Ncc(t, role, max_lag_ms)
        offsets = np.arange(-cents, cents + step / 2, step)
        for i, j in tried.items():
            semis = (cents_off[j] + offsets) / 100
            v, lag = score(np.stack([self.candidate(i, len(t), float(s)) for s in semis]))
            k = int(np.argmax(v))
            if i not in found or v[k] > found[i]["ncc"]:
                found[i] = {"i": i, "ncc": float(v[k]), "lag": int(lag[k]), "semitones": round(float(semis[k]), 4)}
            found[i]["speckle"] = round(float(peak[j]), 3)
        return self._listed(sorted(found.values(), key=lambda r: -r["ncc"]), top)

    def atoms(self, t: np.ndarray, role: str, ranked: list[dict],
              max_lag_ms: float = 30) -> tuple[np.ndarray, np.ndarray]:
        """Each ranked entry at its semitones laid out like t, filtered to the
        role's band and moved to its best lag with t, its sign taken out: the
        parts a layer fit (pair_fit) is made of. Also returns each one's lag
        in samples."""
        tb = band(t, role)
        out, lags = [], []
        for r in ranked:
            c = band(self.candidate(r["i"], len(t), r["semitones"]), role)
            v, lag = ncc(tb, c, samples(max_lag_ms))
            out.append(shift(c, lag) * (1.0 if v >= 0 else -1.0))
            lags.append(lag)
        return np.stack(out).astype(np.float64), np.array(lags)

    def same(self, i: int, j: int, length: int, threshold: float = 0.95, semitones=(0.0, 0.0)) -> bool:
        """Two entries that are one recording (kits copy each other's sounds),
        each played at its semitones: a kit's copy at another pitch too
        (copies())."""
        if i == j:
            return True
        a, b = self.candidate(i, length, semitones[0]), self.candidate(j, length, semitones[1])
        return copies(a, b, semitones[0] != semitones[1], threshold)

    def distinct(self, ranked: list[dict], length: int, keep: int = 5) -> list[dict]:
        """The ranking with copies of a higher-ranked sound removed (copies(),
        each entry played at the transposition it was found at), also a
        kit's copy at another pitch: the fine search finds them, and an open
        hat's copy 0.22 semitones down tied the original at NCC 0.84 and
        left the verdict no margin."""
        out: list[dict] = []
        laid: list[np.ndarray] = []  # each kept entry as played, laid out once (same() would redo it per pair)
        for r in ranked:
            c = self.candidate(r["i"], length, r["semitones"])
            if any(r["i"] == o["i"] or copies(c, x, r["semitones"] != o["semitones"]) for o, x in zip(out, laid)):
                continue
            out.append(r)
            laid.append(c)
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


def pair_fit(target: np.ndarray, atoms: np.ndarray, first: int = 0) -> dict:
    """Least squares of target over atoms[first] and each other atom in turn,
    both gains positive (a pair whose fit wants a gain <= 0 is no layer):
    residual_db per partner (dB re the target; inf where a gain is not
    positive), the gains (first's, partner's) and single_db, the residual of
    atoms[first] alone. Closed form, so 60 partners take microseconds."""
    target, atoms = np.asarray(target, np.float64), np.asarray(atoms, np.float64)
    a = atoms[first]
    e, g_aa, r_a = float(target @ target) or 1e-12, float(a @ a), float(a @ target)
    g_ab, g_bb, r_b = atoms @ a, np.einsum("ij,ij->i", atoms, atoms), atoms @ target
    det = g_aa * g_bb - g_ab ** 2
    safe = np.where(np.abs(det) > 1e-9 * g_aa * g_bb, det, np.inf)
    gain_a, gain_b = (g_bb * r_a - g_ab * r_b) / safe, (g_aa * r_b - g_ab * r_a) / safe
    ok = np.isfinite(safe) & (gain_a > 0) & (gain_b > 0)
    ok[first] = False
    residual = np.maximum(e - gain_a * r_a - gain_b * r_b, 1e-12 * e)
    single = max(e - r_a ** 2 / g_aa, 1e-12 * e) if g_aa > 0 and r_a > 0 else e
    return {"residual_db": np.where(ok, 10 * np.log10(residual / e), np.inf), "gains": (gain_a, gain_b),
            "single_db": float(10 * np.log10(single / e))}


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
