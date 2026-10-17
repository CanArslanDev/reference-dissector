"""Comparing a rebuild with its reference the same way on both sides.

A reference is only ever seen through Demucs (and, for notes, Basic Pitch);
a rebuild's own clean tracks are not. Compared across that gap the numbers
measure the separation: clean HAYIR DEMELIYDIM stems against separated ones
scored 0.625 on a style measure where the song against its own separated
instrumental scored 0.95. So both mixes go through the same separation, and
every measure runs on both sets of stems (`compare`). The reference's halves
against each other give the spread a measure has within one song
(`baseline`): a difference smaller than that is not a difference.

The measures are the ones that separated a flat rebuild from the song where
note statistics could not: loudness, octave balance, width per octave, onset
density, how the 808 and the chords fall after each kick, and each drum
role's level re the loudest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR
from .groove import band_envelope, curve_distance, hit_levels, kick_curve, role_offsets
from .loudness import integrated_lufs
from .space import band_correlation

OCTAVES = (63, 125, 250, 500, 1000, 2000, 4000, 8000)
ROLE_BANDS = {"kick": (30, 150), "snare": (1000, 6000), "hat": (5000, 14000)}


def load_stems(folder: str | Path, span: tuple[float, float] | None = None) -> dict[str, np.ndarray]:
    """Every stem in a folder as float (n, 2) at the file's rate (resampled to SR),
    cut to span (seconds) when given."""
    import soxr
    out = {}
    for f in sorted(Path(folder).glob("*.wav")):
        x, sr = sf.read(f, always_2d=True)
        if sr != SR:
            x = soxr.resample(x, sr, SR)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)
        if span:
            x = x[int(span[0] * SR):int(span[1] * SR)]
        out[f.stem] = x[:, :2].astype(np.float64)
    return out


def octave_balance(x: np.ndarray) -> dict[str, float]:
    """Energy per octave band in dB re the total."""
    import scipy.signal as ss
    m = x.mean(axis=1) if x.ndim == 2 else x
    f, p = ss.welch(m, fs=SR, nperseg=8192)
    total = float(p.sum()) or 1e-20
    out = {}
    for c in OCTAVES:
        share = float(p[(f >= c / np.sqrt(2)) & (f < c * np.sqrt(2))].sum()) / total
        out[str(c)] = round(10 * np.log10(share + 1e-12), 1)
    return out


def onset_rate(x: np.ndarray) -> float:
    """Onsets per second (librosa's detector on the mono sum)."""
    import librosa
    m = (x.mean(axis=1) if x.ndim == 2 else x).astype(np.float32)
    on = librosa.onset.onset_detect(y=m, sr=SR, units="time")
    return round(len(on) / (len(m) / SR), 2) if len(m) else 0.0


def band_onsets(drums: np.ndarray, lo: float, hi: float, min_gap: float = 0.06, rel_db: float = -12.0,
                window_ms: float = 3.0) -> list[float]:
    """Hits in one band of the drum stem: where its envelope rises fastest, at
    least min_gap apart, and peaking within rel_db of the band's loud hits (its
    99th percentile), so bleed and tails are left out."""
    import scipy.signal as ss
    m = drums.mean(axis=1) if drums.ndim == 2 else drums
    env = band_envelope(m, lo, hi, window_ms)
    loud = float(np.percentile(env, 99.5)) or 1e-12
    rise = np.diff(env, prepend=env[0])
    peaks, _ = ss.find_peaks(rise, height=float(np.percentile(rise, 99.5)) * 0.2, distance=int(min_gap * SR))
    w = int(0.03 * SR)
    return [p / SR for p in peaks if env[p:p + w].max(initial=0) >= loud * 10 ** (rel_db / 20)]


def kick_times(drums: np.ndarray, min_gap: float = 0.2) -> list[float]:
    """Kick onsets: hits in the drum stem's 30-150 Hz band. A kick's own body
    retriggers ~0.11 s in (one 16th at 135 BPM); kicks that close are rare,
    so the default gap is 0.2 s."""
    return band_onsets(drums, 30, 150, min_gap, window_ms=5.0)


def measure(stems: dict[str, np.ndarray], kick_gap: float = 0.2) -> dict:
    """Every measure on one set of stems (drums, bass, other, ...).

    Kicks come from the separated drum stem, which can hold other attacks: on
    HAYIR DEMELIYDIM Demucs put the 808's entry, two 16ths after each kick,
    into the drums, and it counted as a second kick until kick_gap was raised
    past it (0.3 s there)."""
    out: dict = {"stems": {}}
    for name, x in stems.items():
        lv = integrated_lufs(x, SR)
        out["stems"][name] = {"lufs": round(lv, 1) if np.isfinite(lv) else None,
                              "octaves_db": octave_balance(x),
                              "width_r": {k: v["r"] for k, v in band_correlation(x, centres=OCTAVES[1:7]).items()},
                              "onsets_per_s": onset_rate(x)}
    drums = stems.get("drums")
    if drums is not None:
        kicks = kick_times(drums, kick_gap)
        out["kicks"] = len(kicks)
        grooves = {}
        for layer, (stem, band) in {"808": ("bass", (30, 150)), "chords": ("other", (150, 2000))}.items():
            if stem in stems and len(kicks) >= 4:
                try:
                    grooves[layer] = kick_curve(band_envelope(stems[stem].mean(axis=1), *band), kicks,
                                                reference="median")
                except ValueError:
                    pass
        out["ducking"] = grooves
        levels = {role: hit_levels(drums.mean(axis=1), band_onsets(drums, *b), *b) for role, b in ROLE_BANDS.items()}
        out["drum_roles_db"] = role_offsets({k: v for k, v in levels.items() if v["hits"]})
    return out


def _flat(d: dict, prefix: str = "") -> dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        elif isinstance(v, int | float) and not isinstance(v, bool):
            out[key] = float(v)
    return out


def compare(reference: dict, rebuild: dict, spread: dict | None = None) -> dict:
    """Measure by measure: reference, rebuild, difference, and whether the
    difference exceeds the reference's own spread (from `baseline`) when given."""
    a, b = _flat(reference), _flat(rebuild)
    rows = {}
    for k in sorted(set(a) & set(b)):
        row = {"reference": round(a[k], 3), "rebuild": round(b[k], 3), "diff": round(b[k] - a[k], 3)}
        if spread and k in spread:
            row["beyond_spread"] = abs(b[k] - a[k]) > spread[k]
        rows[k] = row
    if "ducking" in reference and "ducking" in rebuild:
        for layer in set(reference["ducking"]) & set(rebuild["ducking"]):
            rows[f"ducking.{layer}.curve_distance_db"] = {
                "diff": curve_distance(reference["ducking"][layer], rebuild["ducking"][layer])}
    return rows


def baseline(stems: dict[str, np.ndarray], kick_gap: float = 0.2) -> dict[str, float]:
    """How far each measure moves between the two halves of the same song."""
    n = min(len(x) for x in stems.values()) // 2
    first = measure({k: x[:n] for k, x in stems.items()}, kick_gap)
    second = measure({k: x[n:2 * n] for k, x in stems.items()}, kick_gap)
    a, b = _flat(first), _flat(second)
    return {k: round(abs(a[k] - b[k]), 3) for k in set(a) & set(b)}
