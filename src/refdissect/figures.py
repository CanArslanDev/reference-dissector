"""Pictures a model (or a person) can read: spectrograms side by side with a
difference map, and the calibration's correct/wrong matches by score."""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .audio import SR  # noqa: E402


def _spec(x: np.ndarray, n_fft: int) -> np.ndarray:
    x = x / (np.max(np.abs(x)) or 1.0)
    return librosa.amplitude_to_db(np.abs(librosa.stft(x, n_fft=n_fft, hop_length=64)), ref=np.max, top_db=70)


def spectrogram_rows(rows: list[dict], out: str | Path, title: str | None = None) -> Path:
    """rows: [{"title", "cells": [(label, audio)], "hz": (lo, hi), "ms": length,
    "difference": bool}]; with difference, a third panel shows cell 2 minus
    cell 1 (red: the second is louder there)."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        lo, hi = row["hz"]
        n_fft = 8192 if hi <= 500 else 1024
        specs = []
        for c in range(3):
            ax = axes[r, c]
            if c >= len(row["cells"]):
                ax.axis("off")
                continue
            label, x = row["cells"][c]
            S = _spec(x[: int(row["ms"] * SR / 1000)], n_fft)
            specs.append(S)
            librosa.display.specshow(S, sr=SR, hop_length=64, x_axis="ms",
                                     y_axis="log" if lo >= 100 else "linear", ax=ax, cmap="magma", vmin=-70, vmax=0)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{row['title']}: {label}", fontsize=9)
        if row.get("difference") and len(specs) >= 2:
            ax = axes[r, 2]
            ax.axis("on")
            n = min(specs[0].shape[1], specs[1].shape[1])
            librosa.display.specshow(specs[1][:, :n] - specs[0][:, :n], sr=SR, hop_length=64, x_axis="ms",
                                     y_axis="log" if lo >= 100 else "linear", ax=ax, cmap="coolwarm",
                                     vmin=-30, vmax=30)
            ax.set_ylim(lo, hi)
            ax.set_title(f"{row['title']}: difference (red = match louder)", fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def calibration_scatter(results: list[dict], out: str | Path) -> Path:
    """Top match's NCC against its margin over the next different sound, for
    the oracle runs: correct identifications vs wrong ones, tonal vs noisy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for tonal, marker in ((False, "o"), (True, "^")):
        for correct, color in ((True, "#2a9d8f"), (False, "#e76f51")):
            pts = [(r["ncc_top"], r["margin"]) for r in results
                   if (r["sound"] in ("kick", "808")) == tonal and r["correct"] == correct]
            if pts:
                xs, ys = zip(*pts)
                kind = "kick/808" if tonal else "snare/hat/open hat"
                ax.scatter(xs, ys, c=color, marker=marker, s=50,
                           label=f"{kind}: {'right' if correct else 'wrong'} top match")
    ax.set_xlabel("NCC of the top library match")
    ax.set_ylabel("margin over the next different sound")
    ax.set_title("Identification at known hit times (synthetic beats)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
