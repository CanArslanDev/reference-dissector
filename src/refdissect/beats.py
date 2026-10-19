"""Beats and downbeats from beat_this (Foscarin, Schlüter and Widmer, ISMIR
2024; MIT code and weights), in an environment uv builds on first use, the
way separate.py runs Demucs: torch never loads into this process.

Pinned to beat-this 1.1.0 from PyPI, whose package source is the same as
GitHub CPJKU/beat_this at b95c8ab. Its final0 checkpoint (81 MB) downloads
on first use into torch's hub cache, ~/.cache/torch/hub/checkpoints/
beat_this-final0.ckpt. On an Apple M4 (MPS) it took 1.1 s for PIRAMIT's
126 s, loading included; a call takes 1.3-2.8 s once uv has built the
environment (12 s the first time here, with the wheels already cached).

It reads the sum of the stems: on fixed-grid test beats its downbeats were
right more often on mixes than on drum buses (F 0.94 against 0.82). On 26
constant-tempo test mixes its tempo was the programmed one in 25 (one at
half); it read PIRAMIT's drums at 68.501 BPM, half the 137 its grid is
programmed at. grid.estimate turns its output into a grid and decides the
octave. madmom's models were left out: they are CC BY-NC-SA.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKER = Path(__file__).with_name("workers") / "beats_worker.py"
VERSION = "1.1.0"
PACKAGES = (f"beat-this=={VERSION}", "soundfile>=0.12")
# Demucs' stem names: summed, they are the mix the tracker is best on
STEMS = ("drums", "bass", "other", "vocals", "piano", "guitar")


def command(audio: list[str | Path], device: str | None = None, checkpoint: str = "final0") -> list[str]:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is needed to run beat_this in its own environment: https://docs.astral.sh/uv/")
    cmd = [uv, "run", "--isolated", "--no-project", "--python", "3.12"]
    for p in PACKAGES:
        cmd += ["--with", p]
    cmd += ["python", str(WORKER), *[str(Path(a).expanduser()) for a in audio], "--checkpoint", checkpoint]
    return cmd + (["--device", device] if device else [])


def track(audio: list[str | Path], device: str | None = None, checkpoint: str = "final0") -> dict:
    """Beats and downbeats (s) of the files summed to mono:
    {beats, downbeats, name, version, checkpoint, device, torch, duration_s, seconds}."""
    done = subprocess.run(command(audio, device, checkpoint), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"beat_this failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def track_stems(stems_dir: str | Path, log=print) -> dict | None:
    """track() on the sum of the stems, or None when it cannot run (no uv, no
    network for the first download, a worker error): the grid is then fitted
    without it, as before."""
    paths = [Path(stems_dir) / f"{s}.wav" for s in STEMS if (Path(stems_dir) / f"{s}.wav").is_file()]
    try:
        out = track(paths)
    except Exception as e:
        reason = str(e).strip().splitlines()[-1] if str(e).strip() else type(e).__name__
        log(f"beat tracker unavailable ({reason}); fitting the grid without it")
        return None
    log(f"beat_this {out.get('version')} ({out.get('checkpoint')}, {out.get('device')}): {len(out['beats'])} beats, "
        f"{len(out['downbeats'])} downbeats in {out.get('seconds')} s")
    return out
