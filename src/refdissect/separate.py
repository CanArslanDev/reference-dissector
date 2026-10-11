"""Stem separation with Demucs, in an environment uv builds on first use
(torch and the model: a few hundred MB, cached after that)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

WORKER = Path(__file__).with_name("workers") / "separate_worker.py"
PACKAGES = ("demucs==4.1.0", "soundfile>=0.12", "scipy>=1.11")


def command(audio: str | Path, out_dir: str | Path, model: str = "htdemucs",
            device: str | None = None) -> list[str]:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is needed to run Demucs in its own environment: https://docs.astral.sh/uv/")
    cmd = [uv, "run", "--isolated", "--no-project", "--python", "3.12"]
    for p in PACKAGES:
        cmd += ["--with", p]
    cmd += ["python", str(WORKER), str(Path(audio).expanduser()), str(Path(out_dir).expanduser()), model]
    return cmd + ([device] if device else [])


def separate(audio: str | Path, out_dir: str | Path, model: str = "htdemucs",
             device: str | None = None) -> dict:
    """Stems of audio in out_dir (drums, bass, other, vocals[, guitar, piano])."""
    done = subprocess.run(command(audio, out_dir, model, device), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"Demucs failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])
