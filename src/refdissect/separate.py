"""Stem separation with Demucs, in an environment uv builds on first use
(torch and the model: a few hundred MB, cached after that).

Runs are seeded (seed 0 and one shift by default), so a file separates to the
same stems every time on one device. Two runs of a 30 s v1 beat gave
byte-identical stems on MPS with htdemucs and with htdemucs_ft; seeds 0 and 1
gave drums 20.0 and 20.3 dB SDR apart. separation.json next to the stems says
how they were made (model, seed, shifts, device, package versions, the
input's hash)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections import deque
from pathlib import Path

WORKER = Path(__file__).with_name("workers") / "separate_worker.py"
PACKAGES = ("demucs==4.1.0", "soundfile>=0.12", "scipy>=1.11")
DEFAULT_MODEL = "htdemucs"


def command(jobs: list[tuple[str | Path, str | Path]], model: str = DEFAULT_MODEL, device: str | None = None,
            seed: int = 0, shifts: int = 1) -> list[str]:
    """The worker's command line for (audio, out_dir) jobs."""
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is needed to run Demucs in its own environment: https://docs.astral.sh/uv/")
    cmd = [uv, "run", "--isolated", "--no-project", "--python", "3.12"]
    for p in PACKAGES:
        cmd += ["--with", p]
    cmd += ["python", str(WORKER), model, device or "auto", str(int(seed)), str(int(shifts))]
    for audio, out_dir in jobs:
        cmd += [str(Path(audio).expanduser()), str(Path(out_dir).expanduser())]
    return cmd


def separate_many(jobs: list[tuple[str | Path, str | Path]], model: str = DEFAULT_MODEL, device: str | None = None,
                  seed: int = 0, shifts: int = 1, log=None) -> list[dict]:
    """separate() for several files with one model load. Every job is seeded
    alike, so its stems are the ones separate() gives it alone (byte for byte
    on a v1 beat, MPS). log(info) is called as each job finishes."""
    if not jobs:
        return []
    proc = subprocess.Popen(command(jobs, model, device, seed, shifts), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    done, tail = [], deque(maxlen=40)
    for line in proc.stdout:
        if line.startswith("{") and '"stems"' in line:
            done.append(json.loads(line))
            if log:
                log(done[-1])
        else:
            tail.append(line.rstrip())
    if proc.wait() != 0 or len(done) != len(jobs):
        raise RuntimeError("Demucs failed:\n" + "\n".join(tail)[-2000:])
    return done


def separate(audio: str | Path, out_dir: str | Path, model: str = DEFAULT_MODEL, device: str | None = None,
             seed: int = 0, shifts: int = 1) -> dict:
    """Stems of audio in out_dir (drums, bass, other, vocals[, guitar, piano])
    and what separation.json records about them."""
    return separate_many([(audio, out_dir)], model, device, seed, shifts)[0]
