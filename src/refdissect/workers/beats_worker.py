"""Beats and downbeats with beat_this, run in its own uv environment (torch
stays out of this package's). Standalone on purpose: imports nothing from
refdissect.

usage: beats_worker.py AUDIO [AUDIO ...] [--device DEVICE] [--checkpoint NAME]
The files are summed to mono first (stems sum to the mix, which is what the
tracker was trained on). Prints one JSON line: beat and downbeat times in
seconds, and what ran (version, checkpoint, device).
"""

import argparse
import json
import sys
import time
from importlib.metadata import version


def mono_sum(paths: list[str]):
    """The files averaged to mono and summed, at their (common) rate."""
    import numpy as np
    import soundfile as sf

    total, rate = None, None
    for path in paths:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if rate is not None and sr != rate:
            raise SystemExit(f"{path} is at {sr} Hz, the others at {rate} Hz")
        rate = sr
        if total is None:
            total = x
        else:
            n = max(len(total), len(x))
            total = np.pad(total, (0, n - len(total))) + np.pad(x, (0, n - len(x)))
    return total, rate


def main() -> int:
    import torch
    from beat_this.inference import Audio2Beats

    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+")
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default="final0")
    a = p.parse_args()
    signal, rate = mono_sum(a.audio)
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    start = time.perf_counter()
    try:
        beats, downbeats = Audio2Beats(a.checkpoint, device)(signal, rate)
    except Exception as e:  # an operation missing on this accelerator
        print(f"{device} failed ({e}); using the CPU", file=sys.stderr)
        device = "cpu"
        beats, downbeats = Audio2Beats(a.checkpoint, device)(signal, rate)
    print(json.dumps({"beats": [round(float(b), 4) for b in beats],
                      "downbeats": [round(float(d), 4) for d in downbeats],
                      "name": "beat_this", "version": version("beat-this"), "checkpoint": a.checkpoint,
                      "device": device, "torch": torch.__version__, "duration_s": round(len(signal) / rate, 3),
                      "seconds": round(time.perf_counter() - start, 2)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
