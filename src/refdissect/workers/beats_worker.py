"""Beats and downbeats with beat_this, run in its own uv environment (torch
stays out of this package's). Standalone on purpose: imports nothing from
refdissect.

usage: beats_worker.py AUDIO [AUDIO ...] [--device DEVICE] [--checkpoint NAME]
       beats_worker.py --serve
The files are summed to mono first (stems sum to the mix, which is what the
tracker was trained on). Prints one JSON line: beat and downbeat times in
seconds, and what ran (version, checkpoint, device). With --serve it keeps
running (refdissect/served.py): each line on its standard input is
{"argv": the arguments of a call}, answered on one line, MARK and then
{"result": what the call prints} or {"error": ...}; a model is loaded once
per checkpoint and device. seconds is the call's, its model's loading
included when it loads one.
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


MARK = "#refdissect-served# "  # served.MARK


def track(argv: list[str], models: dict) -> dict:
    """A call's output for its arguments; models: the loaded ones, by
    (checkpoint, device), kept between calls."""
    import torch
    from beat_this.inference import Audio2Beats

    p = argparse.ArgumentParser()
    p.add_argument("audio", nargs="+")
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default="final0")
    a = p.parse_args(argv)
    signal, rate = mono_sum(a.audio)
    device = a.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    def model(device: str):
        if (a.checkpoint, device) not in models:
            models[a.checkpoint, device] = Audio2Beats(a.checkpoint, device)
        return models[a.checkpoint, device]
    start = time.perf_counter()
    try:
        beats, downbeats = model(device)(signal, rate)
    except Exception as e:  # an operation missing on this accelerator
        print(f"{device} failed ({e}); using the CPU", file=sys.stderr)
        device = "cpu"
        beats, downbeats = model(device)(signal, rate)
    return {"beats": [round(float(b), 4) for b in beats],
            "downbeats": [round(float(d), 4) for d in downbeats],
            "name": "beat_this", "version": version("beat-this"), "checkpoint": a.checkpoint,
            "device": device, "torch": torch.__version__, "duration_s": round(len(signal) / rate, 3),
            "seconds": round(time.perf_counter() - start, 2)}


def serve(models: dict) -> int:
    """Calls answered one per line until the standard input closes."""
    import traceback

    for line in sys.stdin:
        try:
            answer = {"result": track(json.loads(line)["argv"], models)}
        except BaseException as e:  # argparse exits on bad arguments
            traceback.print_exc()
            answer = {"error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(MARK + json.dumps(answer) + "\n")
        sys.stdout.flush()
        release()
    return 0


def release() -> None:
    """What torch keeps on the GPU between calls given back: a kept worker
    held 1.71 GB of physical footprint after a call on v1-1's stems, 0.66 GB
    after this (the same beats on the next call)."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    if sys.argv[1:] == ["--serve"]:
        return serve({})
    print(json.dumps(track(sys.argv[1:], {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
