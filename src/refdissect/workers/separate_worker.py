"""Demucs stem separation, run in its own uv environment (torch stays out of
this package's). Standalone on purpose: imports nothing from refdissect.

usage: separate_worker.py MODEL DEVICE SEED SHIFTS AUDIO OUT_DIR [AUDIO OUT_DIR ...]
DEVICE "auto" takes MPS, else CUDA, else the CPU; an accelerator that fails
falls back to the CPU. Per job: OUT_DIR/<stem>.wav (24-bit) and
OUT_DIR/separation.json, also printed as one JSON line. The model loads once
for all jobs.

apply_model moves the input by a random offset per shift, and per model of a
bag (htdemucs_ft holds four), drawn with random.randint. random, numpy and
torch are seeded with SEED before every job, so a job's stems depend on its
input, model, seed, shifts and device, not on the jobs before it: a v1 beat
separated alone and second in a batch gave byte-identical stems (MPS,
htdemucs and htdemucs_ft). Unseeded, runs differ: three htdemucs_ft seeds of
one v1 beat scored drums SDRs within 0.15 dB of each other, and blind kick F1
0.99, 0.99 and 0.00.
Same approach as ableton-live-mcp's `separate_stems` (MCP_Server/ml_worker.py).
"""

import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path


def versions() -> dict:
    import demucs
    import numpy
    import scipy
    import soundfile
    import torch
    return {"python": platform.python_version(), "demucs": demucs.__version__, "torch": torch.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__, "soundfile": soundfile.__version__}


def load(path: str, rate: int, channels: int):
    """The file as float32 (channels, n) at the model's rate (mono doubled)."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = data.T
    if sr != rate:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(sr, rate)
        audio = resample_poly(audio, rate // g, sr // g, axis=1).astype("float32")
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    return np.ascontiguousarray(audio[:channels])


def apply(model, audio, device: str, seed: int, shifts: int):
    """Demucs on audio standardised by its mono mean and deviation, seeded."""
    import numpy as np
    import torch
    from demucs.apply import apply_model
    wav = torch.from_numpy(audio)
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device, shifts=shifts, split=True, overlap=0.25)
    return (out[0] * std + mean).cpu().numpy()


def write(sources, names, out_dir: Path, rate: int) -> tuple[dict, dict, dict]:
    """Each stem as 24-bit WAV, clipped to full scale (which 24-bit holds);
    its peak and the samples clipped are returned with the files."""
    import numpy as np
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    files, peak, clipped = {}, {}, {}
    for x, name in zip(sources, names):
        target = out_dir / f"{name}.wav"
        peak[name] = round(float(np.max(np.abs(x))), 4)
        clipped[name] = int(np.sum(np.abs(x) > 1.0))
        sf.write(str(target), np.clip(x, -1.0, 1.0).T, rate, subtype="PCM_24")
        files[name] = str(target)
    return files, peak, clipped


def main() -> int:
    import torch
    from demucs.pretrained import get_model

    model_name, device, seed, shifts = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    jobs = list(zip(sys.argv[5::2], sys.argv[6::2]))
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = get_model(model_name)
    model.eval()
    load_s = round(time.perf_counter() - t0, 2)
    env = versions()
    for audio_path, out in jobs:
        audio = load(audio_path, model.samplerate, model.audio_channels)
        t1 = time.perf_counter()
        try:
            sources = apply(model, audio, device, seed, shifts)
        except Exception as e:  # an operation missing on this accelerator
            print(f"{device} failed ({e}); using the CPU", file=sys.stderr)
            device = "cpu"
            t1 = time.perf_counter()
            sources = apply(model, audio, device, seed, shifts)
        seconds = round(time.perf_counter() - t1, 2)
        files, peak, clipped = write(sources, model.sources, Path(out), model.samplerate)
        info = {"stems": files, "model": model_name, "seed": seed, "shifts": shifts, "device": device,
                "versions": env, "audio": audio_path,
                "audio_sha256": hashlib.sha256(Path(audio_path).read_bytes()).hexdigest(),
                "audio_s": round(audio.shape[1] / model.samplerate, 3), "seconds": seconds, "model_load_s": load_s,
                "peak": peak, "clipped_samples": clipped}
        (Path(out) / "separation.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        print(json.dumps(info), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
