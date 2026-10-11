"""Demucs stem separation, run in its own uv environment (torch stays out of
this package's). Standalone on purpose: imports nothing from refdissect.

usage: separate_worker.py AUDIO OUT_DIR [MODEL] [DEVICE]
Writes OUT_DIR/<stem>.wav (24-bit) and prints one JSON line with the result.
Same approach as ableton-live-mcp's `separate_stems` (MCP_Server/ml_worker.py).
"""

import json
import sys
from pathlib import Path


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    audio_path, out_dir = sys.argv[1], Path(sys.argv[2])
    model_name = sys.argv[3] if len(sys.argv) > 3 else "htdemucs"
    device = sys.argv[4] if len(sys.argv) > 4 else ("mps" if torch.backends.mps.is_available()
                                                     else "cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name)
    model.eval()
    data, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = data.T
    if rate != model.samplerate:
        from math import gcd

        from scipy.signal import resample_poly
        g = gcd(rate, model.samplerate)
        audio = resample_poly(audio, model.samplerate // g, rate // g, axis=1).astype("float32")
        rate = model.samplerate
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    wav = torch.from_numpy(audio[: model.audio_channels])
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8

    def run(dev):
        with torch.no_grad():
            return apply_model(model, ((wav - mean) / std)[None], device=dev, split=True, overlap=0.25)[0]
    try:
        sources = run(device)
    except Exception as e:  # an operation missing on this accelerator
        print(f"{device} failed ({e}); using the CPU", file=sys.stderr)
        device = "cpu"
        sources = run(device)
    sources = (sources * std + mean).cpu().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for i, name in enumerate(model.sources):
        target = out_dir / f"{name}.wav"
        sf.write(str(target), sources[i].T, rate, subtype="PCM_24")
        files[name] = str(target)
    print(json.dumps({"stems": files, "model": model_name, "device": device}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
