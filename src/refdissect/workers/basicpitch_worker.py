"""Basic Pitch 0.4.0 note transcription, run in its own uv environment (the
Core ML model on macOS; no TensorFlow in this package's). Standalone on
purpose: imports nothing from refdissect.

usage: basicpitch_worker.py OUT_DIR SETTINGS AUDIO [AUDIO ...]
SETTINGS is JSON: {preset name: {onset, frame, min_ms, melodia, min_hz, max_hz}}.
Per audio file the model runs once; its posteriorgrams go to OUT_DIR/<name>.npz
(float16) and every preset is decoded from them into OUT_DIR/<name>.json.
Prints one JSON line with the result.

Three things differ from basic_pitch.inference.predict, each measured:
- 0.4.0 keeps 86 frames per second of the 86.58 it makes (142 frames per
  36,164-sample hop), so a song loses its last 0.67 %: 1.6 s of a 4-minute
  file. The audio is padded by 0.007 x duration + 0.5 s and the frames past
  its end are dropped.
- Frame times follow those 142-frame blocks (block_times). Basic Pitch's own
  model_frames_to_time steps 10.3 ms every 172 frames instead, a periodic
  error of about 18 ms peak to peak.
- Bends are one contour bin (a third of a semitone) sharp: get_pitch_bends
  centres a note on bin 3 * (pitch - 21), the model on the bin above. They
  are returned corrected (bend - 1), in thirds of a semitone per frame.
"""

import contextlib
import io
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

SR = 22050
FFT_HOP = 256
WINDOW_FRAMES = 2 * (SR // FFT_HOP)  # 172 frames per 2 s window
OVERLAP = 30 * FFT_HOP               # 30 frames of overlap between windows
HOP = 2 * SR - FFT_HOP - OVERLAP     # 36,164 samples
BLOCK = WINDOW_FRAMES - 30           # 142 frames kept per window
LOWEST_HZ = 27.5                     # a lower minimum frequency zeroes almost every bin


def block_times(n_frames: int) -> np.ndarray:
    """Time (s) of each unwrapped frame: frame j of block b sits b hops plus
    j frames into the audio."""
    k = np.arange(n_frames)
    return ((k // BLOCK) * HOP + (k % BLOCK) * FFT_HOP) / SR


def min_note_frames(ms: float) -> int:
    """predict()'s conversion: a note needs more than this many frames."""
    return int(np.round(ms / 1000 * SR / FFT_HOP))


def infer(model, audio: np.ndarray) -> dict:
    """run_inference on an array: the same 2 s windows, overlap, lead-in and
    unwrapping, on audio padded at the end; frames past its end are dropped."""
    from basic_pitch.inference import unwrap_output, window_audio_file

    n = len(audio)
    tail = int((0.007 * n / SR + 0.5) * SR)
    x = np.concatenate([np.zeros(OVERLAP // 2, np.float32), np.asarray(audio, np.float32),
                        np.zeros(tail, np.float32)])
    out: dict = {"note": [], "onset": [], "contour": []}
    with contextlib.redirect_stdout(io.StringIO()):  # the Core ML wrapper prints every window
        for window, _ in window_audio_file(x, HOP):
            for k, v in model.predict(window[None]).items():
                out[k].append(v)
    frames = {k: unwrap_output(np.concatenate(v), n + tail, OVERLAP // FFT_HOP) for k, v in out.items()}
    keep = int(np.searchsorted(block_times(len(frames["note"])), n / SR))
    return {k: v[:keep] for k, v in frames.items()}


def decode(output: dict, s: dict) -> list[dict]:
    """One preset's notes: model_output_to_notes's two steps (notes, then
    bends) on copies, since constrain_frequency zeroes bins in place."""
    from basic_pitch import note_creation as nc

    min_hz = max(float(s.get("min_hz") or LOWEST_HZ), LOWEST_HZ)
    events = nc.output_to_notes_polyphonic(output["note"].copy(), output["onset"].copy(),
                                           onset_thresh=s["onset"], frame_thresh=s["frame"],
                                           min_note_len=min_note_frames(s["min_ms"]), infer_onsets=True,
                                           max_freq=s.get("max_hz"), min_freq=min_hz,
                                           melodia_trick=bool(s["melodia"]))
    times = block_times(len(output["note"]) + 1)
    notes = []
    for a, b, pitch, amp, bends in nc.get_pitch_bends(output["contour"], events):
        col = int(pitch) - 21
        onset = float(output["onset"][max(0, a - 2):a + 3, col].max())
        notes.append({"start_s": round(float(times[a]), 4), "end_s": round(float(times[b]), 4),
                      "start_frame": int(a), "end_frame": int(b), "midi": int(pitch),
                      "amplitude": round(float(amp), 4), "onset": round(onset, 4),
                      "bend": [int(v) - 1 for v in bends]})
    return sorted(notes, key=lambda n: (n["start_s"], n["midi"]))


def main() -> int:
    import librosa
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import Model

    out_dir, settings = Path(sys.argv[1]), json.loads(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = Model(ICASSP_2022_MODEL_PATH)
    result = {"stems": {}, "model": "basic-pitch 0.4.0", "backend": model.model_type.name,
              "python": platform.python_version(), "settings": settings}
    for path in sys.argv[3:]:
        name = Path(path).stem
        t = time.time()
        audio, _ = librosa.load(path, sr=SR, mono=True)
        output = infer(model, audio)
        infer_s = time.time() - t
        np.savez_compressed(out_dir / f"{name}.npz", **{k: v.astype(np.float16) for k, v in output.items()})
        notes = {preset: decode(output, s) for preset, s in settings.items()}
        target = out_dir / f"{name}.json"
        target.write_text(json.dumps({"stem": name, "audio": str(path), "seconds": round(len(audio) / SR, 3),
                                      "frames": len(output["note"]), "notes": notes}), encoding="utf-8")
        result["stems"][name] = {"notes": str(target), "posteriorgram": str(out_dir / f"{name}.npz"),
                                 "seconds": round(len(audio) / SR, 3), "infer_s": round(infer_s, 2),
                                 "notes_per_preset": {k: len(v) for k, v in notes.items()}}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
