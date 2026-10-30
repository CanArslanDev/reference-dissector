"""Basic Pitch: running it, and reading its posteriorgrams by time.

Basic Pitch (0.4) runs at 22050 / 256 frames per second, but it predicts in
2-second windows of 172 frames and shifts each window back by ~10.3 ms
(basic_pitch.note_creation.model_frames_to_time). Frame t * 22050 / 256 is
therefore 0.8 s late by the end of a 3-minute song. Index the note and onset
posteriorgrams through `frame_times` / `frame_at`: checked on HAYIR
DEMELIYDIM, the onset posterior at Basic Pitch's own note starts is 0.45-0.47
this way and falls from 0.29 to 0.21 the naive way.

`frame_times` is Basic Pitch's own mapping: right on average, but the frames
really come in blocks of 142 (one per 36,164-sample hop, 8.5 ms back every
142 frames, not 10.3 ms every 172), so it is up to ~18 ms off in between.
`block_times` follows the blocks; the worker (workers/basicpitch_worker.py)
times its notes with it. On 865 synthetic pluck notes (2 ms attack) the onset
error's p90 distance from its median fell from 8.9 to 6.2 ms, note F1 equal.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

FRAME_S = 256 / 22050
WINDOW_FRAMES = 172
# each window's start moves back by this much (Basic Pitch's own constants,
# plus the 1.8 ms its model output is offset by)
WINDOW_SHIFT_S = FRAME_S * (WINDOW_FRAMES - (22050 * 2 - 256) / 256) + 0.0018
BLOCK_FRAMES = WINDOW_FRAMES - 30  # frames kept per window after the overlap is cut
BLOCK_HOP = 2 * 22050 - 256 - 30 * 256  # samples between windows

MIDI_OFFSET = 21  # posteriorgram column 0 is MIDI 21 (A0), up to 108

WORKER = Path(__file__).with_name("workers") / "basicpitch_worker.py"
# 3.11: posteriorgrams identical to 3.10 (Python 3.10 is past its end of life),
# 3.12 cannot install basic-pitch 0.4.0 on macOS. setuptools < 81: resampy
# still imports pkg_resources.
PYTHON = "3.11"
PACKAGES = ("basic-pitch==0.4.0", "setuptools<81")

# Decoding settings (onset / frame thresholds, minimum note length, melodia),
# chosen on 63 synthetic renders (pads, 16th and 32nd plucks at 140 and 75 BPM,
# a lead with glides and bends, full mixes; clean, under a distorted 808, with
# a library loop as bleed, in heavy reverb) by macro onset F1:
# - melodic: 0.705 (held-out seed 0.663) against 0.583 (0.529) for
#   ableton-live-mcp's audio_to_midi defaults (0.5 / 0.3 / 127.7 ms / melodia);
#   55 of 63 renders better. 127.7 ms drops every note under 139 ms, 16ths at
#   140 BPM included; melodia off stops re-triggers from tails.
# - chord: melodia on keeps the chord tones of slow pads that never make an
#   onset peak (without it half of them go missing).
PRESETS = {
    "melodic": {"onset": 0.6, "frame": 0.3, "min_ms": 35.0, "melodia": False},
    "chord": {"onset": 0.6, "frame": 0.3, "min_ms": 35.0, "melodia": True},
}


def frame_times(n_frames: int) -> np.ndarray:
    """Time (s) of every posteriorgram frame, as Basic Pitch maps them."""
    k = np.arange(n_frames)
    return k * FRAME_S - WINDOW_SHIFT_S * np.floor(k / WINDOW_FRAMES)


def block_times(n_frames: int) -> np.ndarray:
    """Time (s) of every frame, block by block: frame j of block b sits b
    window hops plus j frames into the audio."""
    k = np.arange(n_frames)
    return ((k // BLOCK_FRAMES) * BLOCK_HOP + (k % BLOCK_FRAMES) * 256) / 22050


def frame_at(times: np.ndarray, t: float) -> int:
    """The first frame at or after t (times from frame_times or block_times)."""
    return int(np.searchsorted(times, t))


def column(midi: int) -> int:
    """Posteriorgram column of a MIDI note."""
    if not MIDI_OFFSET <= midi <= 108:
        raise ValueError(f"Basic Pitch covers MIDI 21-108, not {midi}")
    return midi - MIDI_OFFSET


def command(audio: list[str | Path], out_dir: str | Path, presets: dict, files: dict | None = None) -> list[str]:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is needed to run Basic Pitch in its own environment: https://docs.astral.sh/uv/")
    cmd = [uv, "run", "--isolated", "--no-project", "--python", PYTHON]
    for p in PACKAGES:
        cmd += ["--with", p]
    return cmd + ["python", "-I", str(WORKER), str(Path(out_dir).expanduser()),
                  json.dumps({"presets": presets, "files": files or {}}), *(str(Path(a).expanduser()) for a in audio)]


def transcribe(audio: list[str | Path], out_dir: str | Path, presets: dict, files: dict | None = None) -> dict:
    """Run the model once per file and decode every preset ({name: {onset,
    frame, min_ms, melodia, min_hz, max_hz}}); files ({file stem: {tuning,
    min_hz}}) overrides the presets for a file, whose tuning (semitones from
    A440) is taken out before inference. Writes OUT_DIR/<stem>.json (notes per
    preset in the file's seconds, tuning, time_scale) and OUT_DIR/<stem>.npz
    (posteriorgrams, float16; frame times block_times x time_scale)."""
    done = subprocess.run(command(audio, out_dir, presets, files), capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"Basic Pitch failed:\n{done.stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])
