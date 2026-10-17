"""Reading Basic Pitch's posteriorgrams by time.

Basic Pitch (0.4) runs at 22050 / 256 frames per second, but it predicts in
2-second windows of 172 frames and shifts each window back by ~10.3 ms
(basic_pitch.note_creation.model_frames_to_time). Frame t * 22050 / 256 is
therefore 0.8 s late by the end of a 3-minute song. Index the note and onset
posteriorgrams through `frame_times` / `frame_at`: checked on HAYIR
DEMELIYDIM, the onset posterior at Basic Pitch's own note starts is 0.45-0.47
this way and falls from 0.29 to 0.21 the naive way.
"""

from __future__ import annotations

import numpy as np

FRAME_S = 256 / 22050
WINDOW_FRAMES = 172
# each window's start moves back by this much (Basic Pitch's own constants,
# plus the 1.8 ms its model output is offset by)
WINDOW_SHIFT_S = FRAME_S * (WINDOW_FRAMES - (22050 * 2 - 256) / 256) + 0.0018

MIDI_OFFSET = 21  # posteriorgram column 0 is MIDI 21 (A0), up to 108


def frame_times(n_frames: int) -> np.ndarray:
    """Time (s) of every posteriorgram frame."""
    k = np.arange(n_frames)
    return k * FRAME_S - WINDOW_SHIFT_S * np.floor(k / WINDOW_FRAMES)


def frame_at(times: np.ndarray, t: float) -> int:
    """The first frame at or after t (times from frame_times)."""
    return int(np.searchsorted(times, t))


def column(midi: int) -> int:
    """Posteriorgram column of a MIDI note."""
    if not MIDI_OFFSET <= midi <= 108:
        raise ValueError(f"Basic Pitch covers MIDI 21-108, not {midi}")
    return midi - MIDI_OFFSET
