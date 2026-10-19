"""Reading, filtering, moving and pitching audio: the small operations
everything else is built on. All analysis runs on mono float32 at 44.1 kHz."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy.signal import butter, sosfiltfilt, sosfreqz

SR = 44100

# Bands that keep a drum role and drop most of what plays with it: a hat
# above 3.5 kHz loses the 808 and kick under it, a snare between 120 Hz and
# 14 kHz loses the sub. Comparisons filter both sides the same way.
BANDS: dict[str, tuple[float | None, float | None]] = {
    "kick": (None, 9000.0),
    "snare": (120.0, 14000.0),
    "hat": (3500.0, None),
    "808": (None, 1500.0),
    "low": (None, 200.0),  # a kick's body: finding kicks under hats and snares
    "full": (None, None),
}


def samples(ms: float) -> int:
    """Milliseconds to samples at SR."""
    return int(round(ms * SR / 1000.0))


PRE = samples(5)  # windows start this long before an onset


def read(path: str | Path, seconds: float | None = None, offset: float = 0.0,
         stereo: bool = False) -> np.ndarray:
    """Audio as float32 at SR: mono (channels averaged) or (channels, n)."""
    info = sf.info(str(path))
    start = int(offset * info.samplerate)
    frames = -1 if seconds is None else int(seconds * info.samplerate) + 1
    data, rate = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    x = data.T if stereo else data.mean(axis=1)
    if rate != SR:
        x = soxr.resample(x.T if stereo else x, rate, SR).astype(np.float32)
        x = x.T if stereo else x
    return np.ascontiguousarray(x, dtype=np.float32)


def write(path: str | Path, x: np.ndarray, peak: float | None = 0.9, subtype: str = "PCM_24") -> Path:
    """Write mono or (n, channels) audio; peak-normalized unless peak is None."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if peak is not None:
        x = x / (np.max(np.abs(x)) or 1.0) * peak
    sf.write(str(path), x, SR, subtype=subtype)
    return path


def _sos(role: str):
    lo, hi = BANDS[role]
    if lo and hi:
        return butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    if lo:
        return butter(4, lo, btype="high", fs=SR, output="sos")
    if hi:
        return butter(4, hi, btype="low", fs=SR, output="sos")
    return None


def band(x: np.ndarray, role: str) -> np.ndarray:
    """x filtered (zero phase) to the role's band."""
    sos = _sos(role)
    if sos is None:
        return np.asarray(x, dtype=np.float32)
    return sosfiltfilt(sos, x).astype(np.float32)


def response(role: str, n: int) -> np.ndarray:
    """band()'s magnitude response at the rfft bins of length n, so spectra can
    be filtered the same way without filtering every signal first."""
    f = np.fft.rfftfreq(n, 1 / SR)
    sos = _sos(role)
    if sos is None:
        return np.ones_like(f, dtype=np.float32)
    _, h = sosfreqz(sos, worN=f, fs=SR)
    return (np.abs(h) ** 2).astype(np.float32)  # forward and backward pass


def attack(x: np.ndarray, rise: float = 0.3, search: int | None = None) -> int:
    """Where the attack is: the first sample whose 1 ms envelope reaches
    `rise` of the peak (of the first `search` samples)."""
    head = np.abs(x[:search] if search else x)
    if not len(head) or not head.any():
        return 0
    w = samples(1)
    env = np.convolve(head, np.ones(w) / w, mode="same")
    return int(np.argmax(env >= rise * env.max()))


# The band each role's attack is read in: a hat's attack is where its high
# band rises, not where the kick under it does.
ATTACK_BAND = {"kick": "kick", "snare": "snare", "clap": "snare", "rim": "snare", "perc": "snare", "tom": "snare",
               "hat": "hat", "open_hat": "hat", "crash": "hat", "ride": "hat", "crash/ride": "hat", "808": "808"}


def attack_time(x: np.ndarray, role: str = "full", rise: float = 0.3, search_s: float = 0.25) -> float:
    """Seconds from the start of x to its attack (attack() in the role's
    band, over the first search_s). The one definition of when a sound
    happens: truth, detections and the plan all use it, so their times can
    be compared without removing an offset per sound."""
    head = x[: int(search_s * SR) + samples(20)]
    return attack(band(head, ATTACK_BAND.get(role, role if role in BANDS else "full")), rise,
                  search=int(search_s * SR)) / SR


def trim_onset(x: np.ndarray, floor_db: float = -30.0, pre_ms: float = 2.0) -> np.ndarray:
    """x from pre_ms before it first rises above floor_db (relative to its peak):
    where the sound starts, lead-in included."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak == 0:
        return x
    first = int(np.argmax(np.abs(x) > peak * 10 ** (floor_db / 20)))
    return x[max(0, first - samples(pre_ms)):]


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """x cut or zero-padded to n samples."""
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def shift(x: np.ndarray, lag: int) -> np.ndarray:
    """x moved by lag samples (positive: later), same length."""
    if lag > 0:
        return np.r_[np.zeros(lag, x.dtype), x[:-lag]]
    if lag < 0:
        return np.r_[x[-lag:], np.zeros(-lag, x.dtype)]
    return x


def transpose(x: np.ndarray, semitones: float) -> np.ndarray:
    """Played faster or slower, as Simpler/Sampler do with warping off: the
    pitch and the length change together."""
    if not semitones:
        return x
    rate = 2 ** (semitones / 12)
    return soxr.resample(x, SR * rate, SR).astype(np.float32)


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))
