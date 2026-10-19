"""What Live will play, rendered offline from a plan: one track per Drum Rack
pad, their sum, and the 808 track, as float32 WAV after the track faders.
They hold the song's pattern played with your samples: keep them local, like
the plan.

The assumptions, each a parameter or an optional plan field (defaults in
brackets); the ones about Live are not measured here, a Live capture is the
check:

- a note starts at start_time * 60 / tempo seconds, on the nearest sample;
- a pad plays its file from frame 0 [start_s 0.0 moves the start], mono
  (channels averaged), transposed by transpose + detune_cents / 100
  [detune_cents 0] semitones the way Simpler does with warping off
  (audio.transpose: pitch and length change together);
- gain = track fader (build_spec's min(6, levels[...]) dB) x 10^(volume_db/20)
  [volume_db 0: build_spec sets Simpler to 0 dB] x a velocity law [linear
  v/127; Simpler's real curve is to be measured in Live];
- pads are One-Shot [playback_mode "one_shot"]: a note's length does not cut
  the file; "classic" cuts it at the note end. Hits of one pad overlap and sum
  unless mono=True: whether Live's One-Shot retrigger cuts the previous hit is
  not measured;
- pads with the same choke_group [None] cut each other: a hit fades out the
  sounding hits of the group's other pads;
- the 808 Simpler plays its one-shot per note, transposed by
  (pitch - 60) + transpose + detune_cents / 100 semitones (Simpler plays a
  sample at its own pitch on C3; with the build's Transpose 60 - root_midi
  that is pitch - root_midi) and cut at the note end (Classic mode,
  cut_808=True);
- every cut is a fade_ms [5 ms] linear release that starts at the cut;
- no effects and no glides; the reference stems Live plays beside these are
  the reference's own and are not rendered.

live.estimate_levels makes the same assumptions for its level estimate
(velocity v/127, One-Shot pads, the 808 per note cut at its end) except two
that leave an RMS over a whole track as it is: it trims each pad's lead-in,
which Live plays (it moves a hit, not its level), and cuts the 808 without a
fade.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, read, samples, transpose

PAD = {"start_s": 0.0, "detune_cents": 0.0, "playback_mode": "one_shot", "choke_group": None, "volume_db": 0.0,
       "velocity_curve": None}
TRACK_808 = {"cents": 0.0, "detune_cents": 0.0, "start_s": 0.0, "volume_db": 0.0, "playback_mode": "classic",
             "velocity_curve": None}


def resolve(path: str | Path, root: str | Path | None = None) -> Path:
    """A plan's path: as written when absolute, else under root (default: the
    working folder, where the pipeline wrote it)."""
    p = Path(path)
    return p if p.is_absolute() or root is None else Path(root) / p


def pad_settings(pad: dict) -> dict:
    """A pad with every optional field filled in (PAD holds the defaults)."""
    return {**PAD, **{k: v for k, v in pad.items() if v is not None},
            "transpose": float(pad.get("transpose", 0.0)), "role": pad.get("role", "perc")}


def settings_808(track: dict) -> dict:
    """The 808 track with every optional field filled in. cents is the
    one-shot's own pitch over root_midi, as metadata: fit_808 renders the
    one-shot at the 808's measured tuning (synth808.json pitch_hz) and
    make_plan writes that offset here; Simpler plays the file as it is, so
    the transposition does not depend on it."""
    root = int(track.get("root_midi", 60))
    return {**TRACK_808, **{k: v for k, v in track.items() if v is not None}, "root_midi": root,
            "transpose": float(track.get("transpose", 60 - root))}


def pitch_808(s: dict, pitch: int) -> tuple[float, float]:
    """(semitones the one-shot is played at, sounding MIDI pitch) of a note:
    shift = (pitch - 60) + transpose + detune_cents/100, sounding at
    root_midi + cents/100 + shift. With the build's Transpose 60 - root_midi
    that is pitch + cents/100."""
    shift = pitch - 60 + s["transpose"] + s["detune_cents"] / 100
    return shift, s["root_midi"] + s["cents"] / 100 + shift


def as_played(x: np.ndarray, start_s: float, semitones: float) -> np.ndarray:
    """A sample as Simpler plays it: from start_s, transposed with its length."""
    return transpose(x[int(round(start_s * SR)):], semitones)


def velocity_gain(velocity: int, curve: str | Callable = "linear") -> float:
    """Linear gain of a velocity: "linear" (v/127), "flat" (1), or curve(v)."""
    if callable(curve):
        return float(curve(velocity))
    if curve == "linear":
        return velocity / 127
    if curve == "flat":
        return 1.0
    raise ValueError(f"Unknown velocity curve {curve!r}")


def fader_db(plan: dict, track: str) -> float:
    """A track's fader as build_spec sets it: levels[track_db] capped at +6 dB,
    -headroom without one, 0 dB for a plan without levels."""
    levels = plan.get("levels") or {}
    return round(min(6.0, levels.get(f"{track}_track_db", -levels.get("headroom_db", 0.0))), 1)


def _cut(x: np.ndarray, length: int, fade: int) -> np.ndarray:
    """x stopped `length` samples in, with a linear release of `fade` samples
    after that (x itself when it ends first)."""
    if length >= len(x):
        return x
    y = x[: length + fade].copy()
    y[length:] *= np.linspace(1.0, 0.0, len(y) - length)
    return y


def _place(track: np.ndarray, x: np.ndarray, a: int, gain: float) -> None:
    b = min(len(track), a + len(x))
    if b > a >= 0:
        track[a:b] += gain * x[: b - a]


def _stops(hits: list[dict], group_of: dict, mono: bool) -> None:
    """Shorten each hit's stop to the first later hit that cuts it: one of
    another pad in its choke group, or (mono) one of its own pad."""
    hits.sort(key=lambda h: h["a"])
    for k, h in enumerate(hits):
        group = group_of.get(h["pad"])
        for j in range(k + 1, len(hits)):
            g = hits[j]
            if g["a"] >= h["stop"]:
                break
            same = g["pad"] == h["pad"]
            if g["a"] > h["a"] and ((mono and same) or (not same and group is not None
                                                         and group_of.get(g["pad"]) == group)):
                h["stop"] = g["a"]
                break


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


def _drum_hits(plan: dict, shots: dict, settings: dict, velocity_curve, faders: bool) -> list[dict]:
    bps = plan["tempo"] / 60
    g_track = 10 ** ((fader_db(plan, "drums") if faders else 0.0) / 20)
    hits = []
    for n in plan["tracks"]["drums"]["notes"]:
        key = str(n["pitch"])
        if key not in shots:
            continue
        s, a = settings[key], int(round(n["start_time"] / bps * SR))
        stop = a + len(shots[key])
        if s["playback_mode"] == "classic":
            stop = min(stop, a + int(round(n["duration"] / bps * SR)))
        curve = s["velocity_curve"] or velocity_curve
        hits.append({"pad": key, "a": a, "stop": stop,
                     "gain": g_track * 10 ** (s["volume_db"] / 20) * velocity_gain(n["velocity"], curve)})
    return hits


def render_plan(plan: dict, out_dir: str | Path, velocity_curve: str | Callable = "linear",
                root: str | Path | None = None, fade_ms: float = 5.0, cut_808: bool = True, mono: bool = False,
                faders: bool = True, n_samples: int | None = None) -> dict:
    """pads/<note>.wav, drums.wav (their sum) and 808.wav in out_dir, and
    render.json naming them with the settings used (also returned). A pad's
    own velocity_curve field overrides the argument. A file that cannot be
    read plays nothing, as in Live, and is listed under "missing"."""
    out = Path(out_dir)
    bps = plan["tempo"] / 60
    fade = samples(fade_ms)
    shots, settings, missing = {}, {}, []
    for note, pad in plan["tracks"]["drums"]["pads"].items():
        s = settings[str(note)] = pad_settings(pad)
        try:
            shots[str(note)] = as_played(read(resolve(pad["path"], root)), s["start_s"],
                                         s["transpose"] + s["detune_cents"] / 100)
        except (OSError, RuntimeError):
            missing.append(str(pad["path"]))
    hits = _drum_hits(plan, shots, settings, velocity_curve, faders)
    _stops(hits, {k: s["choke_group"] for k, s in settings.items()}, mono)
    t808 = plan["tracks"].get("808") or {}
    notes808 = (t808.get("notes") or []) if "sample_path" in t808 else []
    ends = [h["stop"] + fade for h in hits] + [int(round((n["start_time"] + n["duration"]) / bps * SR)) + fade
                                              for n in notes808]
    n = n_samples or max([int(round(plan.get("length_beats", 0) / bps * SR)), *ends, 1])
    manifest = {"sr": SR, "n_samples": n, "tempo": plan["tempo"], "velocity_curve": str(velocity_curve),
                "fade_ms": fade_ms, "mono": mono, "cut_808": cut_808,
                "faders_db": {"drums": fader_db(plan, "drums") if faders else 0.0}, "tracks": {},
                "drums": "drums.wav", "missing": missing}
    total = np.zeros(n, np.float32)
    for key in shots:
        x = np.zeros(n, np.float32)
        for h in hits:
            if h["pad"] == key:
                _place(x, _cut(shots[key], h["stop"] - h["a"], fade), h["a"], h["gain"])
        _write(out / "pads" / f"{key}.wav", x)
        manifest["tracks"][key] = f"pads/{key}.wav"
        total += x
    _write(out / "drums.wav", total)
    if notes808:
        manifest["faders_db"]["808"] = fader_db(plan, "808") if faders else 0.0
        try:
            shot = read(resolve(t808["sample_path"], root))
        except (OSError, RuntimeError):
            missing.append(str(t808["sample_path"]))
        else:
            _write(out / "808.wav", _render_808(t808, shot, n, bps, fade, cut_808,
                                                10 ** (manifest["faders_db"]["808"] / 20), velocity_curve))
            manifest["tracks"]["808"] = "808.wav"
    (out / "render.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _render_808(track: dict, shot: np.ndarray, n: int, bps: float, fade: int, cut: bool, gain: float,
                velocity_curve: str | Callable) -> np.ndarray:
    s = settings_808(track)
    out = np.zeros(n, np.float32)
    cache: dict[float, np.ndarray] = {}
    g = gain * 10 ** (s["volume_db"] / 20)
    for note in track["notes"]:
        shift, _ = pitch_808(s, note["pitch"])
        if shift not in cache:
            cache[shift] = as_played(shot, s["start_s"], shift)
        x = _cut(cache[shift], int(round(note["duration"] / bps * SR)), fade) if cut else cache[shift]
        _place(out, x, int(round(note["start_time"] / bps * SR)),
               g * velocity_gain(note["velocity"], s["velocity_curve"] or velocity_curve))
    return out
