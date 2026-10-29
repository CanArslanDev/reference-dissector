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
  [volume_db 0, a pad's at most 0 (pad_settings); build_spec sets a
  Simpler's Volume to it] x the velocity's level ["live": velocity_db, Live
  12.4.6's law, at the pad's or the 808's vol_vel, which build_spec sets as
  Simpler's "Vol < Vel" (VOL_VEL); "linear", v/127, was the assumption
  before the law was measured];
- pads are One-Shot [playback_mode "one_shot"]: a note's length does not cut
  the file; "classic" cuts it at the note end. A pad plays one voice [mono
  True]: a hit that starts while the pad's hit before still plays (its file
  to the last frame, Snap off) cuts that one with a 1 ms linear fade from
  its note and starts RETRIGGER_S late, as Live 12.4.6 played 391 such hits
  (late_starts); mono=False sums them;
- pads with the same choke_group [None] cut each other: a hit fades out the
  sounding hits of the group's other pads;
- the 808 Simpler plays its one-shot per note, transposed by
  (pitch - 60) + transpose + detune_cents / 100 semitones (Simpler plays a
  sample at its own pitch on C3; with live.tune_808's Transpose and Detune
  a note sounds at its pitch plus the 808's tuning) and cut at the note end
  (Classic mode, cut_808=True);
- with Glide on (the track's voices 1, glide_mode "Glide", glide_time_ms),
  a note that starts while the one before is held does not start the
  one-shot again: the voice slides to the new pitch, linearly in semitones
  over glide_time_ms, and stops at the last note's end (chains_808; measured
  in Live 12.4.6 on a 1 kHz sine: overlapping notes only, exactly the Glide
  Time, for 2 to 12 semitones);
- every cut is a fade_ms [5 ms] linear release that starts at the cut;
- no effects; the reference stems Live plays beside these are the
  reference's own and are not rendered.

live.estimate_levels makes the same assumptions for its level estimate
(velocity_db, One-Shot pads of one voice, the 808 per note cut at its end)
except two that leave an RMS over a whole track as it is: it trims each
pad's lead-in, which Live plays (it moves a hit, not its level), and cuts
the 808 without a fade.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, read, samples, transpose

# Simpler's "Vol < Vel" (0..1) the build sets on every pad and on the 808, so that velocity_db holds however
# the Simpler was made: Live 12.4.6 starts it at 0.35 on a Simpler made with insert_device and at 0.45 on a
# pad loaded through the browser, ableton-live-mcp 2.2.0's only way to load a pad
VOL_VEL = 0.35
PAD = {"start_s": 0.0, "detune_cents": 0.0, "playback_mode": "one_shot", "choke_group": None, "volume_db": 0.0,
       "velocity_curve": None, "vol_vel": VOL_VEL}
TRACK_808 = {"cents": 0.0, "detune_cents": 0.0, "start_s": 0.0, "volume_db": 0.0, "playback_mode": "classic",
             "velocity_curve": None, "vol_vel": VOL_VEL}
# A note that comes while a voice of its pitch still sounds (a pad's hit before it) fades that voice out along a
# straight line over RETRIGGER_S from the note and starts RETRIGGER_S late: in the live check (Live 12.4.6,
# 2026-10-29; 48 kHz captures of 8 suite plans and PIRAMIT's) 391 such hits on 18 pads started 1.000 ms late
# (median; 98 % within 0.95-1.05 ms, read on a 1/32 ms grid), the hit before gone in that ms, and 13 hits within
# their pad's file but after Snap had ended its voice started on time
RETRIGGER_S = 0.001


def resolve(path: str | Path, root: str | Path | None = None) -> Path:
    """A plan's path: as written when absolute, else under root (default: the
    working folder, where the pipeline wrote it)."""
    p = Path(path)
    return p if p.is_absolute() or root is None else Path(root) / p


def pad_settings(pad: dict) -> dict:
    """A pad with every optional field filled in (PAD holds the defaults),
    its volume_db at most 0 dB: the build keeps every pad's Simpler at or
    under 0 dB (live.build_spec sends this Volume), and every render and
    estimate of the plan plays it so."""
    return {**PAD, **{k: v for k, v in pad.items() if v is not None},
            "transpose": float(pad.get("transpose", 0.0)), "role": pad.get("role", "perc"),
            "volume_db": min(0.0, float(pad.get("volume_db") or 0.0))}


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
    root_midi + cents/100 + shift. With live.tune_808's Transpose and Detune
    that is pitch + the 808's tuning (and heard_cents)."""
    shift = pitch - 60 + s["transpose"] + s["detune_cents"] / 100
    return shift, s["root_midi"] + s["cents"] / 100 + shift


def glide_s(track: dict) -> float:
    """The 808 track's Glide Time (s) when it glides (one voice, Glide Mode
    "Glide"), else 0."""
    if track.get("voices") == 1 and track.get("glide_mode") == "Glide":
        return float(track.get("glide_time_ms") or 0.0) / 1000
    return 0.0


def chains_808(track: dict) -> list[list[dict]]:
    """The 808's notes as one voice plays them: with Glide on, a note that
    starts while the one before is held joins its chain (it slides there);
    every other note starts a chain of its own."""
    gliding = glide_s(track) > 0
    chains: list[list[dict]] = []
    for n in sorted(track.get("notes") or [], key=lambda n: n["start_time"]):
        last = chains[-1][-1] if chains else None
        if gliding and last is not None and n["start_time"] < last["start_time"] + last["duration"] - 1e-9:
            chains[-1].append(n)
        else:
            chains.append([n])
    return chains


def chain_semitones(chain: list[dict], s: dict, bps: float, glide: float) -> np.ndarray:
    """Per sample, from the chain's first note to its last note's end, the
    semitones the one-shot plays at (pitch_808's shift): each later note's
    shift reached linearly over `glide` seconds from where the voice was."""
    a0 = chain[0]["start_time"] / bps
    end = (chain[-1]["start_time"] + chain[-1]["duration"]) / bps
    semis = np.full(max(1, int(round((end - a0) * SR))), pitch_808(s, chain[0]["pitch"])[0], dtype=np.float64)
    ramp = max(1, int(round(glide * SR)))
    for n in chain[1:]:
        o = int(round((n["start_time"] / bps - a0) * SR))
        if 0 < o < len(semis):
            here, to = semis[o - 1], pitch_808(s, n["pitch"])[0]
            semis[o:] = here + (to - here) * np.minimum(np.arange(1, len(semis) - o + 1) / ramp, 1.0)
    return semis


def read_at(semis: np.ndarray) -> np.ndarray:
    """Where a sampler reads its sample (samples) at each output sample when
    it plays it `semis` higher per output sample: pitch and speed together."""
    return np.r_[0.0, np.cumsum(2 ** (semis[:-1] / 12))]


def varispeed(x: np.ndarray, semis: np.ndarray) -> np.ndarray:
    """x read at a rate that follows `semis` per output sample (read_at)."""
    return np.interp(read_at(semis), np.arange(len(x)), x, right=0.0).astype(np.float32)


def as_played(x: np.ndarray, start_s: float, semitones: float) -> np.ndarray:
    """A sample as Simpler plays it: from start_s, transposed with its length."""
    return transpose(x[int(round(start_s * SR)):], semitones)


def late_starts(times, ends, shift: bool = True) -> np.ndarray:
    """Which of one voice's notes (in time order: their times, and where
    each one's sound ends when it starts on time, s) Live starts
    RETRIGGER_S late: those that come before the sound of the note before
    has ended. With shift, a note that starts late ends that much later
    too (a One-Shot pad plays its whole file from where it starts)."""
    late = np.zeros(len(times), bool)
    for k in range(1, len(times)):
        late[k] = times[k] < ends[k - 1] + RETRIGGER_S * (shift and late[k - 1])
    return late


def velocity_db(velocity: float, vol_vel: float = VOL_VEL) -> float:
    """A velocity's level over Simpler's Volume (dB), as Live 12.4.6 plays
    it: 0.4 x vol_vel ("Vol < Vel", 0..1) x (velocity - 70) - 0.096.
    ableton-live-mcp's battery measured it within 0.002 dB over 6 amounts
    and 18 velocities (2026-10-19): velocity 127 plays 7.9 dB over the
    Volume at 0.35 and 10.2 dB at 0.45, velocity 70 at it. Velocity 35
    plays 12.9 dB under 127 at 0.35, where v/127 put it 11.2 dB under."""
    return 0.4 * vol_vel * (velocity - 70) - 0.096


def velocity_gain(velocity: float, curve: str | Callable = "live", vol_vel: float = VOL_VEL) -> float:
    """Linear gain of a velocity: "live" (velocity_db at vol_vel), "linear"
    (v/127), "flat" (1), or curve(v)."""
    if callable(curve):
        return float(curve(velocity))
    if curve == "live":
        return 10 ** (velocity_db(velocity, vol_vel) / 20)
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


def _chokes(hits: list[dict], group_of: dict) -> None:
    """Shorten each hit's stop to the first later hit that cuts it: one of
    another pad in its choke group (with the render's fade)."""
    hits.sort(key=lambda h: h["a"])
    for k, h in enumerate(hits):
        group = group_of.get(h["pad"])
        if group is None:
            continue
        for g in hits[k + 1:]:
            if g["a"] >= h["stop"]:
                break
            if g["pad"] != h["pad"] and g["a"] > h["a"] and group_of.get(g["pad"]) == group:
                h["stop"] = g["a"]
                h.pop("fade", None)
                break


def _write(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), SR, subtype="FLOAT")


def pad_hits(plan: dict, shots: dict, settings: dict, velocity_curve="live", faders: bool = True,
             mono: bool = True) -> list[dict]:
    """The plan's drum hits on the pads with a sample in shots (each as
    played, settings[key] its pad_settings), as Live plays them: per hit its
    pad, t (s) and a (sample) where its sample starts, stop (sample) and
    gain (the drum fader with faders, the pad's Volume, the velocity's
    level). mono: one voice per pad (late_starts), a hit that comes while
    the pad's hit before plays cutting that one at its note (its fade
    RETRIGGER_S long) and starting RETRIGGER_S late. Choke groups cut
    across pads after that (_chokes)."""
    bps = plan["tempo"] / 60
    g_track = 10 ** ((fader_db(plan, "drums") if faders else 0.0) / 20)
    late_n = int(round(RETRIGGER_S * SR))
    by_pad: dict[str, list[dict]] = {}
    for n in plan["tracks"]["drums"]["notes"]:
        if str(n["pitch"]) in shots:
            by_pad.setdefault(str(n["pitch"]), []).append(n)
    hits = []
    for key, notes in by_pad.items():
        notes = sorted(notes, key=lambda n: n["start_time"])
        s, length = settings[key], len(shots[key])
        t = np.array([n["start_time"] / bps for n in notes])
        ends = t + length / SR
        if s["playback_mode"] == "classic":
            ends = np.minimum(ends, t + np.array([n["duration"] / bps for n in notes]))
        late = late_starts(t, ends) if mono else np.zeros(len(t), bool)
        for k, n in enumerate(notes):
            on = int(round(t[k] * SR))  # the note
            if late[k]:  # the pad's hit before, appended last, fades out from this note
                hits[-1]["stop"], hits[-1]["fade"] = on, late_n
            a = on + late_n * int(late[k])
            stop = a + length
            if s["playback_mode"] == "classic":
                stop = min(stop, on + int(round(n["duration"] / bps * SR)))
            curve = s["velocity_curve"] or velocity_curve
            hits.append({"pad": key, "t": float(t[k]) + RETRIGGER_S * int(late[k]), "a": a, "stop": stop,
                         "late": bool(late[k]), "velocity": int(n["velocity"]),
                         "gain": g_track * 10 ** (s["volume_db"] / 20) * velocity_gain(n["velocity"], curve,
                                                                                         s["vol_vel"])})
    _chokes(hits, {k: s["choke_group"] for k, s in settings.items()})
    return hits


def pad_track(shot: np.ndarray, hits: list[dict], n: int, fade: int = 0) -> np.ndarray:
    """One pad's hits (pad_hits) of shot, n samples, each cut at its stop
    with its own fade or `fade` samples."""
    x = np.zeros(n, np.float32)
    for h in hits:
        _place(x, _cut(shot, h["stop"] - h["a"], h.get("fade", fade)), h["a"], h["gain"])
    return x


def render_plan(plan: dict, out_dir: str | Path, velocity_curve: str | Callable = "live",
                root: str | Path | None = None, fade_ms: float = 5.0, cut_808: bool = True, mono: bool = True,
                faders: bool = True, n_samples: int | None = None) -> dict:
    """pads/<note>.wav, drums.wav (their sum) and 808.wav in out_dir, and
    render.json naming them with the settings used (also returned). A pad's
    own velocity_curve field overrides the argument. A file that cannot be
    read plays nothing, as in Live, and is listed under "missing". mono:
    each pad one voice, as Live plays it (pad_hits); False sums a pad's
    hits."""
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
    hits = pad_hits(plan, shots, settings, velocity_curve, faders, mono)
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
        x = pad_track(shots[key], [h for h in hits if h["pad"] == key], n, fade)
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
            _write(out / "808.wav", render_808(t808, shot, n, bps, fade, cut_808,
                                                10 ** (manifest["faders_db"]["808"] / 20), velocity_curve))
            manifest["tracks"]["808"] = "808.wav"
    (out / "render.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def render_808(track: dict, shot: np.ndarray, n: int, bps: float, fade: int, cut: bool, gain: float,
               velocity_curve: str | Callable) -> np.ndarray:
    """The 808 track (n samples, at `gain`): one voice per chain (chains_808), each
    cut `fade` samples after its last note's end when cut."""
    s = settings_808(track)
    out = np.zeros(n, np.float32)
    cache: dict[float, np.ndarray] = {}
    g = gain * 10 ** (s["volume_db"] / 20)
    for chain in chains_808(track):
        first = chain[0]
        if len(chain) == 1:
            shift, _ = pitch_808(s, first["pitch"])
            if shift not in cache:
                cache[shift] = as_played(shot, s["start_s"], shift)
            x = cache[shift]
        else:  # the voice slides on: one sample read at the chain's rates, the first note's velocity
            x = varispeed(shot[int(round(s["start_s"] * SR)):], chain_semitones(chain, s, bps, glide_s(track)))
        length = int(round((chain[-1]["start_time"] + chain[-1]["duration"] - first["start_time"]) / bps * SR))
        _place(out, _cut(x, length, fade) if cut else x, int(round(first["start_time"] / bps * SR)),
               g * velocity_gain(first["velocity"], s["velocity_curve"] or velocity_curve, s["vol_vel"]))
    return out
