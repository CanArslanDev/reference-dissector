"""Calibration: beats built from known library samples, so every step can be
scored against the answer.

build_beat() writes a trap beat (kick, snare or clap, hats, open hat, an 808
playing notes, plus a melodic loop and a vocal from the library as bleed),
mastered loud (RMS -12 dBFS, then +5 dB into a soft clipper, about -8.6 dB
RMS, like the references) and passed through mp3 at 256 kb/s. After Demucs:

- oracle(): identification with the true hit times (what a perfect drum
  transcription would give): is the true sample the library's top match?
- blind(): the whole pipeline with nothing known: grid, discover, detect,
  match; scores the drum transcription (F1 per sound) and the identification.

The audio stays local (the samples are yours); only the scores are kept.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audio import SR, read, samples, transpose, trim_onset, write
from .detect import detect, discover
from .grid import fit_grid
from .hits import group, template
from .library import Library

STEP_PATTERN = {  # 32 sixteenths = two bars
    "kick": [0, 7, 10, 16, 23, 26, 30],
    "snare": [8, 24],
    "open_hat": [14],
}
NOTES_808 = [0, 0, 3, 0, 1, -2, 7]  # with the kick steps
LEVELS = {"kick": 0.7, "snare": 0.55, "hat": 0.22, "open_hat": 0.18, "808": 0.8, "melodic": 0.22, "vocal": 0.35}
LENGTH_MS = {"kick": 200, "snare": 150, "hat": 80, "open_hat": 150, "808": 500}
BAND = {"kick": "kick", "snare": "snare", "hat": "hat", "open_hat": "hat", "808": "808"}


def _pick(rows, rng, roles, lo, hi):
    pool = [r for r in rows if r["role"] in roles and r["duration"] and lo <= r["duration"] <= hi]
    if not pool:
        raise ValueError(f"No {roles} between {lo} and {hi} s in the inventory")
    while True:
        r = rng.choice(pool)
        try:
            x = read(r["path"])
        except Exception:
            continue
        if np.max(np.abs(x)) > 0.05:
            return r["path"], trim_onset(x) / np.max(np.abs(x))


def build_beat(rows: list[dict], seed: int, out_dir: str | Path, bpm: float = 136.0, bars: int = 16) -> dict:
    """mix.wav (and mix_mp3.wav) plus truth.json in out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    step = 60 / bpm / 4
    n = int((bars * 16 * step + 2) * SR)
    sounds = {"kick": _pick(rows, rng, {"kick"}, 0.15, 1.5),
              "snare": _pick(rows, rng, {rng.choice(["snare", "clap"])}, 0.1, 1.5),
              "hat": _pick(rows, rng, {"hat"}, 0.03, 0.6),
              "open_hat": _pick(rows, rng, {"open_hat"}, 0.2, 1.5),
              "808": _pick(rows, rng, {"808"}, 0.8, 4.0)}
    melodic = _pick(rows, rng, {"melodic/loop"}, 8, 120)
    vocal = _pick(rows, rng, {"vox"}, 8, 300)
    tracks = {k: np.zeros(n, np.float32) for k in [*sounds, "melodic", "vocal"]}
    positions: dict[str, list] = {k: [] for k in sounds}

    def put(track, x, at_step, gain=1.0, length=None):
        a = int(round(at_step * step * SR))
        if length:
            x = x[:length].copy()
            if len(x) == length:
                x[-samples(5):] *= np.linspace(1, 0, samples(5))
        b = min(n, a + len(x))
        tracks[track][a:b] += gain * x[: b - a]

    for pair in range(bars // 2):
        o = pair * 32
        for s in STEP_PATTERN["kick"]:
            put("kick", sounds["kick"][1], o + s)
            positions["kick"].append((o + s) * step)
        for s in STEP_PATTERN["snare"]:
            put("snare", sounds["snare"][1], o + s)
            positions["snare"].append((o + s) * step)
        hats = set(range(0, 32, 2)) | ({27, 29} if pair % 2 else {13, 15, 29, 30, 31})
        for s in sorted(hats - {14}):
            put("hat", sounds["hat"][1], o + s, rng.uniform(0.55, 1.0))
            positions["hat"].append((o + s) * step)
        put("open_hat", sounds["open_hat"][1], o + 14, 0.9)
        positions["open_hat"].append((o + 14) * step)
        kicks = STEP_PATTERN["kick"]
        for j, (s, semis) in enumerate(zip(kicks, NOTES_808)):
            nxt = (kicks + [32])[j + 1]
            put("808", transpose(sounds["808"][1], semis), o + s, length=int((nxt - s) * step * SR))
            positions["808"].append(((o + s) * step, semis))
    tracks["melodic"] += np.tile(melodic[1], n // len(melodic[1]) + 1)[:n]
    voc = vocal[1][int(5 * SR):] if len(vocal[1]) > n + 5 * SR else vocal[1]
    tracks["vocal"][: min(n, len(voc))] += voc[:n]
    mix = sum(LEVELS[k] * tracks[k] for k in tracks)
    mix *= 10 ** (-12 / 20) / np.sqrt(np.mean(mix ** 2))
    mix = np.tanh(mix * 10 ** (5 / 20)) * 0.95
    write(out / "mix.wav", np.stack([mix, mix], 1), peak=None)
    mixed = out / "mix.wav"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mixed), "-b:a", "256k", str(out / "mix.mp3")],
                       check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "mix.mp3"), "-ar", str(SR),
                        str(out / "mix_mp3.wav")], check=True)
        mixed = out / "mix_mp3.wav"
    truth = {"seed": seed, "bpm": bpm, "sounds": {k: v[0] for k, v in sounds.items()},
             "melodic": melodic[0], "vocal": vocal[0], "positions": positions, "mix": str(mixed),
             "mix_rms_db": round(float(20 * np.log10(np.sqrt(np.mean(mix ** 2)))), 1)}
    (out / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    return truth


def _rank(lib: Library, ranked: list[dict], path: str, length: int) -> int | None:
    true_i = lib.index.get(path)
    if true_i is None:
        return None
    return next((k + 1 for k, r in enumerate(ranked) if lib.same(r["i"], true_i, length)), None)


def oracle(beat_dir: str | Path, lib: Library, lib808: Library | None = None) -> list[dict]:
    """Identification at the true hit times: rank of the true sample, its NCC,
    the top match and its margin over the next different sound."""
    beat = Path(beat_dir)
    truth = json.loads((beat / "truth.json").read_text())
    drums, bass = read(beat / "stems" / "drums.wav"), read(beat / "stems" / "bass.wav")
    out = []
    jobs = [(name, drums, lib, [int(round(t * SR)) for t in truth["positions"][name]], (0,))
            for name in ("kick", "snare", "hat", "open_hat")]
    if lib808 is not None:
        jobs.append(("808", bass, lib808,
                     [int(round(t * SR)) for t, st in truth["positions"]["808"] if st == 0], range(-12, 13)))
    for name, x, library, starts, semis in jobs:
        g = group(x, starts=starts, threshold=-1.0, cluster_ms=LENGTH_MS[name] // 2, min_db=-80)[0]
        t, used = template(x, g, LENGTH_MS[name])
        ranked = library.match(t, BAND[name], semitones=semis, top=0, prefilter=60)
        rank = _rank(library, ranked, truth["sounds"][name], len(t))
        top = ranked[0]
        runner = next(r for r in ranked[1:] if not library.same(r["i"], top["i"], len(t)))
        out.append({"beat": beat.name, "sound": name, "hits": used, "rank": rank,
                    "ncc_true": ranked[rank - 1]["ncc"] if rank else None,
                    "ncc_top": top["ncc"], "margin": round(top["ncc"] - runner["ncc"], 3),
                    "correct": rank == 1})
    return out


def _f1(found: list[float], truth: list[float], tolerance: float = 0.02) -> dict:
    """Precision/recall/F1 of hit times, after removing a constant offset: the
    truth is where the sample file starts, a detection where the attack is
    (a sample with 20 ms of quiet lead-in is found 20 ms late, every time)."""
    found, truth = sorted(found), sorted(truth)
    offset = 0.0
    if found and truth:
        d = [x - truth[int(np.argmin(np.abs(np.array(truth) - x)))] for x in found]
        close = [v for v in d if abs(v) < 0.06]
        offset = float(np.median(close)) if close else 0.0
    used, tp = set(), 0
    for f in found:
        j = next((j for j, t in enumerate(truth)
                  if j not in used and abs(t - (f - offset)) <= tolerance), None)
        if j is not None:
            used.add(j)
            tp += 1
    p = tp / len(found) if found else 0.0
    r = tp / len(truth) if truth else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "offset_ms": round(offset * 1000, 1)}


def blind(beat_dir: str | Path, lib: Library) -> dict:
    """The whole pipeline knowing nothing: grid, templates, hits, matches;
    scored against truth.json."""
    beat = Path(beat_dir)
    truth = json.loads((beat / "truth.json").read_text())
    drums = read(beat / "stems" / "drums.wav")
    grid, _ = fit_grid(drums, (100, 180))
    templates = discover(drums, grid)
    events = detect(drums, templates)
    sounds = ("kick", "snare", "hat", "open_hat")
    assigned: dict[str, list[str]] = {s: [] for s in sounds}
    per_template = []
    for tp in templates:
        times = [e["t"] for e in events[tp.name]]
        overlap = {s: _f1(times, truth["positions"][s])["precision"] for s in sounds}
        best = max(overlap, key=overlap.get)
        if overlap[best] >= 0.5:
            assigned[best].append(tp.name)
        from .detect import LENGTH_MS as MATCH_MS
        ranked = lib.match(tp.audio[: samples(MATCH_MS[tp.band])], tp.band, semitones=(0,), top=30)
        top = ranked[0]
        true_path = truth["sounds"].get(best)
        n = samples(MATCH_MS[tp.band])
        identified = bool(true_path in lib.index and lib.same(lib.index[true_path], top["i"], n))
        runner = next((r for r in ranked[1:] if not lib.same(r["i"], top["i"], n)), ranked[-1])
        per_template.append({"template": tp.name, "hits": len(times), "is": best if overlap[best] >= 0.5 else None,
                             "precision_vs_that_sound": overlap[best], "ncc_top": top["ncc"],
                             "margin": round(top["ncc"] - runner["ncc"], 3), "identified": identified})
    transcription = {}
    for s in sounds:
        scores = [_f1([e["t"] for e in events[name]], truth["positions"][s]) for name in assigned[s]]
        found = [e["t"] - sc["offset_ms"] / 1000 for name, sc in zip(assigned[s], scores) for e in events[name]]
        transcription[s] = {**_f1(found, truth["positions"][s]), "templates": assigned[s]}
    return {"beat": beat.name, "grid_bpm": grid.bpm, "true_bpm": truth["bpm"], "transcription": transcription,
            "templates": per_template}
