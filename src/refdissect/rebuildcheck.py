"""A rebuild checked against its reference the same way on both sides
(`refdissect check-rebuild`).

symmetric.compare measures two stem sets that went through the same
separation; this module makes the rebuild's set. The plan is rendered
offline (render.render_plan: the pads and the 808 as Live plays them, with
the reference stems Live plays beside them at their fader), mixed, mastered
the way the reference is (mastering.master_like: its loudness, tilt and
peak ceiling, measured off the reference mix; optionally the same mp3 round
trip), and separated by the same Demucs model, seed and shifts as the
reference's stems. The separation's artifacts are then on both sides.

What it reports:
- descriptors (symmetric.measure) before and after: "before" is the
  rebuild's own clean tracks (drums, 808 as the bass stem, the reference
  stems it plays) at the master's gain against the reference's separated
  stems, the comparison without this command; "after" is separated against
  separated. Each against the reference's own spread (symmetric.baseline).
- null tests per stem, before and after: the residual (dB re the
  reference stem) after the least-squares gain, on the waveform and on STFT
  magnitudes. A stem the rebuild plays from the reference (other, vocals)
  nulls only as far as mastering and a second separation let it: the
  measure's floor.
- with a truth (a suite beat's events.json and stems_truth/), null tests per
  sound: each truth drum sound against the rendered pads that play it
  (evaluate's assignment, as the suite's level scores use it) and the truth
  808 against the 808 track, before any mastering (the truth's stems are
  clean). A sound no pad plays nulls at 0 dB.

Null at the least-squares gain is 10 log10(1 - r^2), r the normalized
correlation: level errors do not count (level_db reports them), timing,
pitch and timbre do. The waveform null is strict (0.1 ms off at 10 kHz is a
quarter cycle); the magnitude null (46 ms frames) forgives offsets under a
frame and keeps what is heard.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import SR, read
from .mastering import encode, master_like, profile
from .render import render_plan, resolve
from .symmetric import baseline, compare, load_stems, measure

N_FFT, HOP = 2048, 512


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


def _pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(a), len(b))
    return np.pad(a, (0, n - len(a))), np.pad(b, (0, n - len(b)))


def _db(x: float) -> float:
    return round(float(10 * np.log10(max(x, 1e-12))), 2)


def null(reference: np.ndarray, rebuild: np.ndarray) -> dict:
    """The residual of reference - g * rebuild re reference (dB), g the
    least-squares gain, on the waveform (null_db) and on STFT magnitudes
    (spectral_null_db, with its own gain), and level_db, the rebuild's level
    re the reference at the magnitude gain (-20 log10 g: the waveform gain
    goes to 0 where the two do not correlate, as hats 0.2 ms apart do). A
    silent rebuild nulls at 0 dB with no level; a silent reference has no
    null (None)."""
    import scipy.signal as ss
    r, e = _pair(_mono(reference), _mono(rebuild))
    rr, ee = float(r @ r), float(e @ e)
    if rr <= 0:
        return {"null_db": None, "spectral_null_db": None, "level_db": None}
    if ee <= 0:
        return {"null_db": 0.0, "spectral_null_db": 0.0, "level_db": None}
    g = float(r @ e) / ee
    out = {"null_db": _db(float(np.sum((r - g * e) ** 2)) / rr)}
    _, _, sr_ = ss.stft(r, nperseg=N_FFT, noverlap=N_FFT - HOP)
    _, _, se = ss.stft(e, nperseg=N_FFT, noverlap=N_FFT - HOP)
    mr, me = np.abs(sr_).ravel(), np.abs(se).ravel()
    gm = float(mr @ me) / float(me @ me)
    out["spectral_null_db"] = _db(float(np.sum((mr - gm * me) ** 2)) / float(mr @ mr))
    out["level_db"] = round(float(-20 * np.log10(gm)), 2) if gm > 0 else None
    return out


def rebuild_tracks(plan: dict, render_dir: str | Path, manifest: dict, n: int) -> dict[str, np.ndarray]:
    """What Live plays, per group, as (n, 2): drums (the pads' sum), bass
    (the 808 track) and every reference stem the plan does not mute, at its
    fader (render's tracks already carry theirs)."""
    render_dir = Path(render_dir)

    def fit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x = np.stack([x, x], 1) if x.ndim == 1 else x
        return np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n]

    out = {"drums": fit(sf.read(render_dir / manifest["drums"])[0])}
    if "808" in manifest["tracks"]:
        out["bass"] = fit(sf.read(render_dir / manifest["tracks"]["808"])[0])
    levels = plan.get("levels") or {}
    gain = 10 ** (min(6.0, levels.get("stems_track_db", 0.0)) / 20)
    for name, stem in (plan.get("stems") or {}).items():
        if not stem.get("mute") and stem.get("path"):
            x = read(stem["path"], stereo=True).T
            out[name] = out.get(name, 0) + fit(x) * gain
    return out


def sound_nulls(truth: dict, plan: dict, render_dir: str | Path, root: str | Path | None,
                truth_dir: str | Path, render_gain: float = 1.0) -> dict:
    """Per truth sound with a stem: null() of its stem against the rendered
    tracks of the pads assigned to it (evaluate's assignment of the plan's
    estimate), the 808 against the 808 track, and the drums group against
    the pads' sum. balance_db is each sound's level_db minus their median
    (as the suite's level scores' balance)."""
    from .evaluate import _drum_assignment, estimate_from_plan
    render_dir, truth_dir = Path(render_dir), Path(truth_dir)
    tracks = json.loads((render_dir / "render.json").read_text(encoding="utf-8"))["tracks"]
    audio = truth.get("audio") or {}
    stems, gains = audio.get("stems") or {}, (audio.get("master") or {}).get("stems_gain") or {}
    est = estimate_from_plan(plan, root=root, scenario=truth.get("scenario", ""))
    t_times, _, assignment, extra = _drum_assignment(truth, est)

    def stem(key: str) -> np.ndarray:
        return read(resolve(stems[key], truth_dir)) / gains.get(key, 1.0)

    def rendered(ids: list[str]) -> np.ndarray:
        xs = [read(render_dir / tracks[e]) for e in ids if e in tracks]
        return sum(xs) * render_gain if xs else np.zeros(1)

    out: dict = {}
    for s in t_times:
        if not stems.get(s):
            continue
        ids = sorted(e for e, a in assignment.items() if a["to"] == s)
        out[s] = {"pads": ids, **null(stem(s), rendered(ids))}
    if truth.get("bass808") and stems.get("808"):
        out["808"] = {"pads": ["808"] if "808" in tracks else [], **null(stem("808"), rendered(["808"]))}
    levels = [v["level_db"] for v in out.values() if v["level_db"] is not None]
    for v in out.values():
        v["balance_db"] = round(v["level_db"] - float(np.median(levels)), 2) if v["level_db"] is not None else None
    group = None
    if stems.get("drums"):
        group = null(stem("drums"), read(render_dir / "drums.wav") * render_gain)
    return {"render_gain": render_gain, "sounds": out, "drums_group": group,
            "extra_pads": [e["sound"] for e in extra]}


def stem_nulls(reference: dict[str, np.ndarray], rebuild: dict[str, np.ndarray]) -> dict:
    return {k: null(reference[k], rebuild[k]) for k in sorted(set(reference) & set(rebuild))}


def _summary(rows: dict) -> dict:
    beyond = sorted(k for k, v in rows.items() if v.get("beyond_spread"))
    return {"measures": sum(1 for v in rows.values() if "beyond_spread" in v), "beyond_spread": len(beyond),
            "beyond": beyond}


def descriptor_shift(before: dict, after: dict) -> dict:
    """Per measure in both: the difference before and after, and the
    measures that leave or join the beyond-spread list."""
    rows = {k: {"before": before[k]["diff"], "after": after[k]["diff"]}
            for k in sorted(set(before) & set(after))}
    b, a = set(_summary(before)["beyond"]), set(_summary(after)["beyond"])
    return {"rows": rows, "only_before": sorted(b - a), "only_after": sorted(a - b)}


def _cut(stems: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {k: np.pad(x, ((0, max(0, n - len(x))), (0, 0)))[:n] for k, x in stems.items()}


def prepare(plan: dict, root: str | Path | None, reference_mix: str | Path, out: str | Path,
            mp3_kbps: int | None = None, lookahead_ms: float = 5.0) -> dict:
    """Render the plan to out/render, mix what Live plays, master it like
    the reference mix (and through mp3 at mp3_kbps when given) to
    out/rebuild_master.wav. Returns the mastering report."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ref = read(reference_mix, stereo=True).T.astype(np.float64)
    n = len(ref)
    manifest = render_plan(plan, out / "render", root=root, n_samples=n)
    tracks = rebuild_tracks(plan, out / "render", manifest, n)
    mix = sum(tracks.values())
    master, report = master_like(mix, SR, profile(ref, SR), lookahead_ms)
    if mp3_kbps:
        master = encode(master, SR, mp3_kbps)
        report["mp3_kbps"] = mp3_kbps
        report["after_codec"] = profile(master, SR)
    sf.write(out / "rebuild_master.wav", master.astype(np.float32), SR, subtype="FLOAT")
    report["missing"] = manifest.get("missing", [])
    (out / "master.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def evaluate_rebuild(plan: dict, root: str | Path | None, reference_stems: str | Path, out: str | Path,
                     rebuild_stems: str | Path | None = None, kick_gap: float = 0.2,
                     truth: dict | None = None, truth_dir: str | Path | None = None) -> dict:
    """After prepare() and the separation of out/rebuild_master.wav into
    rebuild_stems: the descriptors before and after, the null tests per stem
    and, with a truth, per sound. Writes out/check.json and returns it."""
    out = Path(out)
    report = json.loads((out / "master.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "render" / "render.json").read_text(encoding="utf-8"))
    ref = load_stems(reference_stems)
    n = min(len(x) for x in ref.values())
    ref = _cut(ref, n)
    gain = 10 ** (report.get("gain_db", 0.0) / 20)
    clean = {k: x * gain for k, x in rebuild_tracks(plan, out / "render", manifest, n).items()}
    spread = baseline(ref, kick_gap)
    m_ref = measure(ref, kick_gap)
    before = compare(m_ref, measure(clean, kick_gap), spread)
    result: dict = {"master": report, "reference_spread": spread,
                    "before": {"summary": _summary(before), "rows": before,
                               "stem_nulls": stem_nulls(ref, clean)}}
    if rebuild_stems is not None:
        new = _cut(load_stems(rebuild_stems), n)
        after = compare(m_ref, measure(new, kick_gap), spread)
        result["after"] = {"summary": _summary(after), "rows": after, "stem_nulls": stem_nulls(ref, new)}
        result["shift"] = descriptor_shift(before, after)
    if truth is not None:
        headroom = (plan.get("levels") or {}).get("headroom_db", 0.0)
        result["sounds"] = sound_nulls(truth, plan, out / "render", root, truth_dir or out,
                                       10 ** (headroom / 20))
    (out / "check.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def separation_settings(stems: str | Path) -> dict:
    """The model, seed and shifts the reference's stems were made with
    (their separation.json), so the rebuild goes through the same ones."""
    p = Path(stems) / "separation.json"
    if not p.is_file():
        return {}
    s = json.loads(p.read_text(encoding="utf-8"))
    return {k: s[k] for k in ("model", "seed", "shifts") if k in s}


def check_rebuild(plan: dict, root: str | Path | None, reference_mix: str | Path, reference_stems: str | Path,
                  out: str | Path, separate: Callable[[Path, Path], object] | None = None,
                  mp3_kbps: int | None = None, kick_gap: float = 0.2, truth: dict | None = None,
                  truth_dir: str | Path | None = None, rebuild_stems: str | Path | None = None) -> dict:
    """prepare(), separate(master, stems_dir) unless rebuild_stems are
    given (None and no separate: descriptors before only), evaluate_rebuild()."""
    out = Path(out)
    prepare(plan, root, reference_mix, out, mp3_kbps)
    if rebuild_stems is None and separate is not None:
        rebuild_stems = out / "stems"
        separate(out / "rebuild_master.wav", rebuild_stems)
    return evaluate_rebuild(plan, root, reference_stems, out, rebuild_stems, kick_gap, truth, truth_dir)

