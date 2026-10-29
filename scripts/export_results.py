"""Turn local work folders into the public results/ and figures/.

    uv run python scripts/export_results.py WORK_ROOT
    uv run python scripts/export_results.py --suite SUITE_DIR TAG [TAG ...]

WORK_ROOT holds work/calibration/results.json, work/library*.npz and
<song>/refdissect/ (as in the README). Nothing
audio goes out, and files from a personal sample library are named by role
and a keyed hash (Ableton's Core Library keeps its names): the numbers are
the point, not where the sounds came from. The key is made on the first
export in WORK_ROOT/work/export-key and stays there, like the work folder.

SUITE_DIR is a calibration-v2 suite (refdissect suite); its scores-<tag>.json
becomes results/calibration/suite-<tag>.json with the means per scenario
only: no beat, file, path, note or event time.
"""

import hashlib
import hmac
import json
import re
import secrets
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from refdissect.figures import calibration_scatter, spectrogram_rows
from refdissect.library import Library
from refdissect.synth808 import Params, laid_out

ROOT = Path(__file__).resolve().parents[1]
CORE = "/Core Library/"
KEY = Path("work") / "export-key"  # under WORK_ROOT: the secret behind public_name's hashes, never published


def export_key(work: Path) -> bytes:
    """The key of public_name's hashes, from WORK_ROOT/work/export-key: 32
    random bytes made on the first export and read back after, so a file
    keeps its name from one export to the next."""
    path = work / KEY
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def public_name(path: str, role: str, key: bytes) -> str:
    """A library file as results/ names it: a Core Library file by its own
    name, a personal library's by role and the first 8 hex digits of an
    HMAC-SHA256 of its path under the export key. A plain hash names
    nothing: SHA-1 of the path (or of the bare file name), cut to 6 digits,
    gave back all 27 files results/piramit/summary.json named, from a list
    of the library's 11,990 files."""
    if CORE in path:
        return "Core Library: " + Path(path).name
    if not key:
        raise ValueError("a personal library's file is named with the export key")
    return f"user library {role or 'sample'} #{hmac.new(key, path.encode(), hashlib.sha256).hexdigest()[:8]}"


def library_path(npz: Path, name: str) -> str:
    """The path of a library file kept by its bare name (synth808.json names
    the closest library 808 so), looked up in the library index's paths;
    the name itself without the index."""
    if npz.is_file():
        with np.load(npz) as data:
            for p in data["paths"]:
                if Path(str(p)).name == name:
                    return str(p)
    return name


def calibration(work: Path) -> dict:
    r = json.loads((work / "work" / "calibration" / "results.json").read_text())
    out = {"oracle": r["oracle"], "blind": r["blind"], "summary": {}}
    for snd in ("kick", "snare", "hat", "open_hat", "808"):
        rows = [o for o in r["oracle"] if o["sound"] == snd]
        out["summary"][snd] = {"oracle_found": f"{sum(o['correct'] for o in rows)}/{len(rows)}",
                               "oracle_true_ncc": [o["ncc_true"] for o in rows],
                               "oracle_margins": [o["margin"] for o in rows]}
        if snd != "808":
            f = [b["transcription"][snd] for b in r["blind"]]
            ident = [t for b in r["blind"] for t in b["templates"] if t["is"] == snd]
            out["summary"][snd].update(
                blind_f1=[x["f1"] for x in f], blind_f1_mean=round(float(np.mean([x["f1"] for x in f])), 2),
                blind_identified=f"{sum(t['identified'] for t in ident)}/{len(ident)} templates")
    (ROOT / "results" / "calibration").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "calibration" / "summary.json").write_text(json.dumps(out, indent=1))
    calibration_scatter(r["oracle"], ROOT / "figures" / "calibration_identification.png")
    return out


# cache and cached: whether the run kept results and how many beats read each back (their steps' seconds are
# out of the means), so a published file says why its seconds lack them
SUITE_KEEP = ("schema", "tag", "model", "options", "steps", "reuse", "render", "refdissect", "cache", "cached",
              "beats", "last_run")
SCENARIO_KEEP = ("beats", "seeds", "analysis", "plan", "seconds")
LOCAL = re.compile(r"/Users/|/home/|/Volumes/|/private/|/tmp/|/Applications/|~/|[A-Za-z]:\\\\|"
                   r"\.(wav|flac|mp3|m4a|ogg|opus|aac|aiff?|mid|midi|npy|npz|als|alc|adg|adv)\b", re.IGNORECASE)
WORD = re.compile(r"[A-Za-z0-9_.+:-]{1,64}")  # a model, a curve, a version: no path, no file name with spaces


def plain(value, where: str = "options") -> None:
    """Refuse anything in value but numbers, booleans, null and short
    identifier strings (keys too), so that an option a later step adds can
    not carry a path or a file name into results/."""
    if value is None or isinstance(value, bool | int | float):
        return
    if isinstance(value, str):
        if not WORD.fullmatch(value):
            raise ValueError(f"{where}: {value!r} is not a short identifier")
    elif isinstance(value, list | tuple):
        for i, v in enumerate(value):
            plain(v, f"{where}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            plain(k, f"{where} key")
            plain(v, f"{where}.{k}")
    else:
        raise ValueError(f"{where}: {type(value).__name__} is not exported")


def suite_scores(root: Path, tag: str) -> dict:
    """results/calibration/suite-<tag>.json from root/scores-<tag>.json: how
    the run was made (model, seed, shifts, versions, options, the steps run,
    the tag the others were reused from and the results read back) and the
    means per scenario and overall. The run's description must be plain(), and nothing may look
    like a local path or an audio, MIDI, array or Live file name."""
    agg = json.loads((root / f"scores-{tag}.json").read_text(encoding="utf-8"))
    sep = agg.get("separation") or {}
    out = {k: agg.get(k) for k in SUITE_KEEP}
    out["separation"] = {k: sep.get(k) for k in ("model", "seed", "shifts", "devices", "versions")}
    out["failed"] = len(agg.get("failed") or [])
    for key in ("tag", "model", "options", "steps", "reuse", "render", "refdissect", "cache", "cached", "separation"):
        plain(out[key], key)
    out["scenarios"] = {name: {k: v[k] for k in SCENARIO_KEEP} for name, v in agg["scenarios"].items()}
    out["overall"] = agg["overall"]
    text = json.dumps(out, indent=1)
    if LOCAL.search(text):
        raise ValueError(f"suite-{tag}.json would name a local file: {LOCAL.search(text).group(0)!r}")
    path = ROOT / "results" / "calibration" / f"suite-{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    return out


def song(work: Path, name: str) -> dict:
    w = work / name / "refdissect"
    a = json.loads((w / "analysis.json").read_text())
    m = json.loads((w / "matches.json").read_text())
    s = json.loads((w / "synth808.json").read_text()) if (w / "synth808.json").is_file() else None
    p = json.loads((w / "plan.json").read_text()) if (w / "plan.json").is_file() else None
    notes = a["notes_808"]
    key = export_key(work)
    sounds = {}
    for k, t in a["templates"].items():
        mm = m.get(k, {})
        sounds[k] = {"found_in_band": t["band"], "hits_in_template": t["template_hits"],
                     "detected_hits": t["detected_hits"], "clean_share": t.get("clean"),
                     "grid_slots_of_template_hits": t["slots"],
                     "peak_ncc_histogram_0.3_to_1.0": t["peak_ncc_histogram_0.3_to_1.0"],
                     "descriptors": t["describe"]}
        if mm:
            sounds[k]["match"] = {
                "role": mm["role"], "band": mm["matched_in_band"], "verdict": mm["verdict"],
                "same_as": mm.get("same_as"), "margin": mm["margin"],
                "top": [{"sample": public_name(x["path"], x["role"], key), "ncc": x["ncc"],
                         "semitones": x["semitones"]} for x in mm["top"][:3]],
                "per_hit_wins": list(mm["per_hit"]["wins"].values()) if mm.get("per_hit") else None,
                "null_test_db": list(mm["null_test_db"].values())}
    grid = {k: v for k, v in a["grid"].items() if k != "tempo_map"}  # beat times stay local (docs/09)
    out = {"song": name, "grid": grid, "tuning": a["tuning"], "duration_s": a["source"]["duration_s"],
           "sections": a["sections"], "sounds": sounds,
           "808": {"notes": len(notes), "by_pitch": dict(Counter(n["pitch"] for n in notes).most_common()),
                   "slides_over_1.5_semitones": sum(abs(n["slide"]) >= 1.5 for n in notes)}}
    if s:
        lib = s["compare"].get("best_library")
        if lib:  # named by its path, as the drums' files are
            lib = dict(lib, file=public_name(library_path(work / "work" / "library_808.npz", lib["file"]), "808", key))
        out["808"]["synth"] = {"params": s["params"], "glide_semitones": s["glide_semitones"],
                               "pitch_hz": s["pitch_hz"], "fit": s["fit"], "vs_reference": s["compare"]["synth"],
                               "best_library_808_vs_reference": lib}
    if p:
        out["plan"] = {"tempo": p["tempo"], "drum_notes": len(p["tracks"]["drums"]["notes"]),
                       "pads": {k: {"role": v["role"], "sample": public_name(v["path"], v["role"], key),
                                    "transpose": v["transpose"], "ncc": v["ncc"], "verdict": v["verdict"]}
                                for k, v in p["tracks"]["drums"]["pads"].items()},
                       "808_notes": len(p["tracks"]["808"]["notes"]), "levels": p["levels"],
                       "locators": p["locators"]}
    (ROOT / "results" / name.lower()).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / name.lower() / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def figure(work: Path, name: str, library: Path, library_808: Path) -> None:
    w = work / name / "refdissect"
    m = json.loads((w / "matches.json").read_text())
    lib = Library.load(library)
    rows = []
    pick = [(k, v) for k, v in m.items() if not v.get("same_as")]
    pick.sort(key=lambda kv: -kv[1]["top"][0]["ncc"])
    for k, v in pick[:2]:
        t = np.load(w / "templates" / f"{k}.npy")
        top = v["top"][0]
        i = lib.index[top["path"]]
        ms = {"hat": 100, "snare": 300, "kick": 300}.get(v["matched_in_band"], 300)
        c = lib.candidate(i, len(t), top["semitones"])
        hz = (1000, 16000) if v["matched_in_band"] == "hat" else (100, 16000)
        rows.append({"title": f"{k} ({v['role']})", "hz": hz, "ms": ms, "difference": True,
                     "cells": [(f"{name} (averaged hits)", t),
                               (f"library, {top['semitones']:+.0f} st, NCC {top['ncc']:.2f}", c)]})
    s = json.loads((w / "synth808.json").read_text())
    ref = np.load(w / "templates" / "808.npy")
    p = Params(**s["params"])
    rows.append({"title": "808", "hz": (20, 400), "ms": 600, "difference": False,
                 "cells": [(f"{name} (averaged notes)", ref), ("synth (fitted)", laid_out(p, len(ref)))]})
    if library_808 and s["compare"].get("best_library"):
        lib8 = Library.load(library_808)
        best = next(i for i, path in enumerate(lib8.paths) if Path(path).name == s["compare"]["best_library"]["file"])
        rows[-1]["cells"].append(("closest library 808", lib8.candidate(best, len(ref), 0.0)))
    spectrogram_rows(rows, ROOT / "figures" / f"{name.lower()}_sounds.png")


if __name__ == "__main__" and sys.argv[1] == "--suite":
    root = Path(sys.argv[2]).expanduser()
    for tag in sys.argv[3:]:
        out = suite_scores(root, tag)
        print(f"{tag}: {out['beats']} beats, {len(out['scenarios'])} scenarios -> results/calibration/suite-{tag}.json")
elif __name__ == "__main__":
    work = Path(sys.argv[1]).expanduser()
    c = calibration(work)
    print(json.dumps(c["summary"], indent=1))
    for name in ("PIRAMIT",):
        out = song(work, name)
        print(name, out["grid"]["bpm"], out["tuning"], len(out["sounds"]), "sounds")
        figure(work, name, work / "work" / "library.npz", work / "work" / "library_808.npz")
