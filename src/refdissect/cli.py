"""refdissect: take a reference track apart and rebuild it in Ableton Live.

    refdissect inventory ~/Music/Ableton/"User Library" --out work/inventory.json
    refdissect index-library work/inventory.json --out work/library.npz
    refdissect index-library work/inventory.json --808 --out work/library_808.npz
    refdissect separate song.mp3 --out work/song/stems
    refdissect analyze work/song/stems --work work/song
    refdissect match work/song --library work/library.npz
    refdissect fit-808 work/song --library-808 work/library_808.npz
    refdissect plan work/song
    refdissect live-build work/song          (needs the [live] extra and Live running)
    refdissect calibrate work/inventory.json --library work/library.npz --out work/calibration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def _inventory(a):
    from .library import inventory, summarize
    rows = inventory(a.folders)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} files -> {a.out}")
    for source, roles in summarize(rows).items():
        print(f"  {source}: " + ", ".join(f"{r} {c[0]}" for r, c in sorted(roles.items(), key=lambda kv: -kv[1][0])))


def _index(a):
    from .library import Library, load_rows
    rows = load_rows(a.inventory)
    if a.is_808:
        lib = Library.build(rows, seconds=1.0, max_duration=6.0, roles={"808"})
    else:
        lib = Library.build(rows, seconds=a.seconds)
    lib.save(a.out)
    print(f"{len(lib)} sounds -> {a.out}")


def _separate(a):
    from .separate import separate
    print(json.dumps(separate(a.audio, a.out, a.model, a.device), indent=1))


def _analyze(a):
    from .pipeline import analyze
    analyze(a.stems, a.work, tuple(a.bpm_range))


def _match(a):
    from .library import Library
    from .pipeline import match
    match(a.work, Library.load(a.library), detune=a.detune)


def _fit(a):
    from .library import Library
    from .pipeline import fit_808
    fit_808(a.work, library_808=Library.load(a.library_808) if a.library_808 else None)


def _plan(a):
    from .pipeline import plan
    plan(a.work)


def _build(a):
    from .live import build
    path = Path(a.plan)
    plan = json.loads((path / "plan.json" if path.is_dir() else path).read_text(encoding="utf-8"))
    asyncio.run(build(plan, a.user_library))


def _calibrate(a):
    from .calibrate import blind, build_beat, oracle
    from .library import Library, load_rows
    from .separate import separate
    rows = load_rows(a.inventory)
    lib = Library.load(a.library)
    lib808 = Library.load(a.library_808) if a.library_808 else None
    out = Path(a.out)
    results = {"oracle": [], "blind": []}
    for seed in range(1, a.beats + 1):
        beat = out / f"beat{seed}"
        if not (beat / "truth.json").is_file():
            truth = build_beat(rows, seed, beat)
            print(f"beat {seed}: " + ", ".join(f"{k} {Path(v).name}" for k, v in truth["sounds"].items()))
        if not (beat / "stems" / "drums.wav").is_file():
            separate(json.loads((beat / "truth.json").read_text())["mix"], beat / "stems")
        results["oracle"] += oracle(beat, lib, lib808)
        results["blind"].append(blind(beat, lib))
        right = sum(r["correct"] for r in results["oracle"] if r["beat"] == beat.name)
        print(f"  oracle: {right}/5 found; blind F1: " + ", ".join(
            f"{s} {v['f1']}" for s, v in results["blind"][-1]["transcription"].items()))
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {out / 'results.json'}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="refdissect", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("inventory", help="list a sample library with drum roles")
    s.add_argument("folders", nargs="+")
    s.add_argument("--out", required=True)
    s.set_defaults(run=_inventory)
    s = sub.add_parser("index-library", help="cache the one-shots for matching")
    s.add_argument("inventory")
    s.add_argument("--out", required=True)
    s.add_argument("--seconds", type=float, default=0.6)
    s.add_argument("--808", dest="is_808", action="store_true", help="808s only, 1 s each")
    s.set_defaults(run=_index)
    s = sub.add_parser("separate", help="stems with Demucs (in its own uv environment)")
    s.add_argument("audio")
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="htdemucs")
    s.add_argument("--device", default=None)
    s.set_defaults(run=_separate)
    s = sub.add_parser("analyze", help="grid, tuning, drum templates and hits, 808 notes, sections")
    s.add_argument("stems")
    s.add_argument("--work", required=True)
    s.add_argument("--bpm-range", nargs=2, type=float, default=[100.0, 180.0])
    s.set_defaults(run=_analyze)
    s = sub.add_parser("match", help="the library's closest sound for each template")
    s.add_argument("work")
    s.add_argument("--library", required=True)
    s.add_argument("--detune", type=float, default=None, help="also try every sound shifted by this many semitones")
    s.set_defaults(run=_match)
    s = sub.add_parser("fit-808", help="synthesize the 808 and write a one-shot")
    s.add_argument("work")
    s.add_argument("--library-808", default=None)
    s.set_defaults(run=_fit)
    s = sub.add_parser("plan", help="notes in beats, sounds and levels for Live; MIDI files")
    s.add_argument("work")
    s.set_defaults(run=_plan)
    s = sub.add_parser("live-build", help="build the plan in the open (empty) Live set")
    s.add_argument("plan", help="plan.json or the work folder")
    s.add_argument("--user-library", default=str(Path.home() / "Music" / "Ableton" / "User Library"))
    s.set_defaults(run=_build)
    s = sub.add_parser("calibrate", help="synthetic beats from known samples, scored")
    s.add_argument("inventory")
    s.add_argument("--library", required=True)
    s.add_argument("--library-808", default=None)
    s.add_argument("--beats", type=int, default=6)
    s.add_argument("--out", required=True)
    s.set_defaults(run=_calibrate)
    a = p.parse_args(argv)
    a.run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
