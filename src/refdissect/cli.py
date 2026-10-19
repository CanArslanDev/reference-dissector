"""refdissect: take a reference track apart and rebuild it in Ableton Live.

    refdissect inventory ~/Music/Ableton/"User Library" --out work/inventory.json
    refdissect index-library work/inventory.json --out work/library.npz
    refdissect index-library work/inventory.json --808 --out work/library_808.npz
    refdissect separate song.mp3 --out work/song/stems
    refdissect analyze work/song/stems --work work/song
    refdissect melodic work/song                 (Basic Pitch in its own uv environment)
    refdissect match work/song --library work/library.npz
    refdissect fit-808 work/song --library-808 work/library_808.npz
    refdissect plan work/song
    refdissect live-build work/song          (needs the [live] extra and Live running)
    refdissect calibrate work/inventory.json --library work/library.npz --out work/calibration

  comparing and choosing (docs/11):
    refdissect compare ref/stems rebuild/stems --out compare.json   (both separated by the same model)
    refdissect listening-round ref.wav a.wav b.wav --question "Chords?" --out round1
    refdissect space stems/other.wav --spans 40-56 100-116           (width, and the band-pass between spans)
    refdissect mix-gains stems/ --targets targets.json               (fader moves without re-recording)
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


def _melodic(a):
    from .pipeline import melodic
    melodic(a.work, a.stems, a.min_hz, a.snap_ms, not a.no_vote)


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


def _spans(values):
    out = []
    for v in values or []:
        a, b = v.split("-")
        out.append((float(a), float(b)))
    return out


def _compare(a):
    from .symmetric import baseline, compare, load_stems, measure
    span = _spans([a.span])[0] if a.span else None
    ref, new = load_stems(a.reference, span), load_stems(a.rebuild, span)
    spread = baseline(ref, a.kick_gap)
    rows = compare(measure(ref, a.kick_gap), measure(new, a.kick_gap), spread)
    if a.out:
        Path(a.out).write_text(json.dumps({"rows": rows, "reference_spread": spread}, indent=1), encoding="utf-8")
    beyond = {k: v for k, v in rows.items() if v.get("beyond_spread")}
    for k, v in sorted(beyond.items(), key=lambda kv: -abs(kv[1]["diff"]))[:a.top]:
        print(f"{k:45s} reference {v['reference']:9.2f}  rebuild {v['rebuild']:9.2f}  diff {v['diff']:+8.2f}")
    for k, v in rows.items():
        if k.endswith("curve_distance_db"):
            print(f"{k:45s} {v['diff']:.2f} dB")
    print(f"{len(beyond)} of {len(rows)} measures differ by more than the reference's own halves do")


def _round(a):
    from .rounds import listening_round
    span_r = _spans([a.reference_span])[0] if a.reference_span else None
    span_c = _spans([a.candidate_span])[0] if a.candidate_span else None
    m = listening_round(a.reference, a.candidates, a.out, a.question, span_r, span_c, a.lufs)
    print(f"{len(m['files'])} files -> {a.out} (manifest.md says what each number is)")


def _space(a):
    import soundfile as sf

    from .space import band_correlation, fit_bandpass, median_spectrum_db
    x, sr = sf.read(a.audio, always_2d=True)
    spans = _spans(a.spans) or [(0.0, len(x) / sr)]
    out = {"width": {}}
    for s0, s1 in spans:
        seg = x[int(s0 * sr):int(s1 * sr)]
        if seg.shape[1] >= 2:
            out["width"][f"{s0:g}-{s1:g}"] = band_correlation(seg, sr)
    if len(spans) >= 2:
        (a0, a1), (b0, b1) = spans[:2]
        f, da = median_spectrum_db(x[int(a0 * sr):int(a1 * sr)].mean(1), sr)
        _, db = median_spectrum_db(x[int(b0 * sr):int(b1 * sr)].mean(1), sr)
        out["second_vs_first"] = fit_bandpass(f, db - da, sr=sr)
    print(json.dumps(out, indent=1))


def _gains(a):
    from .mixsim import gains_to_targets
    from .symmetric import load_stems
    targets = json.loads(Path(a.targets).read_text(encoding="utf-8"))
    stems = load_stems(a.stems)
    spans = {k: _spans(v.get("spans")) for k, v in targets.items() if isinstance(v, dict) and v.get("spans")}
    levels = {k: (v["target_db"] if isinstance(v, dict) else float(v)) for k, v in targets.items()}
    from .audio import SR
    print(json.dumps(gains_to_targets(stems, levels, SR, spans, a.measure), indent=1))


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
    s = sub.add_parser("melodic", help="notes per tonal stem and chords, with Basic Pitch (after analyze)")
    s.add_argument("work")
    s.add_argument("--stems", default=None, help="stems folder (default: the one analyze used)")
    s.add_argument("--min-hz", type=float, default=None,
                   help="lowest note frequency (default: just above the 808's highest note)")
    s.add_argument("--snap-ms", type=float, default=None, help="snap onsets to the 16th grid within this many ms")
    s.add_argument("--no-vote", action="store_true", help="no voting across loop repeats")
    s.set_defaults(run=_melodic)
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
    s = sub.add_parser("compare", help="measure a rebuild and its reference the same way (both separated)")
    s.add_argument("reference", help="the reference's stems folder")
    s.add_argument("rebuild", help="the rebuild's stems folder, from the same separation model")
    s.add_argument("--span", default=None, help="seconds a-b to compare (default: all)")
    s.add_argument("--out", default=None)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--kick-gap", type=float, default=0.2,
                   help="seconds between kicks at least (raise it when other attacks bleed into the drums)")
    s.set_defaults(run=_compare)

    s = sub.add_parser("listening-round", help="numbered reference-beep-candidate files to choose by ear")
    s.add_argument("reference")
    s.add_argument("candidates", nargs="+")
    s.add_argument("--question", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--reference-span", default=None, help="seconds a-b")
    s.add_argument("--candidate-span", default=None, help="seconds a-b")
    s.add_argument("--lufs", type=float, default=-14.0)
    s.set_defaults(run=_round)

    s = sub.add_parser("space", help="width per octave, and the band-pass between two spans")
    s.add_argument("audio", help="a stereo stem")
    s.add_argument("--spans", nargs="*", help="seconds a-b (two of them: the band-pass from the first to the second)")
    s.set_defaults(run=_space)

    s = sub.add_parser("mix-gains", help="the gain per recorded stem that reaches a target level")
    s.add_argument("stems", help="folder of recorded (post-fader) stems")
    s.add_argument("--targets", required=True, help='JSON: {"Kick": -12, "Pad": {"target_db": -20, "spans": ["8-24"]}}')
    s.add_argument("--measure", choices=["lufs", "rms"], default="lufs")
    s.set_defaults(run=_gains)

    a = p.parse_args(argv)
    a.run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
