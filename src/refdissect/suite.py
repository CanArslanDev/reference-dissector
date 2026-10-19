"""Calibration v2 as one suite: every scenario beat separated, analysed,
matched, planned and scored, so a change to any step shows up per scenario.

    build()   the scenario beats (scenarios.build) and the v1 rebuilds
              (scenarios.convert_beat) as root/<scenario>-<seed>/
    run()     per beat: Demucs once per model (stems-<model>/, kept), then
              analyze, match, fit-808 and plan into runs/<tag>/, scored by
              evaluate against the truth at analysis level (what the pipeline
              heard) and at plan level (what Live would play): scores.json.
              Their means per scenario go to root/scores-<tag>.json.
    report()  markdown tables of those means per scenario, with the deltas
              between tags.

A beat folder after a run:
    events.json, mix*, stems_truth/   the truth (scenarios)
    stems-<model>/                    Demucs' stems and separation.json
    runs/<tag>/                       analysis.json, matches.json, synth808.json, plan.json, midi/,
                                      templates/, log.txt, scores.json

Everything under root is made of your library and stays local: the stems,
the plans and the scores files name files and hold note times.
scripts/export_results.py publishes the per-scenario means only (docs/09).
"""

from __future__ import annotations

import functools
import hashlib
import json
import multiprocessing
import re
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import __version__, evaluate, pipeline
from . import scenarios as S
from .render import render_plan
from .separate import DEFAULT_MODEL, separate_many

SEP_SEED, SEP_SHIFTS = 0, 1  # every suite separation: one seeded shift, Demucs' own default count
ORDER = [*S.SCENARIOS, "v1orig"]


# ── Beats ──

def _meta(beat: Path) -> dict:
    return json.loads((beat / "events.json").read_text(encoding="utf-8"))


def beat_dirs(root: str | Path, scenarios=None) -> list[Path]:
    """The beat folders under root (those with an events.json) of the given
    scenarios (all when None), in scenario order, then by seed."""
    beats = [d for d in Path(root).iterdir() if (d / "events.json").is_file()]
    picked = []
    for d in beats:
        ev = _meta(d)
        if not scenarios or ev["scenario"] in scenarios:
            picked.append((_rank(ev["scenario"]), ev["seed"], d.name, d))
    return [d for *_, d in sorted(picked)]


def _rank(scenario: str) -> tuple[int, str]:
    return (ORDER.index(scenario) if scenario in ORDER else len(ORDER), scenario)


def has_patterns(midi_dir: str | Path | None) -> bool:
    """Whether midi_dir holds what the producer scenario needs: at least one
    single-role hat pattern (scenarios.producer_files)."""
    return bool(midi_dir and Path(midi_dir).expanduser().is_dir() and S.producer_files(midi_dir)["hat"])


def build(rows: list[dict], root: str | Path, scenarios=None, seeds=(1, 2), midi_dir: str | Path | None = None,
          v1_truth: str | Path | None = None, bars: int | None = None, log=print) -> list[Path]:
    """Every scenario (all of scenarios.SCENARIOS when None) at every seed
    into root/<scenario>-<seed>/, and v1orig-N from v1_truth's beatN folders
    (the v1 beats rebuilt with truth stems) when given and "v1orig" is asked
    for or nothing is named. A beat already built is kept. producer is left
    out without single-role producer MIDI under midi_dir. bars (a multiple of
    4) shortens the scenario beats, for tests."""
    root = Path(root)
    names = list(scenarios or S.SCENARIOS)
    unknown = [n for n in names if n not in S.SCENARIOS and n != "v1orig"]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)} (known: {', '.join(ORDER)})")
    if "producer" in names and not has_patterns(midi_dir):
        log(f"producer: no single-role hat pattern under {midi_dir}; left out")
        names.remove("producer")
    out = []
    for name in (n for n in names if n != "v1orig"):
        for seed in seeds:
            beat = root / f"{name}-{seed}"
            if not (beat / "events.json").is_file():
                t = time.perf_counter()
                S.build(rows, name, seed, beat, midi_dir=midi_dir if name == "producer" else None, bars=bars)
                log(f"{beat.name}: built in {time.perf_counter() - t:.1f} s")
            out.append(beat)
    if v1_truth and (scenarios is None or "v1orig" in names):
        for src in sorted(Path(v1_truth).glob("beat*")):
            seed = int(re.sub(r"\D", "", src.name) or 0)
            beat = root / f"v1orig-{seed}"
            if (src / "truth.json").is_file() and not (beat / "events.json").is_file():
                S.convert_beat(src, beat, seed, rows)
                log(f"{beat.name}: converted")
            if (beat / "events.json").is_file():
                out.append(beat)
    return out


def source() -> dict:
    """The code a process loaded: the package version and, when it runs from
    a git checkout, the commit and whether the package's files differed from
    it (read once, at import: an edit made later does not change what ran)."""
    here = Path(__file__).parent
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True,
                             timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here, capture_output=True,
                               text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return {"version": __version__}
    if sha.returncode != 0:
        return {"version": __version__}
    return {"version": __version__, "commit": sha.stdout.strip(), "dirty": bool(dirty.stdout.strip())}


SOURCE = source()


# ── Separation, once per beat and model ──

def stems_dir(beat: str | Path, model: str) -> Path:
    return Path(beat) / f"stems-{model}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def separated(beat: Path, model: str) -> bool:
    """Whether the beat's stems for model are there, made from its mix as it
    is now, with the suite's seed and shifts."""
    info = stems_dir(beat, model) / "separation.json"
    if not info.is_file() or not (stems_dir(beat, model) / "drums.wav").is_file():
        return False
    s = json.loads(info.read_text(encoding="utf-8"))
    mix = beat / _meta(beat)["audio"]["mix"]
    return s.get("seed") == SEP_SEED and s.get("shifts") == SEP_SHIFTS and s.get("audio_sha256") == _sha256(mix)


def separate_beats(beats: list[Path], model: str, log=print) -> int:
    """Demucs on the mix of each beat (audio.mix: the decoded mp3, what the
    pipeline hears) that has no stems for model yet, in one worker: one model
    load, each job seeded alike, so a beat's stems do not depend on the others.
    Returns how many were separated."""
    jobs = [(b / _meta(b)["audio"]["mix"], stems_dir(b, model)) for b in beats if not separated(b, model)]
    if jobs:
        log(f"{model}: separating {len(jobs)} beats")
        separate_many(jobs, model, seed=SEP_SEED, shifts=SEP_SHIFTS,
                      log=lambda s: log(f"  {Path(s['audio']).parent.name}: {s['seconds']} s on {s['device']}"))
    return len(jobs)


# ── One beat ──

@functools.cache
def _library(path: str):
    from .library import Library
    return Library.load(path)  # about 0.5 GB: once per process


def _stopwatch(seconds: dict):
    last = [time.perf_counter()]

    def lap(name: str) -> None:
        now = time.perf_counter()
        seconds[name] = round(now - last[0], 2)
        last[0] = now
    return lap


def _pipeline(beat: Path, work: Path, model: str, library, library_808, options: dict, lap) -> tuple[dict, dict]:
    """analyze, match, fit-808 and plan, as the commands run them. A beat
    whose 808 never rings alone gets no fitted one-shot (fit_808's
    ValueError), so its plan has no 808 track, as with the CLI."""
    problems = {}
    with open(work / "log.txt", "w", encoding="utf-8", buffering=1) as f:  # line by line, to follow a run
        def log(msg) -> None:
            f.write(f"{msg}\n")
        pipeline.analyze(stems_dir(beat, model), work, log=log, **options.get("analyze", {}))
        lap("analyze")
        lib = _library(str(library))
        lap("library_load")
        pipeline.match(work, lib, log=log, **options.get("match", {}))
        lap("match")
        try:
            pipeline.fit_808(work, library_808=_library(str(library_808)) if library_808 else None, log=log,
                             **options.get("fit_808", {}))
        except ValueError as e:
            problems["fit_808"] = str(e)
        lap("fit_808")
        plan = pipeline.plan(work, log=log)
        lap("plan")
    return plan, problems


def _plan_score(truth: dict, beat: Path, work: Path, plan: dict, render: bool, lap) -> tuple[dict, list]:
    """The plan scored as Live would play it. With render, render_plan's
    tracks give the level scores, at the gain that undoes the plan's headroom
    (every track sits headroom_db under the stems it replaces), and are
    deleted after."""
    est = evaluate.estimate_from_plan(plan, root=work, scenario=truth["scenario"])
    missing = est["generator"]["switches"]["missing"]
    if not render:
        return evaluate.score(truth, est, truth_dir=beat), missing
    out = work / "render"
    render_plan(plan, out, root=work)
    lap("render")
    gain = 10 ** ((plan.get("levels") or {}).get("headroom_db", 0.0) / 20)
    try:
        return evaluate.score(truth, est, out, beat, gain), missing
    finally:
        shutil.rmtree(out, ignore_errors=True)


def run_beat(beat: str | Path, tag: str, model: str, library: str | Path, library_808: str | Path | None = None,
             render: bool = False, options: dict | None = None) -> dict:
    """The pipeline on one beat's stems for model (separated already) into
    runs/<tag>/ (cleared first), scored at analysis and plan level; the record
    is written to runs/<tag>/scores.json and returned. options: keyword
    arguments for the steps, {"analyze": {...}, "match": {...},
    "fit_808": {...}}, passed on unchanged."""
    beat = Path(beat).resolve()
    work = beat / "runs" / tag
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    options = options or {}
    truth = evaluate.load_events(beat / "events.json")
    sep = json.loads((stems_dir(beat, model) / "separation.json").read_text(encoding="utf-8"))
    seconds: dict = {"separate": sep.get("seconds")}
    lap = _stopwatch(seconds)
    plan, problems = _pipeline(beat, work, model, library, library_808, options, lap)
    analysis = evaluate.score(truth, evaluate.estimate_from_analysis(work, truth["scenario"]), truth_dir=beat)
    lap("score_analysis")
    planned, missing = _plan_score(truth, beat, work, plan, render, lap)
    lap("score_plan")
    seconds["total"] = round(sum(v for k, v in seconds.items() if k != "separate" and v is not None), 2)
    record = {"beat": beat.name, "scenario": truth["scenario"], "seed": truth["seed"], "tag": tag, "model": model,
              "separation": {k: sep.get(k) for k in ("model", "seed", "shifts", "device", "versions")},
              "options": options, "refdissect": SOURCE, "render": render, "seconds": seconds,
              "problems": problems, "missing": missing,
              "plan_size": {"pads": len(plan["tracks"]["drums"]["pads"]),
                            "drum_notes": len(plan["tracks"]["drums"]["notes"]),
                            "808_notes": len(plan["tracks"]["808"]["notes"]),
                            "808_one_shot": "sample_path" in plan["tracks"]["808"]},
              "analysis": analysis, "plan": planned}
    (work / "scores.json").write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    return record


def _run_job(args: tuple) -> dict:
    """run_beat for a pool: a beat that fails is a result too (written with
    its error), and the others go on."""
    try:
        return run_beat(*args)
    except Exception as e:
        beat, tag = Path(args[0]), args[1]
        record = {"beat": beat.name, "tag": tag, "failed": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-4000:]}
        (beat / "runs" / tag).mkdir(parents=True, exist_ok=True)
        (beat / "runs" / tag / "scores.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return record


def _done(record: dict, log) -> None:
    if "failed" in record:
        log(f"{record['beat']}: FAILED {record['failed']}")
        return
    a, p = record["analysis"]["summary"]["strict_f1_20"], record["plan"]["summary"]["strict_f1_20"]
    log(f"{record['beat']}: strict F1@20 analysis {_fmt(a)}, plan {_fmt(p)} ({record['seconds']['total']:.0f} s)")


def run(root: str | Path, library: str | Path, library_808: str | Path | None = None, tag: str = "base",
        model: str = DEFAULT_MODEL, scenarios=None, jobs: int = 3, render: bool = False,
        options: dict | None = None, resume: bool = False, log=print) -> dict:
    """Separate what is not separated yet (one Demucs process), then
    run_beat over the beats, `jobs` at a time in their own processes (each
    loads the library once, about 0.5 GB), and write the means per scenario
    to root/scores-<tag>.json (also returned). resume skips beats this tag
    has scored already."""
    root = Path(root).resolve()
    beats = beat_dirs(root, scenarios)
    if resume:
        beats = [b for b in beats if not (b / "runs" / tag / "scores.json").is_file()]
    t0 = time.perf_counter()
    separate_beats(beats, model, log)
    args = [(b, tag, model, str(library), str(library_808) if library_808 else None, render, options or {})
            for b in beats]
    if jobs > 1 and len(args) > 1:
        with ProcessPoolExecutor(min(jobs, len(args)), mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(_run_job, a) for a in args]):
                _done(future.result(), log)
    else:
        for a in args:
            _done(_run_job(a), log)
    summary = aggregate(root, tag)
    summary["last_run"] = {"beats": len(beats), "jobs": jobs, "wall_s": round(time.perf_counter() - t0, 1)}
    (root / f"scores-{tag}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


# ── Means per scenario ──

def _mean(values) -> float | None:
    v = [x for x in values if x is not None]
    return round(float(np.mean(v)), 4) if v else None


def _role_numbers(prefix: str, sounds: list[dict]) -> dict:
    errors = [s["error_ms"] for s in sounds if s.get("error_ms")]
    out = {f"{prefix}.f1_20": _mean(s["f1_20"] for s in sounds), f"{prefix}.f1_50": _mean(s["f1_50"] for s in sounds),
           f"{prefix}.v1_f1": _mean(s["v1"]["f1"] for s in sounds)}
    if errors:
        out |= {f"{prefix}.error_median_ms": _mean(e["median"] for e in errors),
                f"{prefix}.error_p90_ms": _mean(e["p90_abs"] for e in errors)}
    return out


def _by_role(sounds) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for s in sounds:
        out.setdefault(s["role"], []).append(s)
    return out


def _drums(d: dict) -> dict:
    out = {f"drums.{k}": v for k, v in d["macro"].items()}
    out["drums.extra"] = len(d["extra"])
    for role, sounds in _by_role(d["sounds"].values()).items():
        out |= _role_numbers(f"drums.{role}", sounds)
    return out


def _bass(b: dict) -> dict:
    out = {"808.f1_20": b["f1_20"], "808.f1_50": b["f1_50"], "808.v1_f1": b["v1"]["f1"],
           "808.note_f1_20": b["note_f1_20"], "808.note_f1_50": b["note_f1_50"]}
    if b.get("error_ms"):
        out |= {"808.error_median_ms": b["error_ms"]["median"], "808.error_p90_ms": b["error_ms"]["p90_abs"]}
    if b.get("cents"):
        out |= {"808.cents_median": b["cents"]["median"], "808.cents_median_abs": b["cents"]["median_abs"]}
    return out


def _rhythm(r: dict) -> dict:
    out = {"rhythm.meter_equal": float(r["meter"]["equal"])}
    if "tempo" in r:
        out |= {"rhythm.octave_x1": float(r["tempo"]["octave"] == "x1"),
                "rhythm.tempo_error_abs": abs(r["tempo"]["rel_error"])}
    if "downbeat" in r:
        out["rhythm.downbeat_error_ms_abs"] = abs(r["downbeat"]["first_error_ms"])
        if "f" in r["downbeat"]:
            out["rhythm.downbeat_f"] = r["downbeat"]["f"]
    if "beat_f" in r:
        out["rhythm.beat_f"] = r["beat_f"]
    if (r.get("swing") or {}).get("ratio_error") is not None:
        out["rhythm.swing_ratio_error"] = r["swing"]["ratio_error"]
    return out


def _levels(lv: dict) -> dict:
    out = {}
    measured = [s for s in lv["sounds"].values() if s and "level_db" in s]
    for role, sounds in _by_role(measured).items():
        for key in ("level_db", "balance_db", "spread_p90_db"):
            out[f"levels.{role}.{key}"] = round(float(np.median([s[key] for s in sounds])), 2)
    if measured:
        out["levels.balance_abs_db"] = round(float(np.median([abs(s["balance_db"]) for s in measured])), 2)
    return out


def headline(score: dict) -> dict[str, float]:
    """The numbers a suite compares, from one evaluate.score result, under
    flat keys; a part the truth (or the render) does not have is left out.
    Per drum role, the mean over the truth's sounds of that role (pitched
    copies and layers are sounds of their own); error_p90_ms is the mean of
    their p90 |error|."""
    s = score["summary"]
    out = {"strict_f1_20": s["strict_f1_20"], "v1_f1": s["v1_f1"]}
    if score.get("drums"):
        out |= _drums(score["drums"])
    if score.get("808"):
        out |= _bass(score["808"])
    if score.get("rhythm"):
        out |= _rhythm(score["rhythm"])
    if score.get("melodic"):
        out |= {"melodic.note_f1": score["melodic"]["pooled"]["note_f1"],
                "melodic.onset_f1": score["melodic"]["pooled"]["onset_f1"]}
    if score.get("chords"):
        out |= {f"chords.{r}": score["chords"][r] for r in evaluate.CHORD_RULES if score["chords"][r] is not None}
    if score.get("levels"):
        out |= _levels(score["levels"])
    return {k: v for k, v in out.items() if v is not None}


def _means(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    return {k: _mean(r.get(k) for r in rows) for k in keys}


def _distinct(values) -> list:
    """The different values among values (dicts compared by content)."""
    seen = {json.dumps(v, sort_keys=True): v for v in values}
    return [seen[k] for k in sorted(seen)]


def _scenario_means(records: list[dict]) -> dict:
    return {"beats": len(records), "seeds": sorted(r["seed"] for r in records),
            "analysis": _means([headline(r["analysis"]) for r in records]),
            "plan": _means([headline(r["plan"]) for r in records]),
            "seconds": _means([r["seconds"] for r in records])}


def aggregate(root: str | Path, tag: str) -> dict:
    """Means per scenario over the beats this tag has scored (a key over the
    beats that have it), and overall: the mean over scenarios, so each
    scenario counts once whatever its number of beats. options, render and
    refdissect (the code) list every value the records hold: one, unless a
    resumed run mixed them."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in Path(root).glob(f"*/runs/{tag}/scores.json")]
    ok = [r for r in records if "failed" not in r]
    groups: dict[str, list[dict]] = {}
    for r in sorted(ok, key=lambda r: (_rank(r["scenario"]), r["seed"])):
        groups.setdefault(r["scenario"], []).append(r)
    per = {name: _scenario_means(rs) for name, rs in groups.items()}
    overall = {part: _means([v[part] for v in per.values()]) for part in ("analysis", "plan", "seconds")}
    first = ok[0] if ok else {}
    return {"schema": "refdissect.suite/1", "tag": tag, "model": first.get("model"),
            "separation": {**(first.get("separation") or {}),
                           "devices": sorted({str(r["separation"].get("device")) for r in ok})},
            "options": _distinct(r.get("options") for r in ok), "render": _distinct(r.get("render") for r in ok),
            "refdissect": _distinct(r.get("refdissect") for r in ok),
            "beats": len(ok), "failed": sorted(r["beat"] for r in records if "failed" in r),
            "scenarios": per, "overall": {"scenarios": len(per), **overall}}


# ── Report ──

def _fmt(v, digits: int = 3) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def _delta(a, b, digits: int) -> str:
    return "–" if a is None or b is None else f"{b - a:+.{digits}f}"


TABLES = (  # (part, key, title, digits)
    ("plan", "strict_f1_20", "End to end: strict F1@20 of the plan (drum sounds and 808 onsets)", 3),
    ("analysis", "strict_f1_20", "Strict F1@20 of the analysis (drum sounds and 808 onsets)", 3),
    ("analysis", "drums.f1_20", "Drums, strict F1@20, analysis (mean over truth sounds)", 3),
    ("analysis", "drums.v1_f1", "Drums, v1 rule (one offset removed per sound), analysis", 3),
    ("plan", "drums.f1_20", "Drums, strict F1@20, plan", 3),
    ("analysis", "808.f1_20", "808 onsets, strict F1@20, analysis", 3),
    ("analysis", "808.note_f1_50", "808 note F1 (50 ms, 50 cents), analysis", 3),
    ("analysis", "808.cents_median_abs", "808 pitch, median |cents| of matched notes, analysis", 1),
    ("plan", "808.note_f1_50", "808 note F1 (50 ms, 50 cents), plan", 3),
    ("analysis", "rhythm.octave_x1", "Tempo in the right octave (share of beats)", 2),
    ("analysis", "rhythm.downbeat_error_ms_abs", "First downbeat, |error| modulo a bar (ms)", 1),
    ("analysis", "rhythm.beat_f", "Beat F (70 ms)", 3),
    ("analysis", "melodic.note_f1", "Melodic note F1 (where the truth has notes)", 3),
    ("analysis", "chords.majmin", "Chord accuracy, majmin (where the truth has chords)", 3),
    ("plan", "levels.balance_abs_db", "Levels: median |balance| over sounds, plan render (dB)", 2),
    ("seconds", "total", "Seconds per beat after separation", 0),
)
ROLE_TABLES = (("analysis", "f1_20", "Strict F1@20 per drum role, analysis", 3),
               ("plan", "f1_20", "Strict F1@20 per drum role, plan", 3),
               ("analysis", "error_p90_ms", "Timing p90 |error| per drum role, analysis (ms)", 1))


def _scenarios(aggs: dict) -> list[str]:
    names = {n for a in aggs.values() for n in a["scenarios"]}
    return sorted(names, key=_rank)


def _value(agg: dict, scenario: str, part: str, key: str):
    block = agg["overall"] if scenario == "overall" else agg["scenarios"].get(scenario)
    return (block or {}).get(part, {}).get(key)


def _table(aggs: dict, part: str, key: str, title: str, digits: int) -> list[str]:
    tags = list(aggs)
    rows = [n for n in [*_scenarios(aggs), "overall"] if any(_value(a, n, part, key) is not None
                                                            for a in aggs.values())]
    if not rows:
        return []
    head = ["scenario", *tags] + (["Δ"] if len(tags) > 1 else [])
    out = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for n in rows:
        vals = [_value(aggs[t], n, part, key) for t in tags]
        cells = [_fmt(v, digits) for v in vals] + ([_delta(vals[0], vals[-1], digits)] if len(tags) > 1 else [])
        out.append(f"| {'**overall**' if n == 'overall' else n} | " + " | ".join(cells) + " |")
    return out + [""]


def _role_table(aggs: dict, part: str, metric: str, title: str, digits: int) -> list[str]:
    tags = list(aggs)
    roles = [r for r in evaluate.DRUM_ROLES if any(_value(a, n, part, f"drums.{r}.{metric}") is not None
                                                   for a in aggs.values() for n in _scenarios(aggs))]
    if not roles:
        return []
    note = f" ({tags[-1]}; in brackets: minus {tags[0]})" if len(tags) > 1 else ""
    out = [f"### {title}{note}", "", "| scenario | " + " | ".join(roles) + " |", "|" + "---|" * (len(roles) + 1)]
    for n in [*_scenarios(aggs), "overall"]:
        cells = []
        for r in roles:
            vals = [_value(aggs[t], n, part, f"drums.{r}.{metric}") for t in tags]
            cell = _fmt(vals[-1], digits)
            if len(tags) > 1 and vals[-1] is not None and vals[0] is not None:
                cell += f" ({vals[-1] - vals[0]:+.{digits}f})"
            cells.append(cell)
        out.append(f"| {'**overall**' if n == 'overall' else n} | " + " | ".join(cells) + " |")
    return out + [""]


def report(root: str | Path, tags: list[str], out: str | Path | None = None) -> str:
    """Markdown: per scenario, each headline number for every tag (from
    root/scores-<tag>.json) and the change from the first tag to the last."""
    root = Path(root)
    aggs = {t: json.loads((root / f"scores-{t}.json").read_text(encoding="utf-8")) for t in tags}
    lines = [f"# Suite report: {', '.join(tags)}", ""]
    for t, a in aggs.items():
        sep = a.get("separation") or {}
        lines.append(f"- **{t}**: {a['beats']} beats in {len(a['scenarios'])} scenarios, model {a.get('model')} "
                     f"(seed {sep.get('seed')}, shifts {sep.get('shifts')}, {', '.join(sep.get('devices', []))}), "
                     f"options {json.dumps(a.get('options'))}" + (f"; failed: {', '.join(a['failed'])}"
                                                                    if a.get("failed") else ""))
    lines.append("")
    for part, key, title, digits in TABLES:
        lines += _table(aggs, part, key, title, digits)
    for part, metric, title, digits in ROLE_TABLES:
        lines += _role_table(aggs, part, metric, title, digits)
    text = "\n".join(lines)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    return text
