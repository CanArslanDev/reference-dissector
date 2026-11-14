"""A plan as a blueprint (version 1): one JSON document that ableton-live-mcp's
build_from_blueprint builds in an empty Live set, and live-build, which
writes one and has build_from_blueprint build it (build).

The blueprint holds what live-build --direct sends Live, decided here
without Live: the tempo, the meter, the locators, the build_tracks spec
(live.build_spec) and the files to copy into the User Library under the
names live-build --direct copies them by (sample_808, pad_sample, scaled by
a pad's file_db). Its format is ableton-live-mcp's docs/blueprint-v1.md and
its JSON Schema, schemas/blueprint-v1.schema.json (a copy, with the commit
it came from, in tests/fixtures). The tracks are build_spec's, but for how a
sampler names its sample: "file", a key of files, in place of "sample" (the
copy's name), so the build copies and checks the file itself; and for what
loading a file by path leaves to Live's defaults (playback_mode, PAD_OFF).

    refdissect blueprint work/song                       (work/song/blueprint.json)
    refdissect live-build work/song [--bars A-B]         (the blueprint, built in Live)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import __version__
from .live import build_spec, meter_of, one_shot_digest, refused
from .render import pad_settings, settings_808

FORMAT = "ableton-mcp-blueprint"
VERSION = 1
CHOKE_GROUPS = range(1, 17)  # a Drum Rack chain's choke group in Live: 1 to 16 (0 none)
VOICES = range(1, 33)  # a Simpler's voices
# build_from_blueprint loads a file by path (Live 12.4+): a Simpler made with insert_device, then
# replace_sample, where live-build --direct dropped each file on its pad or track through Live's browser.
# What the browser left to Live's defaults the blueprint says itself: the playback mode render.py plays (a
# pad One-Shot, as Live loads a sample dropped on a Drum Rack pad; the 808 Classic: render.PAD, TRACK_808,
# a plan's own playback_mode first), and each pad's filter off (PAD_OFF), as the 808's is already
# (live.simpler_808): render.py plays every Simpler unfiltered
PAD_OFF = {"F On": "Off"}


def made_by() -> str:
    """"refdissect <version> <commit>": the commit of the checkout the code
    runs from, where it is one (with "+" when its files differ from it)."""
    root = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=7", "HEAD"], capture_output=True,
                                text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return f"refdissect {__version__}"
    return f"refdissect {__version__} {commit}{'+' if dirty else ''}"


def song_name(plan: dict, work: str | Path) -> str:
    """The song's name as the plan names the 808's one-shot (plan_808: the
    folder that holds the stems' folder), else the work folder's."""
    stem = next(iter((plan.get("stems") or {}).values()), None)
    folder = Path(stem["path"]).expanduser().absolute().parent.parent if stem else Path(work).expanduser().absolute()
    return folder.name


def check_files(plan: dict) -> None:
    """live-build's checks that need no Live: the 808's one-shot is the one
    the plan named (its digest in the name: fit-808 ran again after plan,
    and the plan's Transpose and Detune were made for the old one), and
    every pad's file can be read. RuntimeError otherwise, as live.build."""
    t808 = plan["tracks"]["808"]
    if "sample_path" in t808 and one_shot_digest(t808["sample_path"]) not in t808["sample"]:
        raise RuntimeError(f"{t808['sample_path']} changed after the plan named it {t808['sample']!r}: "
                           "run refdissect plan again")
    pads = plan["tracks"]["drums"]["pads"]
    unread = [f"pad {note}: {pad['path']}" for note, pad in pads.items() if not Path(pad["path"]).is_file()]
    if unread:
        raise RuntimeError(f"Cannot read the files of {'; '.join(unread)}")


def _absolute(path: str | Path) -> str:
    return str(Path(path).expanduser().absolute())


def make_blueprint(plan: dict, name: str, made: str | None = None) -> dict:
    """The plan as a blueprint: build_spec's tracks with each sampler's
    sample named by a key of files ("808", "pad-<note>"), whose entries copy
    the files live-build --direct copies, under the same names (gain_db: a
    pad's file_db, where it is under 0), and each such sampler's
    playback_mode, with "F On" off on every pad (PAD_OFF). Every file gets
    its SHA-1's start, so the build refuses a file that changed after this.
    Locators at whole beats, as live-build --direct places them. Checks are
    left out (all on). ValueError for what the blueprint cannot hold (a
    choke group past 16, a locator at or past the end)."""
    check_files(plan)
    spec = build_spec(plan)
    t808, pads = plan["tracks"]["808"], plan["tracks"]["drums"]["pads"]
    files: dict = {}
    problems = []
    for track in spec:
        instrument = track.get("instrument")
        if track["name"] == "808 (synth)":
            if "sample_path" in t808:
                files["808"] = {"source": _absolute(t808["sample_path"]), "name": t808["sample"],
                                "sha1": one_shot_digest(t808["sample_path"])}
                instrument.pop("sample")
                track["instrument"] = {"file": "808", **instrument,
                                       "playback_mode": settings_808(t808)["playback_mode"]}
            voices = track.pop("voices", None)
            if voices is not None:  # build_spec's None: the device's own
                if int(voices) not in VOICES:
                    problems.append(f"808 (synth): {voices} voices (a Simpler has 1-32)")
                track["voices"] = int(voices)
        elif isinstance(instrument, dict) and "drum_rack" in instrument:
            for note, sampler in instrument["drum_rack"].items():
                pad = pads[note]
                gain = float(pad_settings(pad)["file_db"])
                key = f"pad-{note}"
                files[key] = {"source": _absolute(pad["path"]), "name": sampler.pop("sample"),
                              **({"gain_db": gain} if gain < 0 else {}), "sha1": one_shot_digest(pad["path"])}
                sampler["settings"] = {**sampler["settings"], **PAD_OFF}
                instrument["drum_rack"][note] = {"file": key, **sampler,
                                                 "playback_mode": pad_settings(pad)["playback_mode"]}
                if "choke_group" in sampler and sampler["choke_group"] not in CHOKE_GROUPS:
                    problems.append(f"pad {note}: choke group {sampler['choke_group']} (Live's are 1-16)")
        for clip in track.get("audio_clips") or []:
            clip["file"] = _absolute(clip["file"])
    length = float(plan["length_beats"])
    locators = [{"name": str(n), "beat": float(round(b))} for n, b in plan["locators"]]
    problems += [f"locator {loc['name']!r} at beat {loc['beat']:g}: at or past the end ({length:g})"
                 for loc in locators if loc["beat"] >= length]
    if problems:
        raise ValueError("The plan does not make a blueprint: " + "; ".join(problems))
    out = {"format": FORMAT, "version": VERSION, "name": name, "made_by": made or made_by(),
           "tempo": float(plan["tempo"]), "time_signature": meter_of(plan), "length_beats": length,
           "locators": locators, "files": files, "tracks": spec}
    return out


def write(work: str | Path, out: str | Path | None = None, log=print) -> Path:
    """WORK/plan.json (or a plan.json given as WORK) as a blueprint, written
    to out (default: blueprint.json beside the plan). meta records the
    plan's SHA-1, which the build ignores."""
    path = Path(work)
    plan_file = path / "plan.json" if path.is_dir() else path
    data = plan_file.read_bytes()
    plan = json.loads(data)
    folder = plan_file.parent
    blueprint = make_blueprint(plan, song_name(plan, folder))
    blueprint["meta"] = {"plan": plan_file.name, "plan_sha1": hashlib.sha1(data).hexdigest()[:12]}
    target = Path(out) if out else folder / "blueprint.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(blueprint, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    pads = sum(1 for k in blueprint["files"] if k.startswith("pad-"))
    scaled = sum(1 for f in blueprint["files"].values() if "gain_db" in f)
    log(f"blueprint: {len(blueprint['tracks'])} tracks, {pads} pads ({scaled} scaled), "
        f"{'an 808 one-shot, ' if '808' in blueprint['files'] else ''}{len(blueprint['locators'])} locators, "
        f"{blueprint['time_signature']} at {blueprint['tempo']:g} BPM: {target}")
    return target


# ── live-build: the blueprint, built by build_from_blueprint (needs the [live] extra) ──

def _key(name: str) -> str:
    """build_from_blueprint's name of a sampler ("808 (synth)", "Drums
    (library) pad 36") as live-build --direct's report names it ("808",
    "36")."""
    return "808" if name == "808 (synth)" else name.rsplit(" pad ", 1)[-1]


def as_live_build(found: dict, blueprint: dict, log=print) -> dict:
    """build_from_blueprint's report in live-build --direct's keys, each
    difference logged as live-build --direct logs it: tracks (its rows) and
    problems per track, locators (how many Live placed) and locators_moved,
    choke_groups per pad (plan, live, same), volumes_held and
    volumes_not_held (a pad that held its Volume only once sent again as
    text: text and held_by_text), samples_checked and wrong_samples ("plan":
    the copy in the User Library the Simpler is to hold), 808_voices; and
    what --direct did not check or report: time_signature (Live's against
    the plan's), files (the copies), excerpt and ok, false where anything
    differs. The whole report is under build_from_blueprint."""
    checks = found.get("checks") or {}
    report: dict = {"tracks": found.get("tracks", [])}
    problems = {t["track"]: t["problems"] for t in report["tracks"] if t.get("problems")}
    if problems:
        report["problems"] = problems
    if "locators" in found:
        report["locators"] = found["locators"]
    for o in found.get("locators_moved") or []:
        where = "made none" if o.get("live") is None else f"put it at beat {o['live']:g}"
        log(f"Live {where} for locator {o['name']!r} at beat {o['asked']}"
            + ("" if o.get("live") is None else ": deleted" if o.get("deleted") else ": left there"))
    if found.get("locators_moved"):
        report["locators_moved"] = found["locators_moved"]
    pads = {str(note): pad for t in blueprint["tracks"] if isinstance(t.get("instrument"), dict)
            for note, pad in (t["instrument"].get("drum_rack") or {}).items()}
    groups = {note: int(p["choke_group"]) for note, p in pads.items() if p.get("choke_group")}
    if groups and "choke_groups" in checks:
        wrong = {_key(k): v for k, v in (checks["choke_groups"].get("wrong") or {}).items()}
        report["choke_groups"] = {
            note: ({"plan": wrong[note].get("want", g), "live": wrong[note].get("live"), "same": False,
                    **({"error": wrong[note]["error"]} if "error" in wrong[note] else {})}
                   if note in wrong else {"plan": g, "live": g, "same": True})
            for note, g in groups.items()}
        for note, c in report["choke_groups"].items():
            if not c["same"]:
                log(f"pad {note} is in choke group {c['live']}, not the plan's {c['plan']}"
                    + (f" ({c['error']})" if "error" in c else ""))
    volumes = checks.get("volumes") or {}
    report["volumes_held"] = volumes.get("held", 0)
    unheld = {}
    for name in volumes.get("held_by_text") or []:
        note = _key(name)
        unheld[note] = {"plan": (pads.get(note, {}).get("settings") or {}).get("Volume"), "text": True,
                        "held_by_text": True}
        log(f"pad {note} did not hold Volume {unheld[note]['plan']} dB sent as a number: sent as text, "
            "which it holds")
    for name, row in (volumes.get("not_held") or {}).items():
        note = _key(name)
        unheld[note] = {"plan": row.get("want"), **{k: row[k] for k in ("value", "display", "error", "then",
                                                                         "text_error") if k in row},
                        "text": "then" in row}
        log(f"pad {note} holds Volume {row.get('value')} ({row.get('display')}), not the plan's {row.get('want')} dB"
            + (": sent as text" if "then" in row else ""))
    if unheld:
        report["volumes_not_held"] = unheld
    samples = checks.get("samples") or {}
    report["samples_checked"] = samples.get("checked", 0)
    wrong = {_key(k): {"plan": v.get("want"), "live": v.get("live"), "same": False,
                       **({"error": v["error"]} if "error" in v else {})}
             for k, v in (samples.get("wrong") or {}).items()}
    if wrong:
        report["wrong_samples"] = wrong
        for key, h in wrong.items():
            log(f"{'the 808' if key == '808' else f'pad {key}'} holds {h['live']}, not the plan's {h['plan']}"
                + (f" ({h['error']})" if "error" in h else ""))
    voices = (checks.get("voices") or {}).get("808 (synth)")
    if voices is not None:
        report["808_voices"] = voices
        if not voices.get("same"):
            log(f"the 808 has {voices.get('live')} voices, not the plan's {voices.get('want')}")
    signature = checks.get("time_signature")
    if signature is not None:
        report["time_signature"] = signature
        if not signature.get("same"):
            log(f"Live's time signature is {signature.get('live')}, not the plan's {signature.get('want')}")
    for key in ("files", "excerpt", "ok"):
        if key in found:
            report[key] = found[key]
    report["build_from_blueprint"] = found
    return report


async def build(work: str | Path, bars: tuple[int, int] | None = None, stems_out: str | Path | None = None,
                log=print) -> dict:
    """live-build: WORK's plan as a blueprint (write: WORK/blueprint.json,
    or beside a plan.json given as WORK), built in the open, empty Live set
    by ableton-live-mcp's build_from_blueprint, called in-process over the
    server's own connection to Live (its full profile, set here before the
    server is imported). With bars (a, b), build_from_blueprint builds bars
    a to b from 0 and cuts the stems to them into excerpt-a-b/ beside the
    blueprint, named by their content; with stems_out too, the excerpt is
    made here by ableton-live-mcp's own blueprint.excerpt, its stems cut
    into stems_out, and built as it is. After it, as live-build --direct
    did: the Arrangement's loop off and the 808's track selected (the
    playhead at 0 and the Arrangement shown are build_from_blueprint's).
    Returns its report in live-build --direct's keys (as_live_build). A
    connection Live refuses, and a build that stops, end in a RuntimeError
    that says why (live.refused; build_from_blueprint names the tracks it
    left in the set)."""
    try:
        path = write(work, log=log)
    except ValueError as e:
        raise ValueError(f"{e} (live-build --direct builds such a plan the way it did before)") from e
    blueprint = json.loads(path.read_text(encoding="utf-8"))
    os.environ["ABLETON_MCP_PROFILE"] = "full"  # build_from_blueprint is a tool of the full profile
    try:
        from MCP_Server.server import mcp
    except ImportError as e:
        raise RuntimeError("Building in Live needs ableton-live-mcp: "
                           "pip install 'reference-dissector[live]' (and Live with its Remote Script)") from e

    async def call(tool_name, **args):
        r = await mcp.call_tool(tool_name, args)
        text = "".join(getattr(c, "text", "") for c in r.content)
        if getattr(r, "is_error", False):
            raise RuntimeError(f"{tool_name}: {text}")
        return text

    args: dict = {"blueprint": str(path)}
    folder = None
    if bars:
        a, b = (int(v) for v in bars)
        if stems_out is not None:
            from MCP_Server import blueprint as bp
            folder = Path(stems_out)
            args["blueprint"] = bp.excerpt(blueprint, (a, b), path.parent, folder)
        else:
            folder = path.parent / f"excerpt-{a}-{b}"
            args["bars"] = f"{a}-{b}"
    try:
        found = json.loads(await call("build_from_blueprint", **args))
    except Exception as e:
        text = str(e)
        if "Unknown tool" in text and "build_from_blueprint" in text:
            text += (" (ableton-live-mcp registers build_from_blueprint in its full profile: MCP_Server.server was "
                     "imported before ABLETON_MCP_PROFILE=full, or its version has no such tool)")
        raise RuntimeError(refused(RuntimeError(text))) from e
    excerpt = found.get("excerpt")
    if excerpt is None and isinstance(args["blueprint"], dict):  # made here (stems_out)
        excerpt = (args["blueprint"].get("meta") or {}).get("excerpt")
    if bars and excerpt:
        found.setdefault("excerpt", excerpt)
        log(f"excerpt: bars {excerpt['bars'][0]}-{excerpt['bars'][1]}, {excerpt['seconds'][0]:.2f}-"
            f"{excerpt['seconds'][1]:.2f} s of the plan, stems cut into {folder}")
    await call("set_song_properties", loop=False)
    overview = json.loads(await call("get_session_overview"))
    first = next((t["index"] for t in overview["tracks"] if t["name"] == "808 (synth)"), None)
    await call("select_in_live", view="arrangement", track=first or 0)
    report = as_live_build(found, blueprint, log)
    log(json.dumps(report, ensure_ascii=False, default=str))
    return report
