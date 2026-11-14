"""A plan as a blueprint (version 1): one JSON document that ableton-live-mcp's
build_from_blueprint builds in an empty Live set.

The blueprint holds what live-build sends Live, decided here without Live:
the tempo, the meter, the locators, the build_tracks spec (live.build_spec)
and the files to copy into the User Library under the names live-build
copies them by (sample_808, pad_sample, scaled by a pad's file_db). Its
format is ableton-live-mcp's docs/blueprint-v1.md and its JSON Schema,
schemas/blueprint-v1.schema.json (a copy, with the commit it came from, in
tests/fixtures). The tracks are build_spec's, but for how a sampler names
its sample: "file", a key of files, in place of "sample" (the copy's name),
so the build copies and checks the file itself. live-build itself is
unchanged.

    refdissect blueprint work/song                       (work/song/blueprint.json)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from . import __version__
from .live import build_spec, meter_of, one_shot_digest
from .render import pad_settings

FORMAT = "ableton-mcp-blueprint"
VERSION = 1
CHOKE_GROUPS = range(1, 17)  # a Drum Rack chain's choke group in Live: 1 to 16 (0 none)
VOICES = range(1, 33)  # a Simpler's voices


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
    the files live-build copies, under the same names (gain_db: a pad's
    file_db, where it is under 0). Every file gets its SHA-1's start, so the
    build refuses a file that changed after this. Locators at whole beats,
    as live-build places them. Checks are left out (all on). ValueError for
    what the blueprint cannot hold (a choke group past 16, a locator at or
    past the end)."""
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
                track["instrument"] = {"file": "808", **instrument}
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
                instrument["drum_rack"][note] = {"file": key, **sampler}
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
