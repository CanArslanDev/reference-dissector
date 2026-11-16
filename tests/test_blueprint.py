"""refdissect blueprint: a plan as ableton-live-mcp's blueprint v1, checked
against the format's JSON Schema (tests/fixtures/blueprint-v1.schema.json, a
copy whose $comment names the commit it came from). No Live, no Demucs, no
sample library: synthetic plans over files written here."""

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from refdissect.blueprint import as_live_build, make_blueprint, song_name, write
from refdissect.cli import main
from refdissect.live import build_spec, one_shot_digest, pad_sample, pad_volume, scale_files_to_full

SR = 44100
SCHEMA = json.loads((Path(__file__).parent / "fixtures" / "blueprint-v1.schema.json").read_text(encoding="utf-8"))


# ── The schema, without the jsonschema package (it is not a dependency) ──────

def _types(value) -> set[str]:
    if isinstance(value, bool):
        return {"boolean"}
    if isinstance(value, int):
        return {"integer", "number"}
    if isinstance(value, float):
        return {"number"} | ({"integer"} if value.is_integer() else set())
    return {{str: "string", dict: "object", list: "array", type(None): "null"}[type(value)]}


def schema_errors(value, schema: dict = SCHEMA, where: str = "blueprint") -> list[str]:
    """Every error of value against schema, for the keywords the blueprint's
    schema uses (Draft 2020-12): enough to hold a blueprint to it where the
    jsonschema package is not installed."""
    if "$ref" in schema:
        return schema_errors(value, SCHEMA["$defs"][schema["$ref"].rsplit("/", 1)[1]], where)
    errors = []
    if "type" in schema:
        allowed = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not _types(value) & set(allowed):
            return [f"{where}: {value!r} is not of type {allowed}"]
    if "const" in schema and value != schema["const"]:
        errors.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where}: {value!r} is not one of {schema['enum']}")
    if "oneOf" in schema:
        fits = [s for s in schema["oneOf"] if not schema_errors(value, s, where)]
        if len(fits) != 1:
            errors.append(f"{where}: fits {len(fits)} of oneOf's shapes, not 1")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{where}: too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{where}: {value!r} does not match {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where}: {value} < {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{where}: {value} > {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{where}: {value} <= {schema['exclusiveMinimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{where}: fewer than {schema['minItems']} items")
        for i, item in enumerate(value):
            if "items" in schema:
                errors += schema_errors(item, schema["items"], f"{where}[{i}]")
    if isinstance(value, dict):
        errors += [f"{where}: {k!r} is required" for k in schema.get("required", []) if k not in value]
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{where}: fewer than {schema['minProperties']} properties")
        props = schema.get("properties", {})
        for k, v in value.items():
            if "propertyNames" in schema:
                errors += schema_errors(k, schema["propertyNames"], f"{where} key {k!r}")
            if k in props:
                errors += schema_errors(v, props[k], f"{where}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{where}: {k!r} is not allowed")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors += schema_errors(v, schema["additionalProperties"], f"{where}.{k}")
    return errors


def cross_errors(bp: dict) -> list[str]:
    """What ableton-live-mcp's validator checks beyond the schema
    (MCP_Server.blueprint.validate): unique track names, every sampler's file a
    key of files, notes that start inside their clip, voices on a Simpler,
    locators before the end, one name per file content."""
    errors = []
    names = [t["name"] for t in bp["tracks"]]
    errors += [f"track name {n!r} twice" for n in set(names) if names.count(n) > 1]
    for t in bp["tracks"]:
        inst = t.get("instrument")
        samplers = (inst["drum_rack"].values() if isinstance(inst, dict) and "drum_rack" in inst
                    else [inst] if isinstance(inst, dict) else [])
        errors += [f"{t['name']}: file {s.get('file')!r}" for s in samplers if s.get("file") not in bp["files"]]
        errors += [f"{t['name']}: file and sample both" for s in samplers if "sample" in s or "sample_path" in s]
        if "voices" in t and not (isinstance(inst, dict) and "drum_rack" not in inst):
            errors.append(f"{t['name']}: voices without a Simpler")
        for c in t.get("clips") or []:
            late = [n["start_time"] for n in c["notes"] if n["start_time"] >= c["length"]]
            errors += [f"{t['name']}: a note at {at}" for at in late]
    errors += [f"locator {loc}" for loc in bp.get("locators") or [] if loc["beat"] >= bp["length_beats"]]
    by_name: dict = {}
    for f in bp["files"].values():
        by_name.setdefault(f["name"].lower(), set()).add((f["source"], f.get("gain_db", 0)))
    errors += [f"{n!r} names two files" for n, s in by_name.items() if len(s) > 1]
    return errors


def assert_valid(bp: dict) -> None:
    assert schema_errors(bp) == []
    assert cross_errors(bp) == []
    json.dumps(bp, allow_nan=False)  # plain JSON: no NaN or infinity


# ── Synthetic plans ──────────────────────────────────────────────────────────

def _wav(path: Path, x: np.ndarray, subtype: str = "PCM_16") -> str:
    sf.write(path, x, SR, subtype=subtype)
    return str(path)


def plan_for(tmp_path: Path, *, one_shot: bool = True, meter: tuple[int, int] = (4, 4)) -> dict:
    """A small plan as make_plan writes one: an 808 on a one-shot, three
    pads, a melodic track and four stems in <song>/stems."""
    rng = np.random.default_rng(0)
    lib = tmp_path / "library"
    lib.mkdir(parents=True)
    pads = {}
    for note, name, role in ((36, "kick", "kick"), (38, "snare one", "snare"), (42, "hat", "hat")):
        path = _wav(lib / f"{name}.wav", rng.standard_normal(SR // 8) * 0.3)
        pads[str(note)] = {"template": f"sound_{note}", "path": path, "transpose": 0, "ncc": 0.95,
                           "verdict": "same sample", "role": role, "vol_vel": 0.35, "volume_db": -3.0 - note / 10}
    pads["42"]["detune_cents"] = -8
    song = tmp_path / "Song"
    (song / "stems").mkdir(parents=True)
    stems = {s: {"path": _wav(song / "stems" / f"{s}.wav", rng.standard_normal(SR) * 0.1), "mute": s != "other"}
             for s in ("drums", "bass", "other", "vocals")}
    bar = meter[0] * 4 / meter[1]
    length = 4 * bar
    t808 = {"cents": 0.0, "detune_cents": -12, "root_midi": 24, "transpose": 36, "vol_vel": 0.35,
            "notes": [{"pitch": 24, "start_time": 0.0, "duration": 1.5, "velocity": 115},
                      {"pitch": 27, "start_time": bar, "duration": 1.0, "velocity": 100}]}
    if one_shot:
        shot = _wav(song / "808 (synth).wav", np.sin(2 * np.pi * 32.7 * np.arange(SR) / SR) * 0.9)
        t808 |= {"sample": f"Song 808 (synth) {one_shot_digest(shot)}.wav", "sample_path": shot}
    drums = [{"pitch": p, "start_time": t, "duration": 0.25, "velocity": 100}
             for p, t in ((36, 0.0), (38, 1.0), (42, 0.5), (36, bar), (38, bar + 1.0))]
    return {"tempo": 140.0, "beats_per_bar": meter[0], "beat_unit": meter[1], "length_beats": length,
            "tracks": {"drums": {"pads": pads, "notes": drums}, "808": t808,
                       "melodic": {"other": {"instrument": "Drift", "cents": 0,
                                             "notes": [{"pitch": 60, "start_time": 0.0, "duration": 2.0,
                                                        "velocity": 90}]}}},
            "stems": stems, "locators": [["A (1)", 0.053], ["B (3)", 2 * bar + 0.053]],
            "levels": {"headroom_db": 4.0, "drums_track_db": -12.6, "808_track_db": -13.9, "stems_track_db": -4.0}}


def as_build_spec(bp: dict) -> list[dict]:
    """The blueprint's tracks put back the way live-build --direct sends
    them: each sampler's file as the name it is copied under (sample),
    without the playback mode and the pads' "F On" that loading by path
    leaves to Live and the blueprint says (blueprint.PAD_OFF), and voices
    only where the plan has them."""
    tracks = json.loads(json.dumps(bp["tracks"]))
    for t in tracks:
        inst = t.get("instrument")
        if isinstance(inst, dict):
            for s in (inst["drum_rack"].values() if "drum_rack" in inst else [inst]):
                key = s.pop("file")
                s["sample"] = bp["files"][key]["name"]
                s.pop("playback_mode")
                if "drum_rack" in inst:
                    assert s["settings"].pop("F On") == "Off" and s["settings"].pop("Fade In") == 0.0
            if "drum_rack" not in inst:
                t["voices"] = t.get("voices")
    return tracks


def _same_as_live_build(plan: dict, bp: dict) -> None:
    spec = build_spec(plan)
    for t in spec:
        for c in t.get("audio_clips") or []:
            c["file"] = str(Path(c["file"]).absolute())
    assert as_build_spec(bp) == json.loads(json.dumps(spec))


# ── Cases ────────────────────────────────────────────────────────────────────

def test_a_plain_beat(tmp_path):
    plan = plan_for(tmp_path)
    bp = make_blueprint(plan, song_name(plan, tmp_path), made="refdissect test")
    assert_valid(bp)
    assert (bp["format"], bp["version"], bp["name"], bp["made_by"]) == (
        "ableton-mcp-blueprint", 1, "Song", "refdissect test")
    assert (bp["tempo"], bp["time_signature"], bp["length_beats"]) == (140.0, "4/4", 16.0)
    assert bp["locators"] == [{"name": "A (1)", "beat": 0.0}, {"name": "B (3)", "beat": 8.0}]
    assert set(bp) == {"format", "version", "name", "made_by", "tempo", "time_signature", "length_beats",
                       "locators", "files", "tracks"}  # no checks: all on
    t808 = plan["tracks"]["808"]
    assert bp["files"]["808"] == {"source": t808["sample_path"], "name": t808["sample"],
                                  "sha1": one_shot_digest(t808["sample_path"])}
    pad = plan["tracks"]["drums"]["pads"]["38"]
    assert bp["files"]["pad-38"] == {"source": pad["path"], "name": pad_sample(pad["path"]),
                                     "sha1": one_shot_digest(pad["path"])}
    tracks = {t["name"]: t for t in bp["tracks"]}
    assert list(tracks) == ["808 (synth)", "Drums (library)", "other (notes)", "REF drums", "REF bass",
                            "REF other", "REF vocals"]
    s808 = tracks["808 (synth)"]
    assert s808["instrument"] == {"file": "808", "warping": False, "playback_mode": "classic"} and "voices" not in s808
    assert (s808["settings"]["Transpose"], s808["settings"]["Detune"], s808["settings"]["Volume"]) == (
        36.0, -12.0, "0.0 dB")
    rack = tracks["Drums (library)"]["instrument"]["drum_rack"]
    assert rack["42"] == {"file": "pad-42", "warping": False, "playback_mode": "one_shot",
                          "settings": {"Volume": pad_volume(plan["tracks"]["drums"]["pads"]["42"]),
                                       "Transpose": 0.0, "Detune": -8.0, "Vol < Vel": 0.35, "Snap": "Off",
                                       "F On": "Off", "Fade In": 0.0}}
    assert all("choke_group" not in p for p in rack.values())
    assert tracks["REF other"]["audio_clips"][0]["file"] == plan["stems"]["other"]["path"]
    _same_as_live_build(plan, bp)


def test_a_choke_group_in_three_four(tmp_path):
    plan = plan_for(tmp_path, meter=(3, 4))
    plan["tracks"]["drums"]["pads"]["42"]["choke_group"] = 1
    plan["tracks"]["drums"]["pads"]["38"]["choke_group"] = 0  # none: left out
    bp = make_blueprint(plan, "Song", made="refdissect test")
    assert_valid(bp)
    assert bp["time_signature"] == "3/4" and bp["locators"][1]["beat"] == 6.0
    rack = bp["tracks"][1]["instrument"]["drum_rack"]
    assert rack["42"]["choke_group"] == 1 and "choke_group" not in rack["38"]
    _same_as_live_build(plan, bp)


def test_a_file_over_full_scale_is_copied_scaled(tmp_path):
    plan = plan_for(tmp_path)
    pad = plan["tracks"]["drums"]["pads"]["38"]
    x = np.random.default_rng(3).standard_normal(SR // 8)
    _wav(Path(pad["path"]), x / np.max(np.abs(x)) * 10 ** (4.23 / 20), subtype="FLOAT")  # +4.23 dBFS
    scale_files_to_full(plan)
    assert pad["file_db"] == -4.3
    bp = make_blueprint(plan, "Song", made="refdissect test")
    assert_valid(bp)
    entry = bp["files"]["pad-38"]
    assert entry["gain_db"] == -4.3 and entry["name"] == pad_sample(pad["path"], -4.3)
    assert entry["name"].endswith(" -4.30dB.wav")
    assert all("gain_db" not in f for k, f in bp["files"].items() if k != "pad-38")  # only under 0
    volume = bp["tracks"][1]["instrument"]["drum_rack"]["38"]["settings"]["Volume"]
    assert volume == pad_volume(pad) and volume <= 0 and not math.isnan(volume)
    _same_as_live_build(plan, bp)


def test_an_808_with_glide(tmp_path):
    plan = plan_for(tmp_path)
    plan["tracks"]["808"].update(voices=1, glide_mode="Glide", glide_time_ms=45.0)
    bp = make_blueprint(plan, "Song", made="refdissect test")
    assert_valid(bp)
    s808 = bp["tracks"][0]
    assert s808["voices"] == 1
    assert (s808["settings"]["Glide Mode"], s808["settings"]["Glide Time"]) == ("Glide", "45.0 ms")
    _same_as_live_build(plan, bp)


def test_a_plan_without_an_808_one_shot(tmp_path):
    """No one-shot: build_spec makes no 808 track, so no files["808"]."""
    plan = plan_for(tmp_path, one_shot=False)
    bp = make_blueprint(plan, "Song", made="refdissect test")
    assert_valid(bp)
    assert "808" not in bp["files"] and [t["name"] for t in bp["tracks"]][0] == "Drums (library)"
    _same_as_live_build(plan, bp)


def test_what_a_blueprint_cannot_hold(tmp_path):
    plan = plan_for(tmp_path)
    plan["tracks"]["drums"]["pads"]["42"]["choke_group"] = 17
    plan["tracks"]["808"]["voices"] = 33
    with pytest.raises(ValueError, match=r"33 voices.*choke group 17"):
        make_blueprint(plan, "Song")


def test_a_locator_at_the_end_is_left_out(tmp_path):
    """A section that starts in the plan's last half beat rounds to the end
    (16.0 here), where a blueprint holds no locator: it is left out and
    logged, and the blueprint is made (a job's last step does not fail)."""
    plan = plan_for(tmp_path)
    plan["locators"] += [["C (5)", 15.7], ["D (6)", 19.0]]
    logged = []
    bp = make_blueprint(plan, "Song", made="refdissect test", log=logged.append)
    assert_valid(bp)
    assert [loc["name"] for loc in bp["locators"]] == ["A (1)", "B (3)"]
    assert len(logged) == 2 and all("at or past the plan's end (16)" in line for line in logged)
    assert "'C (5)' (beat 15.7)" in logged[0] and "'D (6)' (beat 19)" in logged[1]


def test_a_plan_without_pads_has_no_drums_track(tmp_path):
    """build_spec's Drums track with an empty drum_rack: the schema refuses
    it and build_tracks would not make it a rack, so it is left out."""
    plan = plan_for(tmp_path)
    plan["tracks"]["drums"]["pads"] = {}
    plan["tracks"]["drums"]["notes"] = []
    bp = make_blueprint(plan, "Song", made="refdissect test")
    assert_valid(bp)
    assert "Drums (library)" not in [t["name"] for t in bp["tracks"]]
    assert not any(k.startswith("pad-") for k in bp["files"])


def test_two_locators_on_one_beat_keep_the_first(tmp_path):
    """Live keeps one cue per beat (ableton-live-mcp 2.4.0 refuses two):
    after rounding to whole beats the first stays, the other is logged."""
    plan = plan_for(tmp_path)
    beat = float(round(plan["locators"][1][1]))
    plan["locators"].insert(2, ["B2 (3)", beat + 0.3])
    logged = []
    bp = make_blueprint(plan, "Song", made="refdissect test", log=logged.append)
    assert_valid(bp)
    beats = [loc["beat"] for loc in bp["locators"]]
    assert len(beats) == len(set(beats)) == len(plan["locators"]) - 1
    assert "B2 (3)" not in [loc["name"] for loc in bp["locators"]]
    assert len(logged) == 1 and "'B2 (3)'" in logged[0] and "one cue per beat" in logged[0]


def test_made_by_outside_a_checkout(monkeypatch):
    """No .git: no git (it may open macOS's developer tools dialog), the
    commit from the install's direct_url.json, as uv --with <archive zip>
    writes it."""
    import subprocess

    from refdissect import blueprint

    commit = "21d82d72ee89e21b82a7bff4d8d8805af283bac9"

    class Dist:
        def __init__(self, found):
            self.found = found

        def read_text(self, name):
            return json.dumps(self.found) if name == "direct_url.json" else None

    def no_git(*a, **k):
        raise AssertionError("git ran")

    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: False if self.name == ".git" else real_exists(self))
    monkeypatch.setattr(subprocess, "run", no_git)
    import importlib.metadata as md
    for found, want in (({"url": f"https://github.com/o/r/archive/{commit}.zip", "archive_info": {}}, " 21d82d7"),
                        ({"url": "https://x/r.git", "vcs_info": {"vcs": "git", "commit_id": commit}}, " 21d82d7"),
                        ({"url": "file:///somewhere/r", "dir_info": {}}, "")):
        monkeypatch.setattr(md, "distribution", lambda _n, f=found: Dist(f))
        assert blueprint.made_by() == f"refdissect {blueprint.__version__}{want}"


# ── The command ──────────────────────────────────────────────────────────────

def test_the_command_writes_beside_the_plan(tmp_path, capsys):
    plan = plan_for(tmp_path)
    work = tmp_path / "Song" / "work"
    work.mkdir()
    (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    assert main(["blueprint", str(work)]) == 0
    bp = json.loads((work / "blueprint.json").read_text(encoding="utf-8"))
    assert bp["name"] == "Song" and bp["made_by"].startswith("refdissect ")
    assert set(bp["meta"]) == {"plan", "plan_sha1"}
    assert_valid(bp)
    out = tmp_path / "elsewhere" / "b.json"
    assert write(work / "plan.json", out, log=lambda _: None) == out and out.is_file()


def test_the_command_refuses_what_live_build_refuses(tmp_path, capsys):
    """live-build's checks that need no Live: an 808 one-shot fitted again
    after the plan, and a pad file that is gone. The command exits 1."""
    plan = plan_for(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    _wav(Path(plan["tracks"]["808"]["sample_path"]), np.zeros(SR))
    with pytest.raises(SystemExit) as stop:
        main(["blueprint", str(work)])
    assert stop.value.code == 1 and "changed after the plan named it" in capsys.readouterr().err
    plan = plan_for(tmp_path / "again")
    Path(plan["tracks"]["drums"]["pads"]["36"]["path"]).unlink()
    (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["blueprint", str(work)])
    assert "Cannot read the files of pad 36" in capsys.readouterr().err
    assert not (work / "blueprint.json").exists()



def test_a_report_with_everything_wrong_in_direct_s_keys(tmp_path):
    """build_from_blueprint's report where Live took nothing as asked (its
    keys as ableton-live-mcp 2.4 writes them, CLAUDE.md's interface), mapped
    onto live-build --direct's keys, each difference logged."""
    plan = plan_for(tmp_path)
    plan["tracks"]["drums"]["pads"]["42"]["choke_group"] = 1
    bp = make_blueprint(plan, "Song", made="refdissect test")
    pad = bp["tracks"][1]["instrument"]["drum_rack"]
    then = {"value": -7.0, "display": "-7.0 dB"}
    found = {
        "tracks": [{"track": "808 (synth)", "devices": ["Simpler"], "notes": 2,
                    "problems": ["audio clip x.wav: Live left it warped"]},
                   {"track": "Drums (library)", "devices": ["Drum Rack"], "notes": 5}],
        "locators": 1,
        "locators_moved": [{"name": "B (3)", "asked": 8.0, "live": 12.0, "deleted": True},
                           {"name": "C (4)", "asked": 12.0, "live": None, "deleted": False}],
        "checks": {
            "time_signature": {"want": "4/4", "live": "3/4", "same": False},
            "tempo": {"want": 140.0, "live": 139.5, "same": False},
            "samples": {"checked": 4, "wrong": {
                "Drums (library) pad 38": {"want": "/lib/a.wav", "live": "/lib/b.wav", "same": False},
                "808 (synth)": {"want": "/lib/808.wav", "live": None, "same": False, "error": "no sample"}}},
            "volumes": {"checked": 3, "held": 1, "held_by_text": ["Drums (library) pad 36"], "not_held": {
                "Drums (library) pad 42": {"want": -7.2, "value": -7.0, "display": "-7.0 dB", "held": False,
                                           "then": then, "by_text": False}}},
            "choke_groups": {"checked": 1, "wrong": {
                "Drums (library) pad 42": {"want": 1, "live": 0, "same": False, "error": "refused"}}},
            "voices": {"808 (synth)": {"want": 1, "live": 6, "same": False}}},
        "files": {"folder": "/lib", "copied": 4}, "ok": False}
    logged = []
    report = as_live_build(found, bp, logged.append)
    assert report["problems"] == {"808 (synth)": ["audio clip x.wav: Live left it warped"]}
    assert report["locators"] == 1 and report["locators_moved"] == found["locators_moved"]
    assert report["choke_groups"] == {"42": {"plan": 1, "live": 0, "same": False, "error": "refused"}}
    assert report["volumes_held"] == 1
    assert report["volumes_not_held"] == {
        "36": {"plan": pad["36"]["settings"]["Volume"], "text": True, "held_by_text": True},
        "42": {"plan": -7.2, "value": -7.0, "display": "-7.0 dB", "then": then, "text": True}}
    assert report["samples_checked"] == 4
    assert report["wrong_samples"] == {
        "38": {"plan": "/lib/a.wav", "live": "/lib/b.wav", "same": False},
        "808": {"plan": "/lib/808.wav", "live": None, "same": False, "error": "no sample"}}
    assert report["808_voices"] == {"want": 1, "live": 6, "same": False}
    assert not report["tempo"]["same"] and not report["time_signature"]["same"]
    assert report["ok"] is False and report["files"] == found["files"] and report["build_from_blueprint"] is found
    text = "\n".join(logged)
    for line in ("Live put it at beat 12 for locator 'B (3)' at beat 8.0: deleted",
                 "Live made none for locator 'C (4)'",
                 "pad 42 is in choke group 0, not the plan's 1 (refused)",
                 "pad 36 did not hold Volume",
                 "pad 42 holds Volume -7.0 (-7.0 dB), not the plan's -7.2 dB: sent as text",
                 "pad 38 holds /lib/b.wav, not the plan's /lib/a.wav",
                 "the 808 holds None, not the plan's /lib/808.wav",
                 "the 808 has 6 voices, not the plan's 1",
                 "Live's tempo is 139.5 BPM, not the plan's 140.0",
                 "Live's time signature is 3/4, not the plan's 4/4"):
        assert line in text, line


def test_a_clean_report_logs_nothing(tmp_path):
    bp = make_blueprint(plan_for(tmp_path), "Song", made="refdissect test")
    found = {"tracks": [{"track": "808 (synth)", "notes": 2}], "locators": 2, "ok": True,
             "checks": {"samples": {"checked": 4, "wrong": {}}, "volumes": {"checked": 3, "held": 3},
                        "tempo": {"want": 140.0, "live": 140.0, "same": True},
                        "time_signature": {"want": "4/4", "live": "4/4", "same": True}}}
    logged = []
    report = as_live_build(found, bp, logged.append)
    assert logged == [] and report["ok"] is True
    for key in ("problems", "locators_moved", "volumes_not_held", "wrong_samples", "choke_groups", "808_voices"):
        assert key not in report, key

# ── The same blueprints through the real validators, where they are installed ──

def _cases(tmp_path):
    plain = plan_for(tmp_path / "plain")
    glide = plan_for(tmp_path / "glide")
    glide["tracks"]["808"].update(voices=1, glide_mode="Glide", glide_time_ms=45.0)
    choke = plan_for(tmp_path / "choke", meter=(3, 4))
    choke["tracks"]["drums"]["pads"]["42"]["choke_group"] = 1
    return [make_blueprint(p, "Song", made="refdissect test")
            for p in (plain, glide, choke, plan_for(tmp_path / "none", one_shot=False))]


def test_the_schema_with_jsonschema(tmp_path):
    if os.environ.get("REFDISSECT_REQUIRE_LIVE"):  # CI: fail, not skip (the [live] extra brings jsonschema)
        import jsonschema  # noqa: F401
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(SCHEMA)
    for bp in _cases(tmp_path):
        assert [e.message for e in validator.iter_errors(bp)] == []
    assert list(validator.iter_errors({**bp, "checks": {"samples": True}, "extra": 1}))  # unknown fields refused


def test_ableton_live_mcps_validator(tmp_path):
    if os.environ.get("REFDISSECT_REQUIRE_LIVE"):  # CI: fail, not skip
        import MCP_Server.blueprint  # noqa: F401
    blueprint = pytest.importorskip("MCP_Server.blueprint",
                                    reason="ableton-live-mcp 2.4.0 or later (the [live] extra) not installed")
    for bp in _cases(tmp_path):
        assert blueprint.validate(bp) == []
        assert blueprint.check_sources(bp, None) == []
    # the schema copy here is the installed one's, but for its $comment
    assert {k: v for k, v in SCHEMA.items() if k != "$comment"} == blueprint.schema()


def test_the_minimal_check_catches_what_the_schema_refuses(tmp_path):
    bp = make_blueprint(plan_for(tmp_path), "Song", made="refdissect test")
    assert schema_errors({**bp, "extra": 1}) == ["blueprint: 'extra' is not allowed"]
    bad = json.loads(json.dumps(bp))
    bad["tracks"][1]["instrument"]["drum_rack"]["42"]["choke_group"] = 17
    bad["files"]["pad-36"]["name"] = "kick"
    bad["time_signature"] = "4/3"
    assert len(schema_errors(bad)) == 3
