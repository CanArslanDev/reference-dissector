"""cache: results kept under the package's code as the process loaded it. A
file edited after the import (an edit, a commit or a checkout while a run
goes on) keeps the process from reading a result back or keeping one."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np

from refdissect import cache, pipeline

# one process: imports synth808, then (edit) synth808.py changes on disk, then a fit with a cache folder
FIT_AFTER_AN_EDIT = """
import json, sys
from pathlib import Path
import refdissect
from refdissect import synth808
src = Path(refdissect.__file__).parent
assert src == Path(sys.argv[1]), src
if sys.argv[3] == "edit":
    (src / "synth808.py").write_text((src / "synth808.py").read_text() + "\\n# edited\\n")
ref = synth808.laid_out(synth808.Params(41.2, 80.0, 40.0, 0.8, 2.0, 2.5, 0.1, 0.5), int(0.6 * synth808.SR))
p, info = synth808.fit(ref, 41.2, maxiter=2, popsize=4, cache=sys.argv[2])
print(json.dumps(info))
"""


def _copy(tmp_path):
    copy = tmp_path / "src" / "refdissect"
    shutil.copytree(cache.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    return copy


def test_a_process_whose_code_changed_on_disk_keeps_nothing(tmp_path):
    """The review's case: a process imports synth808, synth808.py is edited
    on disk, and the process fits with a cache. Keyed on the files as they
    were when the fit was kept, its fit (of the old code) was filed under
    the edited code, and a process of the edited code read it back: f_start
    93.3 Hz, where its own search finds 53.2 Hz. Now nothing is kept, the
    log says why, and the next process fits and keeps its own."""
    copy, kept = _copy(tmp_path), tmp_path / "kept"

    def run(mode: str) -> tuple[dict, str]:
        env = {**os.environ, "PYTHONPATH": str(copy.parent), "PYTHONDONTWRITEBYTECODE": "1"}
        out = subprocess.run([sys.executable, "-c", FIT_AFTER_AN_EDIT, str(copy), str(kept), mode],
                             capture_output=True, text=True, env=env, timeout=300)
        assert out.returncode == 0, out.stderr[-2000:]
        *said, last = out.stdout.splitlines()
        return json.loads(last), "\n".join(said)

    info, said = run("edit")
    assert not list(kept.glob("fit-808-*")) and "cached" not in info
    assert "cache: synth808.py changed on disk since this process started" in said and "not kept" in said
    info, said = run("same")
    assert "cached" not in info and not said and len(list(kept.glob("fit-808-*.json"))) == 1
    assert run("same")[0]["cached"] == next(kept.glob("fit-808-*.json")).name


def test_a_key_covers_every_file_of_the_package(monkeypatch):
    """Two lists of files kept by hand (synth808.CODE, pipeline.CODE_808)
    would miss a module the code comes to import, as the 808 reader reached
    scenarios.py from inside a function: the key reads every .py file of
    the package, its workers too, as well as the samples and settings."""
    names = set(cache.LOADED)
    assert names == {p.relative_to(cache.PACKAGE).as_posix() for p in cache.PACKAGE.rglob("*.py")}
    assert {"bass808.py", "scenarios.py", "synth808.py", "cache.py", "workers/beats_worker.py"} <= names
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    first = cache.key((x,), {"f1": 41.2, "seed": 1})
    loaded = dict(cache.LOADED)
    for name in sorted(names):
        monkeypatch.setattr(cache, "LOADED", {**loaded, name: "0" * 64})
        assert cache.key((x,), {"f1": 41.2, "seed": 1}) != first, name
    monkeypatch.setattr(cache, "LOADED", loaded)
    assert cache.key((x,), {"seed": 1, "f1": 41.2}) == first
    for arrays, settings in [((x * 2,), {"f1": 41.2, "seed": 1}), ((x.astype(np.float64),), {"f1": 41.2, "seed": 1}),
                             ((x, x), {"f1": 41.2, "seed": 1}), ((x,), {"f1": 41.2 + 1e-12, "seed": 1})]:
        assert cache.key(arrays, settings) != first


def test_the_808_reader_is_neither_read_back_nor_kept_while_the_code_on_disk_differs(tmp_path, monkeypatch):
    """The disk is read again before a result is read back and before one is
    kept: while a file differs from what the process loaded, the reader
    runs and keeps nothing, and the log names the file; once the file is as
    it was, the kept result is read back."""
    copy = _copy(tmp_path)
    monkeypatch.setattr(cache, "PACKAGE", copy)
    monkeypatch.setattr(cache, "LOADED", cache.digest(copy))
    calls = []
    monkeypatch.setattr(pipeline.bass808, "transcribe", lambda bass, drums, log: calls.append(1) or {"notes": []})
    bass, drums, kept = np.ones(100, np.float32), np.zeros(100, np.float32), tmp_path / "kept"
    assert pipeline._transcribe(bass, drums, kept, print) == ({"notes": []}, None)
    name = next(kept.glob("808-*.json")).name
    source = (copy / "bass808.py").read_text()
    (copy / "bass808.py").write_text(source + "\n# edited\n")
    said = []
    assert pipeline._transcribe(bass, drums, kept, said.append) == ({"notes": []}, None) and len(calls) == 2
    assert said == [f"  cache: bass808.py changed on disk since this process started: {name} not read back",
                    f"  cache: bass808.py changed on disk since this process started: {name} not kept"]
    (copy / "bass808.py").write_text(source)
    said = []
    assert pipeline._transcribe(bass, drums, kept, said.append) == ({"notes": []}, str(kept / name))
    assert len(calls) == 2 and said == [f"808 reader: read back from {kept / name}; its log when it ran:"]
