"""cache: results kept under the code of their step (the module that makes
them and every module it reaches) as the process loaded it. A file of the
step edited after the import (an edit, a commit or a checkout while a run
goes on) keeps the process from reading a result back or keeping one; a
file the step does not reach does not."""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

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


@pytest.mark.slow
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


def test_a_key_covers_every_file_its_step_reaches(monkeypatch):
    """Two lists of files kept by hand (synth808.CODE, pipeline.CODE_808)
    would miss a module the code comes to import, as the 808 reader reached
    scenarios.py from inside a function: the key reads, from the code
    itself, every module of the package the step's module imports at any
    depth (cache.reach), inside functions too, with cache.py and the
    package's __init__, and the samples and settings. A file the step does
    not reach leaves its key as it is."""
    names = set(cache.LOADED)
    assert names == {p.relative_to(cache.PACKAGE).as_posix() for p in cache.PACKAGE.rglob("*.py")}
    assert {"bass808.py", "scenarios.py", "synth808.py", "cache.py", "workers/beats_worker.py"} <= names
    x = np.linspace(-1, 1, 64, dtype=np.float32)
    loaded = dict(cache.LOADED)
    for step, outside in [("bass808", {"synth808.py", "pipeline.py", "live.py", "render.py", "suite.py", "cli.py"}),
                          ("synth808", {"bass808.py", "pipeline.py", "live.py", "render.py", "suite.py", "cli.py"})]:
        reached = set(cache.reach(step))
        assert {f"{step}.py", "scenarios.py", "library.py", "audio.py", "cache.py", "__init__.py"} <= reached
        assert not reached & outside and reached < names
        first = cache.key((x,), {"f1": 41.2, "seed": 1}, step)
        for name in sorted(names):
            monkeypatch.setattr(cache, "LOADED", {**loaded, name: "0" * 64})
            assert (cache.key((x,), {"f1": 41.2, "seed": 1}, step) != first) == (name in reached), (step, name)
        monkeypatch.setattr(cache, "LOADED", loaded)
        assert cache.key((x,), {"seed": 1, "f1": 41.2}, step) == first
        same = {"f1": 41.2, "seed": 1}
        for arrays, settings in [((x * 2,), same), ((x.astype(np.float64),), same), ((x, x), same),
                                 ((x,), {"f1": 41.2 + 1e-12, "seed": 1})]:
            assert cache.key(arrays, settings, step) != first
    assert cache.key((x,), {}, "bass808") != cache.key((x,), {}, "synth808")
    with pytest.raises(ValueError):
        cache.reach("no_such_module")


def test_imports_are_read_at_any_depth_and_in_every_form():
    """cache.imports: relative and absolute imports, a module named in
    `from . import`, an import inside a function, a package's __init__ on
    the way, a worker script named by its file name; not a name that is no
    module, nor the file itself."""
    files = {"__init__.py", "a.py", "b.py", "c.py", "d.py", "e.py", "sub/__init__.py", "sub/f.py",
             "workers/x_worker.py", "workers/y_worker.py"}
    source = b"""
import numpy
from . import a
from .b import thing
from refdissect.sub import f
import refdissect.e
def later():
    from .c import other
    from . import nothing_here
    return "x_worker.py"
"""
    assert cache.imports("d.py", source, files) == {"a.py", "b.py", "c.py", "e.py", "sub/__init__.py", "sub/f.py",
                                                     "__init__.py", "workers/x_worker.py"}
    assert cache.imports("sub/f.py", b"from .. import a\nfrom . import f\n", files) == {"a.py", "sub/__init__.py",
                                                                                     "__init__.py"}


# one fresh process: a step run, then the package's modules it loaded, as files
STEP_LOADS = """
import json, sys
import numpy as np
step = sys.argv[1]
sr = 44100
t = np.arange(4 * sr) / sr
drums = np.zeros(len(t), np.float32)
bass = np.zeros(len(t), np.float32)
for k, hz in enumerate([41.2, 41.2, 55.0, 49.0, 41.2, 61.7, 41.2, 55.0]):
    a = int(k * 0.5 * sr)
    drums[a:a + 400] += np.hanning(800)[400:] * np.sin(2 * np.pi * 60 * t[:400])
    n = len(t) - a
    bass[a:] += (0.8 * np.exp(-t[:n] / 0.35) * np.sin(2 * np.pi * hz * t[:n])).astype(np.float32)
import refdissect, importlib
module = importlib.import_module(f"refdissect.{step}")
if step == "bass808":
    module.transcribe(bass, drums, log=lambda m: None, slides=True)
else:
    module.fit(bass[: int(0.6 * sr)].copy(), 41.2, maxiter=2, popsize=4, log=lambda m: None)
from pathlib import Path
root = Path(refdissect.__file__).parent
print(json.dumps(sorted(Path(m.__file__).relative_to(root).as_posix() for name, m in sys.modules.items()
                        if (name == "refdissect" or name.startswith("refdissect.")) and getattr(m, "__file__", None))))
"""


@pytest.mark.slow
@pytest.mark.parametrize("step", ["bass808", "synth808"])
def test_every_module_a_step_loads_is_reached(step):
    """A step run in a fresh process (the 808 reader with its slides, a fit)
    loads no module of the package that cache.reach does not find: an
    import the source does not show would leave its file out of the key."""
    out = subprocess.run([sys.executable, "-c", STEP_LOADS, step], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    loaded = set(json.loads(out.stdout.splitlines()[-1]))
    assert f"{step}.py" in loaded and loaded <= set(cache.reach(step)), loaded - set(cache.reach(step))


def test_the_808_reader_is_neither_read_back_nor_kept_while_the_code_on_disk_differs(tmp_path, monkeypatch):
    """The disk is read again before a result is read back and before one is
    kept: while a file differs from what the process loaded, the reader
    runs and keeps nothing, and the log names the file; once the file is as
    it was, the kept result is read back. A file the reader reaches only
    from inside a function (scenarios.py) counts; one it does not reach
    (live.py) does not."""
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
    (copy / "live.py").write_text((copy / "live.py").read_text() + "\n# edited\n")
    said = []
    assert pipeline._transcribe(bass, drums, kept, said.append) == ({"notes": []}, str(kept / name))
    assert len(calls) == 2 and said == [f"808 reader: read back from {kept / name}; its log when it ran:"]
    (copy / "scenarios.py").write_text((copy / "scenarios.py").read_text() + "\n# edited\n")
    said = []
    assert pipeline._transcribe(bass, drums, kept, said.append) == ({"notes": []}, None) and len(calls) == 3
    assert said[0] == f"  cache: scenarios.py changed on disk since this process started: {name} not read back"
