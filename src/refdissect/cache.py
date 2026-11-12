"""Results kept on disk and read back instead of computed again: the 808
reader's (pipeline._transcribe, 20-40 s of a suite beat's analyze) and the
808 fits (synth808.fit, 50-140 s).

A result is kept under a key of everything that decides it: its inputs'
samples, its settings, numpy's, scipy's and soxr's versions, and the code
of its step: the module whose function makes the result and every module
of the package that module reaches by import, at any depth and from inside
a function too, as this process loaded them (reach(): read from the source
of every .py file of the package, kept when the package is first imported,
before any other of its modules, as __init__ imports this one first). A
worker script a reached module names ("..._worker.py") is reached too, and
so are this module and the package's __init__. An edit to a module the step
does not reach (live.py, render.py, the plan, the suite) leaves its kept
results as they are; with the whole package hashed, any edit made every
kept result stale.

Two lists of files kept by hand would miss a module the code comes to
import (the 808 reader reached scenarios.py from inside a function): the
graph is read from the code itself. A module imported in a way the source
does not show (importlib, __import__) would be missed: the package has
none, and a test runs each step in a fresh process and checks that every
module of the package it loaded is reached.

A key read from disk when a result is kept filed a run's results under code
edited after the run had started: a fit read back that way had f_start 93.3
Hz where the edited code finds 53.2 Hz. So the step's files are read again
before a result is read back and before one is kept: while any differs from
LOADED (an edit, a commit or a checkout during a run), what runs may not be
what is on disk, and nothing is read back or kept; the log names the files.
An edit undone before the next check is not seen.

Code changed in memory is not seen either: a script that patches a module's
function or constant, then reads a result back, gets the unpatched code's.
The pipeline's functions keep nothing unless given a folder, and the
commands that keep by default take --no-cache: such a script uses neither.
The caller of a step passes all it decides through the settings
(pipeline._transcribe: slides); its own module is not in the key.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).parent
NAME = __name__.rpartition(".")[0] or "refdissect"


def _read(root: Path) -> dict[str, bytes]:
    """Every .py file under root (its workers/ too), by path."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*.py"))}


def digest(root: Path) -> dict[str, str]:
    """sha256 of every .py file under root (its workers/ too), by path."""
    return {name: hashlib.sha256(data).hexdigest() for name, data in _read(root).items()}


SOURCE = _read(PACKAGE)
LOADED = {name: hashlib.sha256(data).hexdigest() for name, data in SOURCE.items()}


def _module_file(module: str, files) -> str | None:
    """The file of the package's module (dotted, under NAME), if it has one."""
    if module != NAME and not module.startswith(NAME + "."):
        return None
    path = module[len(NAME) + 1:].replace(".", "/")
    for name in ((f"{path}.py", f"{path}/__init__.py") if path else ("__init__.py",)):
        if name in files:
            return name
    return None


def imports(name: str, source: bytes, files) -> set[str]:
    """The package's files that file `name` (its source) imports anywhere in
    it, inside functions too, with the __init__ of every package on the way,
    and the worker scripts it names."""
    parts = name.split("/")[:-1]  # the file's package below NAME
    here = ".".join([NAME, *parts])
    out: set[str] = set()

    def add(module: str) -> None:
        f = _module_file(module, files)
        if f:
            out.add(f)
        while "." in module:  # importing a.b.c runs a/__init__ and a/b/__init__
            module = module.rpartition(".")[0]
            f = _module_file(module, files)
            if f:
                out.add(f)
    for node in ast.walk(ast.parse(source, filename=name)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = here.split(".")
                base = ".".join(base[:len(base) - node.level + 1])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            add(module)
            for alias in node.names:  # from . import cache: a module, not a name in one
                add(f"{module}.{alias.name}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith("_worker.py"):
            out |= {f for f in files if f == f"workers/{node.value}"}
    out.discard(name)
    return out


@functools.cache
def reach(module: str) -> tuple[str, ...]:
    """The files of the step whose function is in `module` (its name in the
    package: "bass808"): the module, every one it reaches by import, this
    module and the package's __init__, as this process loaded them."""
    start = _module_file(f"{NAME}.{module}", SOURCE)
    if start is None:
        raise ValueError(f"{module} is not a module of the package")
    seen, todo = {start, "__init__.py", "cache.py"}, [start]
    while todo:
        name = todo.pop()
        for f in imports(name, SOURCE[name], SOURCE) - seen:
            seen.add(f)
            todo.append(f)
    return tuple(sorted(seen))


def changed(module: str | None = None) -> list[str]:
    """The files that differ on disk from LOADED (added and removed ones
    too): of the step in module (reach()), or of the whole package."""
    now = digest(PACKAGE)
    names = set(reach(module)) if module else LOADED.keys() | now.keys()
    return sorted(k for k in names if LOADED.get(k) != now.get(k))


def key(arrays, settings: dict, module: str) -> str:
    """What decides a result of the step in module: the arrays (type, shape
    and samples), the settings (by repr, which is exact for a float), the
    versions and the step's code as loaded (reach())."""
    import numpy
    import scipy
    import soxr
    head = {"code": {f: LOADED[f] for f in reach(module)}, "step": module,
            "versions": [numpy.__version__, scipy.__version__, soxr.__version__],
            "settings": repr(sorted(settings.items()))}
    h = hashlib.sha256(json.dumps(head, sort_keys=True).encode())
    for a in arrays:
        h.update(f"{a.dtype.str} {a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def kept(path: Path | None, log, module: str) -> dict | None:
    """What is kept at path, or None: nothing is, it cannot be read, or the
    step's code (module's, reach()) changed on disk since this process
    loaded it (logged)."""
    if path is None or not path.is_file() or _moved(path, "not read back", log, module):
        return None
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def keep(path: Path | None, obj: dict, log, module: str) -> None:
    """obj kept at path, written whole and then moved in (a run reading it at
    the same time never sees half a file), unless the step's code (module's,
    reach()) changed on disk since this process loaded it (logged)."""
    if path is None or _moved(path, "not kept", log, module):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(part, path)


def _moved(path: Path, what: str, log, module: str) -> bool:
    moved = changed(module)
    if moved:
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        log(f"  cache: {shown} changed on disk since this process started: {path.name} {what}")
    return bool(moved)
