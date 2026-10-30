"""Results kept on disk and read back instead of computed again: the 808
reader's (pipeline._transcribe, 20-40 s of a suite beat's analyze) and the
808 fits (synth808.fit, 50-140 s).

A result is kept under a key of everything that decides it: its inputs'
samples, its settings, numpy's, scipy's and soxr's versions, and the code,
every .py file of the package as this process loaded it (LOADED: read when
the package is first imported, before any other of its modules, as
__init__ imports this one first). Two lists of files kept by hand would
miss a module the code comes to import, and a key read from disk when a
result is kept filed a run's results under code edited after the run had
started: a fit read back that way had f_start 93.3 Hz where the edited code
finds 53.2 Hz. So the files are read again before a result is read back and
before one is kept: while any differs from LOADED (an edit, a commit or a
checkout during a run), what runs may not be what is on disk, and nothing is
read back or kept; the log names the files. An edit undone before the next
check is not seen.

Code changed in memory is not seen either: a script that patches a module's
function or constant, then reads a result back, gets the unpatched code's.
The pipeline's functions keep nothing unless given a folder, and the
commands that keep by default take --no-cache: such a script uses neither.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).parent


def digest(root: Path) -> dict[str, str]:
    """sha256 of every .py file under root (its workers/ too), by path."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


LOADED = digest(PACKAGE)


def changed() -> list[str]:
    """The package's .py files that differ on disk from LOADED (added and
    removed ones too)."""
    now = digest(PACKAGE)
    return sorted(k for k in LOADED.keys() | now.keys() if LOADED.get(k) != now.get(k))


def key(arrays, settings: dict) -> str:
    """What decides a result: the arrays (type, shape and samples), the
    settings (by repr, which is exact for a float), the versions and the
    code as loaded."""
    import numpy
    import scipy
    import soxr
    head = {"code": LOADED, "versions": [numpy.__version__, scipy.__version__, soxr.__version__],
            "settings": repr(sorted(settings.items()))}
    h = hashlib.sha256(json.dumps(head, sort_keys=True).encode())
    for a in arrays:
        h.update(f"{a.dtype.str} {a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:32]


def kept(path: Path | None, log) -> dict | None:
    """What is kept at path, or None: nothing is, it cannot be read, or the
    code changed on disk since this process loaded it (logged)."""
    if path is None or not path.is_file() or _moved(path, "not read back", log):
        return None
    try:
        out = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return out if isinstance(out, dict) else None


def keep(path: Path | None, obj: dict, log) -> None:
    """obj kept at path, written whole and then moved in (a run reading it at
    the same time never sees half a file), unless the code changed on disk
    since this process loaded it (logged)."""
    if path is None or _moved(path, "not kept", log):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f"{path.name}.{os.getpid()}.part")
    part.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(part, path)


def _moved(path: Path, what: str, log) -> bool:
    moved = changed()
    if moved:
        shown = ", ".join(moved[:3]) + (f" and {len(moved) - 3} more" if len(moved) > 3 else "")
        log(f"  cache: {shown} changed on disk since this process started: {path.name} {what}")
    return bool(moved)
