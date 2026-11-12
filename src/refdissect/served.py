"""A worker kept running between calls: beat_this's and Basic Pitch's (beats.py,
basicpitch.py), each in its own uv environment.

A call to a worker's command starts uv, Python, torch or Basic Pitch and its
model again: 1.2 of beat_this's 1.7 s a suite beat and 1.6 of Basic Pitch's
2.3 s (v1-1, warm). Kept (keep(True), as a suite run's jobs do), a worker
starts once, with --serve, and then answers one request per line on its
standard input with one line on its standard output, starting with MARK
(whatever else a library prints to it is skipped). Its standard error goes
to a file, read back when it fails. A request runs the code a call runs on
the same arguments, in the same order, so the answers are the same.

Off (the default, and for the commands), every call starts the worker
anew. A worker that cannot start, or that fails or dies on a request, is
let go: the call raises as a call would, with what the worker said, and
the next one starts it again.
"""

from __future__ import annotations

import atexit
import json
import subprocess
import tempfile
import threading

MARK = "#refdissect-served# "

_keep = False
_running: dict[tuple[str, ...], Worker] = {}
_lock = threading.Lock()


def keep(on: bool = True) -> None:
    """Keep the workers running between calls (on), or stop them (off)."""
    global _keep
    _keep = on
    if not on:
        close()


def kept() -> bool:
    return _keep


class Worker:
    """One worker started with `command + ["--serve"]`."""

    def __init__(self, command: list[str]):
        self.err = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen([*command, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, text=True, bufsize=1)

    def ask(self, request: dict) -> dict:
        """The worker's answer to request; RuntimeError with the end of its
        standard error if it fails or has died."""
        try:
            self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            for line in self.proc.stdout:
                if line.startswith(MARK):
                    answer = json.loads(line[len(MARK):])
                    if "error" in answer:
                        raise RuntimeError(answer["error"])
                    return answer["result"]
        except (OSError, ValueError) as e:
            raise RuntimeError(f"{e}\n{self.said()}") from e
        raise RuntimeError(f"the worker stopped ({self.proc.wait()})\n{self.said()}")

    def said(self) -> str:
        self.err.seek(0)
        return self.err.read()[-2000:]

    def close(self) -> None:
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.err.close()


def ask(command: list[str], request: dict) -> dict:
    """request answered by the worker kept for command, started if needed;
    one that fails is let go."""
    key = tuple(command)
    with _lock:
        worker = _running.pop(key, None) or Worker(command)
        try:
            out = worker.ask(request)
        except Exception:
            worker.close()
            raise
        _running[key] = worker
        return out


def close() -> None:
    """Every kept worker stopped."""
    while _running:
        _running.popitem()[1].close()


atexit.register(close)
