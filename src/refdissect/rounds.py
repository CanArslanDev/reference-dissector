"""A listening round: candidates the listener picks from, before anything is built.

The sketch step of a style beat (docs/11-style-beat-lessons.md):
each candidate (an 8-bar render of one section, or a sound) becomes one
numbered A/B file - the reference's same bars, a beep, the candidate, at equal
loudness - and a manifest lists what each number is and what to answer. Sent
one question at a time, a listener could answer ("the chords: 1"); ten
multi-part files at once could not be judged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .listening import ab_file


def _read(path: str | Path, span: tuple[float, float] | None) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True)
    if span:
        x = x[int(span[0] * sr):int(span[1] * sr)]
    return x, sr


def listening_round(reference: str | Path, candidates: list[str | Path], out_dir: str | Path,
                    question: str, reference_span: tuple[float, float] | None = None,
                    candidate_span: tuple[float, float] | None = None, target_lufs: float = -14.0) -> dict:
    """One A/B file per candidate in out_dir (1.wav, 2.wav, ...) and manifest.json /
    manifest.md naming them. Spans are seconds; the candidate's rate must match
    the reference's."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref, sr = _read(reference, reference_span)
    rows = []
    for k, c in enumerate(candidates, 1):
        x, csr = _read(c, candidate_span)
        if csr != sr:
            import soxr
            x = soxr.resample(x, csr, sr)
        info = ab_file(ref, x, sr, out / f"{k}.wav", target_lufs=target_lufs)
        rows.append({"number": k, "candidate": str(c), **info})
    manifest = {"question": question, "reference": str(reference), "reference_span_s": reference_span,
                "candidate_span_s": candidate_span, "files": rows,
                "answer": "the numbers that work, and for the rest one word on why"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = [f"# {question}", "", "Each file: the reference, a beep, then one candidate (same loudness).", ""]
    for r in rows:
        scaled = f" (scaled {r['scaled_db']} dB)" if r["scaled_db"] else ""
        lines.append(f"{r['number']}. `{Path(r['candidate']).name}`{scaled}")
    lines += ["", "Answer: " + manifest["answer"] + "."]
    (out / "manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
