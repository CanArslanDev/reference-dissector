import json

import numpy as np
import soundfile as sf

from refdissect.audio import SR, transpose
from refdissect.library import Library, inventory, role_of, summarize, verdict


def test_roles_from_names():
    assert role_of("Kits/808 Snare Tight.wav") == "snare"
    assert role_of("hats/Open Hat 2.wav") == "open_hat"
    assert role_of("808s/dirty 808 C.wav") == "808"
    assert role_of("Loops/Keys 90bpm.wav") == "melodic/loop"
    assert role_of("weird/thing.wav") == "other"


def _library(tmp_path, kit):
    folder = tmp_path / "kit"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for i in range(30):  # distractors: other snares and hats
        x = kit["snare"] if i % 2 else kit["hat"]
        x = np.convolve(x, rng.standard_normal(32) * 0.2, mode="same")
        sf.write(folder / f"snare {i}.wav", x / np.max(np.abs(x)), SR)
    sf.write(folder / "the snare.wav", kit["snare"], SR)
    sf.write(folder / "the snare copy.wav", kit["snare"], SR)
    sf.write(folder / "big kick.wav", kit["kick"], SR)
    rows = inventory([folder])
    return rows, Library.build(rows, seconds=0.4)


def test_inventory_and_summary(tmp_path, kit):
    rows, lib = _library(tmp_path, kit)
    assert len(rows) == 33 and len(lib) == 33
    counts = summarize(rows)["kit"]
    assert counts["snare"][0] == 32 and counts["kick"][0] == 1
    json.dumps(rows)


def test_finds_a_transposed_sample(tmp_path, kit):
    """A snare pitched down a semitone in the mix: the shortlist proposes -1
    and the waveform match confirms it."""
    _, lib = _library(tmp_path, kit)
    heard = transpose(kit["snare"], -1)[: int(0.2 * SR)]
    heard = np.r_[np.zeros(int(0.005 * SR), np.float32), heard]
    ranked = lib.distinct(lib.match(heard, "snare", semitones=range(-3, 4), top=10), len(heard))
    assert ranked[0]["path"].endswith("the snare.wav") or ranked[0]["path"].endswith("the snare copy.wav")
    assert ranked[0]["semitones"] == -1.0 and ranked[0]["ncc"] > 0.95
    # its copy under another name is not listed again
    assert sum(r["path"].endswith(("the snare.wav", "the snare copy.wav")) for r in ranked) == 1


def test_save_and_load(tmp_path, kit):
    _, lib = _library(tmp_path, kit)
    lib.save(tmp_path / "lib.npz")
    back = Library.load(tmp_path / "lib.npz")
    assert back.paths == lib.paths and back.spectra.shape == lib.spectra.shape


def test_verdicts():
    assert verdict("snare", 0.98, 0.3) == "same sample"
    assert verdict("hat", 0.9, 0.01).startswith("same family")
    assert verdict("kick", 0.95, 0.02) == "same family"  # tonal sounds all correlate
    assert verdict("808", 0.98, 0.08).startswith("same sample")
    assert verdict("snare", 0.5, 0.2) == "no close match"
