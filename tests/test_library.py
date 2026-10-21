import json

import numpy as np
import pytest
import soundfile as sf
from conftest import placed
from scipy.optimize import nnls

from refdissect.audio import SR, transpose
from refdissect.library import Library, inventory, pair_fit, role_of, speckle_path, summarize, verdict


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


# ── The speckle index: noisy sounds at any pitch ──

@pytest.mark.parametrize("d", [0.0, 0.3, -0.47, 2.4, -7.15])
def test_a_detuned_snare_and_its_cents(noisy, d):
    """A snare played d semitones off (-7.15: over half an octave): the
    speckle search names the file and the transposition within a cent."""
    lib, sounds = noisy
    t = placed(sounds[30], d, 150)
    top = lib.distinct(lib.match_fine(t, "snare", top=20), len(t))[0]
    assert top["i"] == 30 and abs(100 * (top["semitones"] - d)) <= 1.0 and top["ncc"] > 0.95


@pytest.mark.parametrize("d", [0.25, -0.33])
def test_a_detuned_hat(noisy, d):
    lib, sounds = noisy
    t = placed(sounds[14], d, 80)
    old = lib.distinct(lib.match(t, "hat", semitones=range(-3, 4), top=20), len(t))[0]
    new = lib.distinct(lib.match_fine(t, "hat", top=20), len(t))[0]
    assert new["i"] == 14 and abs(100 * (new["semitones"] - d)) <= 1.0
    assert new["ncc"] > old["ncc"] + 0.2  # off by a quarter of a semitone, no whole semitone fits


def test_the_semitone_search_joins_the_speckle_one(noisy):
    """union: match()'s best candidates are tried at their speckle
    transposition as well; nothing match() found is lost."""
    lib, sounds = noisy
    t = placed(sounds[30], -1.0, 150)
    both = lib.match_fine(t, "snare", keep=5, semitones=range(-3, 4), union=10, top=0)
    alone = lib.match(t, "snare", semitones=range(-3, 4), top=10)
    found = {r["i"]: r["ncc"] for r in both}
    assert all(found.get(r["i"], 0) >= r["ncc"] for r in alone)
    assert both[0]["i"] == 30 and both[0]["semitones"] == -1.0


def test_the_template_length_is_the_index_one(noisy):
    lib, sounds = noisy
    with pytest.raises(ValueError, match="150 ms"):
        lib.match_fine(placed(sounds[30], 0.0, 100), "snare")


def test_speckle_index_saved_next_to_the_library(noisy, tmp_path):
    lib, _ = noisy
    lib.save(tmp_path / "lib.npz")
    lib.save_speckle(speckle_path(tmp_path / "lib.npz"))
    back = Library.load(tmp_path / "lib.npz")
    assert back.speckle_info["version"] == 1 and back.speckle_info["entries"] == 120  # kicks not indexed
    assert np.array_equal(back.speckles["hat"][1], lib.speckles["hat"][1])
    other = Library(lib.paths[:-1], lib.roles[:-1], lib.audio[:-1], lib.seconds)
    assert other.load_speckle(speckle_path(tmp_path / "lib.npz")) == \
        {"ignored": "lib_speckle.npz: built for another library's entries"}
    assert other.speckles == {}


def test_pair_fit_is_nonnegative_least_squares():
    rng = np.random.default_rng(1)
    atoms = rng.standard_normal((6, 400))
    target = 1.0 * atoms[0] + 0.5 * atoms[3] + 0.05 * rng.standard_normal(400)
    out = pair_fit(target, atoms)
    k = int(np.argmin(out["residual_db"]))
    assert k == 3 and np.isinf(out["residual_db"][0])
    g, res = nnls(atoms[[0, 3]].T, target)
    assert np.allclose([out["gains"][0][3], out["gains"][1][3]], g)
    assert out["residual_db"][3] == pytest.approx(10 * np.log10(res ** 2 / np.sum(target ** 2)))
    assert out["single_db"] > out["residual_db"][3] + 3
