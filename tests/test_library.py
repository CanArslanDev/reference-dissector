import json

import numpy as np
import pytest
import soundfile as sf
from conftest import burst, placed
from scipy.optimize import nnls

from refdissect.audio import SR, fit, samples, transpose, trim_onset
from refdissect.library import (
    SILENT_DB,
    Library,
    copies,
    inventory,
    pair_fit,
    role_of,
    speckle_path,
    summarize,
    verdict,
)


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


def test_a_silent_file_is_never_listed(tmp_path, kit, monkeypatch):
    """A float WAV of a snare 420 dB down (the suite library's fx file at
    -432 dBFS matched v1orig-3's hat at NCC 0.961, and its pad played
    silence): build() leaves it out, and from an index built before that
    floor, which holds it at full scale, no search lists it."""
    from refdissect import library

    folder = tmp_path / "kit"
    folder.mkdir()
    sounds = {"quiet snare.wav": kit["snare"] * 1e-21, "hat.wav": kit["hat"], "kick.wav": kit["kick"]}
    for name, x in sounds.items():
        sf.write(folder / name, x, SR, subtype="FLOAT")
    lib = Library.build(inventory([folder]), seconds=0.4)
    assert sorted(p.rsplit("/", 1)[1] for p in lib.paths) == ["hat.wav", "kick.wav"] and lib.floor_db == SILENT_DB
    old = Library([str(folder / name) for name in sounds], ["snare", "hat", "kick"],
                  np.stack([fit(trim_onset(x), samples(400)) / np.max(np.abs(x)) for x in sounds.values()])
                  .astype(np.float16), 0.4)
    t = placed(kit["snare"], 0.0, 150)
    assert old.floor_db is None and not old.audible(0) and old.audible(1)
    assert sorted(r["path"].rsplit("/", 1)[1] for r in old.match(t, "snare", top=10)) == ["hat.wav", "kick.wav"]
    old.save(tmp_path / "old.npz")
    lib.save(tmp_path / "lib.npz")
    assert Library.load(tmp_path / "old.npz").floor_db is None
    back = Library.load(tmp_path / "lib.npz")
    monkeypatch.setattr(library, "read", lambda *a, **k: 1 / 0)  # built with the floor: no file is read again
    assert back.floor_db == SILENT_DB and back.audible(0) and back.audible(1)


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


def test_a_copy_at_another_pitch_is_the_same_sound(noisy):
    """A kit's copy of a snare, resampled 0.22 semitones down: found by the
    fine search at +0.22 next to the original at 0, it is one sound, listed
    once (at 0 semitones the two correlate far less)."""
    lib, sounds = noisy
    n = int(0.6 * SR)
    copy = transpose(sounds[30], -0.22)
    entries = np.r_[lib.audio, (fit(copy, n) / np.max(np.abs(copy)))[None].astype(np.float16)]
    both = Library(lib.paths + ["/lib/copy.wav"], lib.roles + ["snare"], entries, lib.seconds)
    both.build_speckle()
    t = placed(sounds[30], 0.0, 150)
    ranked = both.match_fine(t, "snare", top=10)
    assert {ranked[0]["i"], ranked[1]["i"]} == {30, len(sounds)}
    assert not both.same(30, len(sounds), len(t)) and both.same(30, len(sounds), len(t), semitones=(0.0, 0.22))
    listed = [r["i"] for r in both.distinct(ranked, len(t))]
    assert listed[0] in (30, len(sounds)) and (30 in listed) != (len(sounds) in listed)


def test_two_kicks_at_one_pitch_are_two_sounds():
    """Two different sine-like kicks, the second tuned 2 semitones up:
    played at one pitch they agree in the full band, where the tone is
    (0.99), but not in the snare band (0.10), so distinct lists both. (A
    kit's copy at another pitch agrees in both: the test above.)"""
    n, length = int(0.6 * SR), samples(200)
    a, b = burst(500, 400, 30, 2000, 150, tone_hz=50.0), burst(501, 400, 30, 2000, 150, tone_hz=50 * 2 ** (2 / 12))
    entries = np.stack([fit(trim_onset(x), n) / np.max(np.abs(x)) for x in (a, b)]).astype(np.float16)
    kicks = Library(["/lib/kick a.wav", "/lib/kick b.wav"], ["kick", "kick"], entries, 0.6)
    played = kicks.candidate(0, length), kicks.candidate(1, length, -2.0)
    assert copies(*played, pitched=False)  # what the full band alone calls one recording
    assert not copies(*played, pitched=True) and not kicks.same(0, 1, length, semitones=(0.0, -2.0))
    ranked = [{"i": 1, "semitones": -2.0}, {"i": 0, "semitones": 0.0}]
    assert [r["i"] for r in kicks.distinct(ranked, length)] == [1, 0]


def test_windows_searched_together_rank_as_they_do_alone(noisy, monkeypatch):
    """prepare(): the windows of a band searched together get the rankings
    each gets alone (the same numbers), the library laid out once per band
    and block of entries, not once per window."""
    lib, sounds = noisy
    snares = [placed(sounds[k], d, 150) for k, d in ((30, 0.0), (33, -1.0), (31, 0.4))]
    kicks = [placed(sounds[k], d, 200) for k, d in ((120, 0.0), (121, -2.0))]
    alone = [lib.match_fine(t, "snare", semitones=range(-3, 4), union=10, top=10) for t in snares]
    alone += [lib.match(t, "kick", semitones=range(-4, 5), top=10) for t in kicks]
    blocks = []
    block = lib.block
    monkeypatch.setattr(lib, "block", lambda rows, length, semitones=0.0: blocks.append(length) or
                        block(rows, length, semitones))
    lib.prepare({"snare": snares, "kick": kicks})
    together = [lib.match_fine(t, "snare", semitones=range(-3, 4), union=10, top=10) for t in snares]
    together += [lib.match(t, "kick", semitones=range(-4, 5), top=10) for t in kicks]
    assert together == alone and sorted(blocks) == [samples(150), samples(200)]
    lib.forget()
    lib.match(kicks[0], "kick", top=10)
    assert len(blocks) == 3  # forgotten: searched again


def test_a_transposed_candidate_is_resampled_from_the_start_it_needs(noisy, monkeypatch):
    """Library.candidate resamples only the start of an entry a transposed
    window holds (and TRANSPOSE_MARGIN more): the samples are the same as
    from the whole entry."""
    from refdissect import library

    lib, _ = noisy
    read = []
    monkeypatch.setattr(library, "transpose", lambda x, st: read.append(len(x)) or transpose(x, st))
    for i in (0, 30, 61, 121):
        for st in (-4, -1.5, 0.0025, 1, 3.7):
            whole = fit(np.r_[np.zeros(samples(3), np.float32), transpose(lib.audio[i].astype(np.float32), st)],
                        samples(80))
            assert np.array_equal(lib.candidate(i, samples(80), st), whole)
    assert max(read) < lib.audio.shape[1]
