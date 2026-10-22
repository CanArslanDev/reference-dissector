"""pipeline.match's parts on synthetic one-shots: kicks left to whole
semitones, layered and plain templates, a kick under an 808, the role
rule, kick sources only when asked."""

import json

import numpy as np
import pytest
from conftest import burst, placed

from refdissect import pipeline
from refdissect.audio import SR, band, fit, samples, shift, transpose, write
from refdissect.detect import Template, detect


def test_kicks_keep_whole_semitones(noisy):
    """The kick band is searched by whole semitones, never refined at
    fractional pitch (that let other kicks win); the snare and hat bands at
    any pitch."""
    lib, sounds = noisy
    ranked, how = pipeline._search(lib, placed(sounds[-2], 0.3, 200), "kick", None)
    assert how == "semitones" and all(float(r["semitones"]).is_integer() for r in ranked)
    ranked, _ = pipeline._search(lib, placed(sounds[-2], 0.0, 200), "kick", None)
    assert ranked[0]["i"] == len(sounds) - 2 and ranked[0]["semitones"] == 0.0
    noisy_t = placed(sounds[30], 0.3, 150)
    assert pipeline._search(lib, noisy_t, "snare", None)[1] == "speckle"


def test_a_snare_found_in_the_low_band_keeps_its_cents(noisy, tmp_path):
    """discover() finds snares in the low band too (drum_pitch-2's, 0.70
    semitones down, was matched at -1 in the kick band, 30 cents off). Such
    a template is matched at any pitch in the snare band: a snare 0.47
    semitones down is found there at -0.47."""
    lib, sounds = noisy
    work = _kick_work(tmp_path, transpose(sounds[30], -0.47), hiss=0.0)
    out = pipeline.match(work, lib, log=lambda *a: None)["sound_1"]
    assert out["matched_in_band"] == "snare" and out["top"][0]["path"] == lib.paths[30]
    assert out["top"][0]["semitones"] == pytest.approx(-0.47, abs=0.01)


def test_a_layered_snare_and_clap_are_both_found(noisy):
    """A clap 4 dB under a snare and 3 ms after it: the snare is the top
    match and the clap its layer, at its level and offset."""
    lib, sounds = noisy
    clap = sounds[4]  # role clap
    t = placed(sounds[30], 0.0, 150, [(clap, 10 ** (-4 / 20), samples(3))])
    ranked, _ = pipeline._search(lib, t, "snare", None)
    assert ranked[0]["i"] == 30
    layers, evidence = pipeline._layer(lib, t, "snare", ranked, taken=[])
    assert evidence["accepted"] and [x["path"] for x in layers] == [lib.paths[4]]
    assert layers[0]["gain_db"] == pytest.approx(-4.0, abs=1.0) and layers[0]["lag_ms"] == pytest.approx(3.0, abs=0.5)
    assert layers[0]["transpose"] == 0 and abs(layers[0]["cents"]) <= 1.0


def test_a_plain_snare_has_no_layer(noisy):
    """A snare alone (and a little noise): no partner takes 3 dB off. With
    the song's hat on it, the hat explains more, but it is another
    template's sound: passed over, and nothing else is a layer."""
    lib, sounds = noisy
    rng = np.random.default_rng(7)
    plain = placed(sounds[30], 0.0, 150) + 0.01 * rng.standard_normal(samples(150)).astype(np.float32)
    ranked, _ = pipeline._search(lib, plain, "snare", None)
    layers, evidence = pipeline._layer(lib, plain, "snare", ranked, taken=[])
    assert layers == [] and not evidence["accepted"] and evidence["better_db"] < 3
    hat = sounds[14]
    with_hat = placed(sounds[30], 0.0, 150, [(hat, 0.5, 0)])
    ranked, _ = pipeline._search(lib, with_hat, "snare", None)
    layers, evidence = pipeline._layer(lib, with_hat, "snare", ranked, taken=[14])
    assert layers == [] and evidence["partner"] == lib.paths[14].rsplit("/", 1)[1] and evidence["better_db"] > 3
    assert evidence["why"] == "the partner is another template's sound"


def test_the_808_is_explained_before_a_kick_is_matched():
    """A kick under an 808 note: least squares in the kick band takes the
    808 out (lined up within 5 ms, never added), leaving the kick."""
    kick = burst(5, 250, 40, 6000, 60, tone_hz=60.0)
    tone = np.sin(2 * np.pi * 45 * np.arange(samples(200)) / SR).astype(np.float32)
    t = fit(kick, samples(200)) + 0.8 * shift(tone, samples(2))
    out = pipeline._explained(t, tone)
    kb = band(fit(kick, samples(200)), "kick")
    assert np.sum((band(out, "kick") - kb) ** 2) < 0.02 * np.sum(kb ** 2)
    assert np.array_equal(pipeline._explained(t, np.zeros_like(tone)), t)


def test_a_sound_heard_all_over_the_song_is_no_layer(noisy):
    """The hat on every snare explains the snare template best, but no
    template was given it: it is heard at 60 other moments of the stem, so
    it is a sound of its own. A clap heard only with the snares stays."""
    lib, sounds = noisy
    snare, clap, hat = sounds[30], sounds[4], sounds[14]
    snares, hats = [int((0.5 + k) * SR) for k in range(16)], [int((0.25 + 0.25 * k) * SR) for k in range(64)]
    for with_clap in (False, True):
        x = np.random.default_rng(3).standard_normal(int(17 * SR)).astype(np.float32) * 0.002
        for a in snares:
            x[a:a + len(snare)] += snare
            if with_clap:
                x[a + samples(3):a + samples(3) + len(clap)] += 0.6 * clap
        for a in hats:
            x[a:a + len(hat)] += 0.7 * hat
        others = [(clap, 0.6, samples(3))] if with_clap else [(hat, 0.7, 0)]
        t = placed(snare, 0.0, 150, others)
        ranked, _ = pipeline._search(lib, t, "snare", None)
        layers, evidence = pipeline._layer(lib, t, "snare", ranked, taken=[], drums=x, hits=np.array(snares) / SR)
        if with_clap:
            assert [r["path"] for r in layers] == [lib.paths[4]] and evidence["heard"]["elsewhere"] <= 2
        else:
            assert layers == [] and evidence["partner"] == lib.paths[14].rsplit("/", 1)[1]
            assert evidence["why"] == "the partner plays elsewhere in the song" and evidence["heard"]["elsewhere"] > 40


def test_a_kick_read_from_a_cleaner_source(noisy):
    """A kick template from a muddy drum stem: the same hits read from a
    cleaner source find the kick with a closer match, and win, unless that
    kick was given to another template."""
    lib, sounds = noisy
    kick, n = sounds[-2], int(4 * SR)
    starts = [int((0.3 + 0.5 * k) * SR) for k in range(6)]
    clean = np.zeros(n, np.float32)
    for a in starts:
        clean[a:a + len(kick)] += kick
    muddy = 0.3 * clean + 0.3 * np.random.default_rng(2).standard_normal(n).astype(np.float32)
    info = {"members": [(a, 0, 1.0, 1.0, 1.0) for a in starts]}
    ranked = lib.match(pipeline.template(muddy, pipeline.Group(info["members"]), 200)[0], "kick",
                       semitones=range(-4, 5), top=40)
    evidence, t, found = pipeline._kick_source(lib, info, ranked, {"mix": clean}, None, lambda path: False)
    assert evidence["source"] == "mix" and found[0]["i"] == len(sounds) - 2
    assert evidence["top_ncc"]["mix"] > evidence["top_ncc"]["drums"] + 0.1
    evidence, t, found = pipeline._kick_source(lib, info, ranked, {"mix": clean}, None, lambda path: True)
    assert evidence["source"] == "drums" and t is None and found is None


def test_kick_sources_take_the_808_out(tmp_path):
    """drums + bass and the mix, with the 808 notes played by fit-808's
    one-shot at their pitches; nothing but the drum stem without one."""
    stems, work = tmp_path / "stems", tmp_path / "work"
    n = int(2 * SR)
    drums, bass = np.zeros(n, np.float32), np.zeros(n, np.float32)
    write(stems / "drums.wav", drums, peak=None)
    write(stems / "bass.wav", bass, peak=None)
    write(tmp_path / "mix.wav", drums, peak=None)
    (stems / "separation.json").write_text(f'{{"audio": "{tmp_path / "mix.wav"}"}}')
    notes = [{"t": 0.2, "end": 0.7, "pitch": 30, "peak": 1.0}, {"t": 1.0, "end": 1.5, "pitch": 33, "peak": 0.5}]
    analysis = {"notes_808": notes}
    work.mkdir()
    sources, r808, info = pipeline.kick_sources(work, analysis, stems, drums)
    assert sources == {} and r808 is None and "run fit-808 first" in info["why"]
    shot = np.sin(2 * np.pi * 46.25 * np.arange(SR) / SR).astype(np.float32)
    write(work / "808 (synth).wav", shot, peak=None)
    (work / "synth808.json").write_text(f'{{"file": "{work / "808 (synth).wav"}", "root_midi": 30}}')
    sources, r808, info = pipeline.kick_sources(work, analysis, stems, drums)
    assert sorted(sources) == ["drums+bass-808", "mix-808"] and info == {"808_from": "synth808.json"}
    a, b = int(0.2 * SR), int(1.0 * SR)
    assert np.allclose(r808[a:a + 1000], shot[:1000], atol=1e-6)
    assert np.max(np.abs(r808[int(0.75 * SR):b])) == 0  # cut at its end
    up3 = r808[b:b + int(0.4 * SR)] / 0.5
    assert abs(np.argmax(np.abs(np.fft.rfft(up3))) * SR / len(up3) - 46.25 * 2 ** (3 / 12)) < 3


def test_a_close_match_names_the_role_in_the_snare_and_hat_bands():
    """A hat found at NCC 0.99 in the hat band is a hat, whatever the other
    four names say and however the template's energy guessed it; below
    0.95 the vote and the guess decide as before. Not in the kick band, nor
    for a family that is no drum: an open hat matched there to an fx file
    at 0.957, with three hats among the five, stays a hat, and a hat
    template matched to a kick at 0.952, with three percs, a perc."""
    from collections import Counter

    def named(top):
        return Counter(pipeline.FAMILY.get(r["role"], r["role"]) for r in top)

    top = [{"role": "hat", "ncc": 0.994}, {"role": "kick", "ncc": 0.4}, {"role": "perc", "ncc": 0.38},
           {"role": "other", "ncc": 0.37}, {"role": "other", "ncc": 0.35}]
    assert pipeline._role(top, named(top), "kick", "hat") == "hat"
    assert pipeline._role([{"role": "open_hat", "ncc": 0.97}], Counter({"hat": 1}), "kick", "snare") == "hat"
    assert pipeline._role([dict(top[0], ncc=0.9)] + top[1:], named(top), "kick", "hat") == "kick"
    assert pipeline._role([{"role": "hat", "ncc": 0.8}], Counter({"snare": 3, "hat": 2}), "kick", "hat") == "snare"
    fx = [{"role": "fx", "ncc": 0.957}, *({"role": "crash/ride", "ncc": v} for v in (0.885, 0.874, 0.84)),
          {"role": "fx", "ncc": 0.778}]
    assert pipeline._role(fx, named(fx), "hat", "kick") == "hat"
    assert pipeline._role(fx, named(fx), "snare", "hat") == "hat"  # fx is no drum family: the vote decides
    kick = [{"role": "kick", "ncc": 0.952}, {"role": "perc", "ncc": 0.835}, {"role": "perc", "ncc": 0.826},
            {"role": "fx", "ncc": 0.799}, {"role": "perc", "ncc": 0.768}]
    assert pipeline._role(kick, named(kick), "kick", "kick") == "perc"
    assert pipeline._role(kick, named(kick), "kick", "snare") == "kick"


def test_heard_is_detects_matched_filter(noisy):
    """_heard finds a sound where detect() finds it (the matched filter
    detect() starts from, at its threshold), from xcorr and scipy alone: a
    hat placed at 24 moments of a quiet stem is heard at each of them; a
    snare that never plays there is not heard."""
    lib, sounds = noisy
    hat = lib.candidate(14, samples(40))
    x = np.random.default_rng(4).standard_normal(int(7 * SR)).astype(np.float32) * 0.002
    at = np.array([int((0.2 + 0.27 * k) * SR) for k in range(24)])
    for a in at:
        x[a:a + len(sounds[14])] += 0.5 * sounds[14]
    found = pipeline._heard(x, hat, "hat")
    by_detect = detect(x, [Template("h", "hat", "hat", hat, 24)])["h"]
    assert len(found) == 24 and np.allclose(found, [e["t"] for e in by_detect], atol=1e-4)
    assert np.max(np.abs(found - at / SR)) <= 0.003
    assert not len(pipeline._heard(x, lib.candidate(30, samples(60)), "snare"))


def _kick_work(tmp_path, kick: np.ndarray, hiss: float = 0.3):
    """A work folder whose one template, found in the low band, is a kick
    heard 8 times in the drum stem (with hiss above the kick band on the
    template, so the kick band matches it best), stems with bass.wav and
    the mix they came from."""
    stems, work = tmp_path / "stems", tmp_path / "work"
    n = int(4.5 * SR)
    starts = [int((0.3 + 0.5 * k) * SR) for k in range(8)]
    drums = np.random.default_rng(5).standard_normal(n).astype(np.float32) * 0.002
    for a in starts:
        drums[a:a + len(kick)] += kick
    write(stems / "drums.wav", drums, peak=None)
    write(stems / "bass.wav", np.zeros(n, np.float32), peak=None)
    write(tmp_path / "mix.wav", drums, peak=None)
    (stems / "separation.json").write_text(json.dumps({"audio": str(tmp_path / "mix.wav")}))
    (work / "templates").mkdir(parents=True)
    np.save(work / "templates" / "sound_1.npy",
            placed(kick, 0.0, 350) + hiss * fit(burst(6, 350, 10000, 16000, 1000), samples(350)))
    analysis = {"source": {"stems": str(stems)}, "notes_808": [],
                "templates": {"sound_1": {"role_guess": "kick", "band": "low",
                                          "members": [[a, 0, 1.0, 1.0, 1.0] for a in starts]}}}
    (work / "analysis.json").write_text(json.dumps(analysis))
    return work


def test_kick_sources_only_when_asked(noisy, tmp_path):
    """match() reads a kick template from drums + bass and the mix only with
    kick_sources_on: by default matches.json names no source. Here there is
    no 808 note, so both sources are read as they are."""
    lib, sounds = noisy
    work = _kick_work(tmp_path, sounds[-2])
    out = pipeline.match(work, lib, log=lambda *a: None)["sound_1"]
    assert out["matched_in_band"] == "kick" and out["top"][0]["path"] == lib.paths[len(sounds) - 2]
    assert "source" not in out and "kick_source" not in out
    on = pipeline.match(work, lib, kick_sources_on=True, log=lambda *a: None)["sound_1"]
    assert on["kick_source"]["808_from"] is None and set(on["kick_source"]["top_ncc"]) == {"drums", "drums+bass",
                                                                                         "mix"}
