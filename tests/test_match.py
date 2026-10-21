"""pipeline.match's parts on synthetic one-shots: kicks left to whole
semitones, layered and plain templates, a kick under an 808."""

import numpy as np
import pytest
from conftest import burst, placed

from refdissect import pipeline
from refdissect.audio import SR, band, fit, samples, shift


def test_kicks_keep_whole_semitones(noisy):
    """A tonal template is searched by whole semitones in every band, never
    refined at fractional pitch (that let other kicks win)."""
    lib, sounds = noisy
    kick = placed(sounds[-2], 0.3, 200)
    for b in ("kick", "snare", "hat"):
        ranked, how = pipeline._search(lib, kick[: samples({"kick": 200, "snare": 150, "hat": 80}[b])], b, True,
                                       None)
        assert how == "semitones" and all(float(r["semitones"]).is_integer() for r in ranked)
    ranked, _ = pipeline._search(lib, placed(sounds[-2], 0.0, 200), "kick", True, None)
    assert ranked[0]["i"] == len(sounds) - 2 and ranked[0]["semitones"] == 0.0
    noisy_t = placed(sounds[30], 0.3, 150)
    assert pipeline._search(lib, noisy_t, "snare", False, None)[1] == "speckle"


def test_a_layered_snare_and_clap_are_both_found(noisy):
    """A clap 4 dB under a snare and 3 ms after it: the snare is the top
    match and the clap its layer, at its level and offset."""
    lib, sounds = noisy
    clap = sounds[4]  # role clap
    t = placed(sounds[30], 0.0, 150, [(clap, 10 ** (-4 / 20), samples(3))])
    ranked, _ = pipeline._search(lib, t, "snare", False, None)
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
    ranked, _ = pipeline._search(lib, plain, "snare", False, None)
    layers, evidence = pipeline._layer(lib, plain, "snare", ranked, taken=[])
    assert layers == [] and not evidence["accepted"] and evidence["better_db"] < 3
    hat = sounds[14]
    with_hat = placed(sounds[30], 0.0, 150, [(hat, 0.5, 0)])
    ranked, _ = pipeline._search(lib, with_hat, "snare", False, None)
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
        ranked, _ = pipeline._search(lib, t, "snare", False, None)
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
    from refdissect.audio import write
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
