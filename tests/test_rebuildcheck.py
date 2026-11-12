"""mastering.master_like, rebuildcheck's null tests and check-rebuild end to
end on synthetic audio, with a stand-in for Demucs: no library, no Demucs."""

import json

import numpy as np
import pytest
import soundfile as sf

from refdissect.audio import SR, attack_time
from refdissect.evaluate import SCHEMA


def white(seconds, seed=0, amp=0.1):
    return np.random.default_rng(seed).normal(0, amp, int(seconds * SR))


def pink(seconds, seed=0):
    n = int(seconds * SR)
    spec = np.fft.rfft(np.random.default_rng(seed).normal(size=n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spec, n)
    return 0.1 * x / x.std()


def test_tilt_reads_white_and_pink_noise():
    from refdissect.mastering import apply_tilt, tilt
    assert tilt(white(10), SR) == pytest.approx(3.0, abs=0.3)
    assert tilt(pink(10), SR) == pytest.approx(0.0, abs=0.3)
    assert tilt(apply_tilt(pink(10), SR, -4.0), SR) == pytest.approx(-4.0, abs=0.3)


def test_the_limiter_holds_its_ceiling_and_leaves_quiet_audio_alone():
    from refdissect.mastering import limit
    x = white(4, amp=0.3)
    y = limit(x, SR, 0.5)
    assert np.abs(y).max() <= 0.5 + 1e-12
    quiet = white(2, amp=0.01)
    assert np.array_equal(limit(quiet, SR, 0.9)[:, 0], quiet)


def test_master_like_takes_the_references_loudness_tilt_and_ceiling():
    from refdissect.mastering import master_like, profile
    ref = np.tanh(3 * pink(12, seed=1)) * 0.9  # loud, clipped, about pink
    rebuild = 0.05 * white(12, seed=2)  # quiet and bright
    p = profile(ref, SR)
    y, rep = master_like(rebuild, SR, p)
    after = profile(y, SR)
    assert after["lufs"] == pytest.approx(p["lufs"], abs=0.1)
    assert after["peak"] <= p["peak"] + 1e-9
    assert after["tilt_db_per_octave"] == pytest.approx(p["tilt_db_per_octave"], abs=0.4)
    assert rep["max_reduction_db"] < 0 and rep["gain_db"] > 10


def test_null_scores_copies_shifts_and_silence():
    from refdissect.rebuildcheck import null
    x = white(3, seed=3)
    same = null(x, 0.5 * x)
    assert same["null_db"] < -100 and same["spectral_null_db"] < -100
    assert same["level_db"] == pytest.approx(-6.02, abs=0.01)
    late = null(x, np.r_[np.zeros(4), x[:-4]])  # 0.09 ms late: the waveform does not null, the magnitudes do
    assert late["null_db"] > -1 and late["spectral_null_db"] < -15
    other = null(x, white(3, seed=4))
    assert other["null_db"] > -0.1 and other["spectral_null_db"] > -12
    assert null(x, np.zeros_like(x)) == {"null_db": 0.0, "spectral_null_db": 0.0, "level_db": None}
    assert null(np.zeros(10), x)["null_db"] is None


def wav(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(x, np.float32), SR, subtype="FLOAT")
    return str(path)


def fake_demucs(audio, out):
    """Stand-in for Demucs: under 150 Hz to drums (as a kick), over 4 kHz
    to drums too, the band between to other, nothing to bass or vocals."""
    from scipy.signal import butter, sosfiltfilt
    x, _ = sf.read(audio, always_2d=True)
    lo = sosfiltfilt(butter(4, 150, "low", fs=SR, output="sos"), x, axis=0)
    hi = sosfiltfilt(butter(4, 4000, "high", fs=SR, output="sos"), x, axis=0)
    out.mkdir(parents=True, exist_ok=True)
    for name, y in (("drums", lo + hi), ("other", x - lo - hi), ("bass", 0 * x), ("vocals", 0 * x)):
        sf.write(out / f"{name}.wav", y, SR, subtype="FLOAT")


def song(tmp_path, kit, seconds=8.0, snare_gain=1.0):
    """At 120 BPM a kick every beat, a snare on 2 and 4, a 1 kHz pad: the
    truth stems, a mastered mix (tanh) and its fake separation."""
    n = int(seconds * SR)
    kick, snare = np.zeros(n), np.zeros(n)
    kicks, snares = np.arange(0.0, seconds - 0.5, 0.5), np.arange(0.5, seconds - 0.5, 1.0)
    for t in kicks:
        kick[int(t * SR):int(t * SR) + len(kit["kick"])] += 0.5 * kit["kick"]
    for t in snares:
        snare[int(t * SR):int(t * SR) + len(kit["snare"])] += 0.4 * snare_gain * kit["snare"]
    pad = 0.05 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)
    truth_dir = tmp_path / "beat"
    stems = {"kick": kick, "snare": snare, "drums": kick + snare, "other": pad}
    for k, x in stems.items():
        wav(truth_dir / "stems_truth" / f"{k}.wav", x)
    mix = np.tanh(2 * (kick + snare + pad)) * 0.9
    wav(truth_dir / "mix.wav", np.stack([mix, mix], 1))
    fake_demucs(truth_dir / "mix.wav", truth_dir / "stems")
    events = {"schema": SCHEMA, "kind": "truth", "scenario": "t", "seed": 0,
              "sounds": {"kick": {"role": "kick"}, "snare": {"role": "snare"}},
              "audio": {"sr": SR, "n_samples": n, "mix": "mix.wav",
                        "stems": {k: f"stems_truth/{k}.wav" for k in stems}},
              "drums": sorted([{"sound": "kick", "role": "kick", "t": float(t),
                                "t_attack": float(t) + attack_time(kit["kick"], "kick"), "gain": 0.5}
                               for t in kicks] +
                              [{"sound": "snare", "role": "snare", "t": float(t),
                                "t_attack": float(t) + attack_time(kit["snare"], "snare"), "gain": 0.4}
                               for t in snares], key=lambda e: e["t_attack"])}
    (truth_dir / "events.json").write_text(json.dumps(events))
    return truth_dir, events, kicks, snares


def plan_for(tmp_path, kit, truth_dir, kicks, snares, snare_shift=0.0):
    pads = {"36": {"path": wav(tmp_path / "kit" / "kick.wav", kit["kick"]), "transpose": 0, "role": "kick"},
            "38": {"path": wav(tmp_path / "kit" / "snare.wav", kit["snare"]), "transpose": 0, "role": "snare"}}
    notes = [{"pitch": 36, "start_time": 2 * t, "duration": 0.25, "velocity": 127} for t in kicks]
    notes += [{"pitch": 38, "start_time": 2 * (t + snare_shift), "duration": 0.25, "velocity": 127} for t in snares]
    stems = {"drums": {"path": str(truth_dir / "stems" / "drums.wav"), "mute": True},
             "other": {"path": str(truth_dir / "stems" / "other.wav"), "mute": False}}
    return {"tempo": 120.0, "length_beats": 16.0, "tracks": {"drums": {"pads": pads, "notes": notes}, "808": {}},
            "stems": stems, "levels": {"headroom_db": 4.0, "drums_track_db": -4.0, "stems_track_db": -4.0}}


def test_check_rebuild_end_to_end(kit, tmp_path):
    """A plan that plays the truth's kit on the truth's times nulls well per
    sound; one whose snare comes 30 ms late does not null its snare, and the
    kick stays as it was."""
    from refdissect.rebuildcheck import check_rebuild
    truth_dir, events, kicks, snares = song(tmp_path, kit)
    good = plan_for(tmp_path, kit, truth_dir, kicks, snares)
    r = check_rebuild(good, None, truth_dir / "mix.wav", truth_dir / "stems", tmp_path / "good", fake_demucs,
                      truth=events, truth_dir=truth_dir)
    assert r["master"]["after"]["lufs"] == pytest.approx(r["master"]["reference"]["lufs"], abs=0.1)
    assert (tmp_path / "good" / "stems" / "drums.wav").is_file() and (tmp_path / "good" / "check.json").is_file()
    sounds = r["sounds"]["sounds"]
    assert sounds["kick"]["pads"] == ["36"] and sounds["snare"]["pads"] == ["38"]
    assert sounds["kick"]["null_db"] < -30 and sounds["snare"]["null_db"] < -30
    assert r["sounds"]["drums_group"]["spectral_null_db"] < -15  # the pads play 0.5 : 0.4 at one level
    assert r["before"]["summary"]["measures"] > 20 and r["after"]["summary"]["measures"] > 20
    assert r["before"]["stem_nulls"]["other"]["spectral_null_db"] < -60  # the reference's own stem, played back
    assert r["after"]["stem_nulls"]["drums"]["spectral_null_db"] < -10
    assert r["after"]["stem_nulls"]["bass"]["null_db"] is None  # the stand-in leaves the bass silent
    assert set(r["shift"]["rows"]) == set(r["before"]["rows"]) & set(r["after"]["rows"])
    late = plan_for(tmp_path, kit, truth_dir, kicks, snares, snare_shift=0.03)
    s = check_rebuild(late, None, truth_dir / "mix.wav", truth_dir / "stems", tmp_path / "late", fake_demucs,
                      truth=events, truth_dir=truth_dir)["sounds"]["sounds"]
    assert s["snare"]["null_db"] > -3 and s["snare"]["spectral_null_db"] > sounds["snare"]["spectral_null_db"] + 10
    assert s["kick"]["null_db"] == pytest.approx(sounds["kick"]["null_db"], abs=0.5)


def test_check_rebuild_without_separation_and_the_command(kit, tmp_path, capsys):
    from refdissect.cli import main
    truth_dir, events, kicks, snares = song(tmp_path, kit)
    work = tmp_path / "work"
    work.mkdir()
    (work / "plan.json").write_text(json.dumps(plan_for(tmp_path, kit, truth_dir, kicks, snares)))
    assert main(["check-rebuild", str(work), "--truth", str(truth_dir / "events.json"), "--no-separate"]) == 0
    out = capsys.readouterr().out
    assert "before:" in out and "after:" not in out and "kick" in out
    r = json.loads((work / "check" / "check.json").read_text())
    assert "after" not in r and r["sounds"]["sounds"]["kick"]["null_db"] < -30
    # with the stems given, the same evaluation runs after as well, without Demucs
    fake_demucs(work / "check" / "rebuild_master.wav", tmp_path / "given")
    assert main(["check-rebuild", str(work), "--reference", str(truth_dir / "mix.wav"),
                 "--rebuild-stems", str(tmp_path / "given"), "--out", str(tmp_path / "c2")]) == 0
    r2 = json.loads((tmp_path / "c2" / "check.json").read_text())
    assert "after" in r2 and "sounds" not in r2
