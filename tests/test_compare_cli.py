"""symmetric.compare, rounds.listening_round and the commands that run them,
on synthetic stems whose differences are known."""

import json

import numpy as np
import soundfile as sf

from refdissect.audio import SR


def write_stems(folder, duck_db, seconds=8.0, hat_db=-19.0):
    """drums (kick every 0.5 s, a hat between), bass (60 Hz, ducked duck_db at
    each kick, back in 0.3 s) and other (a 440 Hz pad)."""
    folder.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    kicks = np.arange(0.25, seconds - 0.5, 0.5)
    drums = np.zeros_like(t)
    for k in kicks:
        i = int(k * SR)
        n = int(0.15 * SR)
        drums[i:i + n] += np.sin(2 * np.pi * 55 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.05 * SR))
        j = int((k + 0.25) * SR)
        drums[j:j + 400] += 10 ** (hat_db / 20) * rng.normal(size=400)
    g = np.zeros_like(t)
    for k in kicks:
        m = (t >= k) & (t < k + 0.3)
        u = t[m] - k
        g[m] = np.minimum(g[m], duck_db * np.minimum(u / 0.01, 1) * (1 - u / 0.3))
    bass = 0.3 * np.sin(2 * np.pi * 60 * t) * 10 ** (g / 20)
    other = 0.1 * np.sin(2 * np.pi * 440 * t)
    for name, x in (("drums", drums), ("bass", bass), ("other", other)):
        sf.write(folder / f"{name}.wav", np.stack([x, x], 1), SR)
    return folder


def test_compare_sees_missing_ducking(tmp_path):
    from refdissect.symmetric import baseline, compare, load_stems, measure
    ref = load_stems(write_stems(tmp_path / "ref", duck_db=-20))
    flat = load_stems(write_stems(tmp_path / "flat", duck_db=0))
    a, b = measure(ref), measure(flat)
    assert a["ducking"]["808"]["depth_db"] < -12 and abs(b["ducking"]["808"]["depth_db"]) < 3
    rows = compare(a, b, baseline(ref))
    assert rows["ducking.808.curve_distance_db"]["diff"] > 5
    assert rows["ducking.808.depth_db"]["beyond_spread"]
    assert not rows["stems.other.lufs"]["beyond_spread"]  # the pad is the same on both sides
    assert a["drum_roles_db"]["kick"] == 0.0 and a["drum_roles_db"]["hat"] < -10


def test_listening_round_numbers_the_files(tmp_path):
    from refdissect.rounds import listening_round
    t = np.arange(2 * SR) / SR
    ref = tmp_path / "ref.wav"
    sf.write(ref, 0.1 * np.sin(2 * np.pi * 220 * t), SR)
    cands = []
    for k, f in enumerate((330, 440)):
        p = tmp_path / f"cand{k}.wav"
        sf.write(p, 0.5 * np.sin(2 * np.pi * f * t), SR)
        cands.append(p)
    m = listening_round(ref, cands, tmp_path / "round", "Which pad?", (0.0, 1.0), (0.0, 1.0))
    assert [r["number"] for r in m["files"]] == [1, 2]
    assert (tmp_path / "round" / "1.wav").is_file() and (tmp_path / "round" / "2.wav").is_file()
    text = (tmp_path / "round" / "manifest.md").read_text()
    assert "Which pad?" in text and "cand1.wav" in text


def test_commands_run(tmp_path, capsys):
    from refdissect.cli import main
    write_stems(tmp_path / "ref", duck_db=-20)
    write_stems(tmp_path / "new", duck_db=0)
    assert main(["compare", str(tmp_path / "ref"), str(tmp_path / "new"), "--out", str(tmp_path / "c.json")]) == 0
    out = capsys.readouterr().out
    assert "ducking.808.curve_distance_db" in out and "differ by more" in out
    assert json.loads((tmp_path / "c.json").read_text())["rows"]
    assert main(["space", str(tmp_path / "ref" / "other.wav"), "--spans", "0-4", "4-8"]) == 0
    assert "second_vs_first" in capsys.readouterr().out
    (tmp_path / "targets.json").write_text(json.dumps({"bass": -20, "other": {"target_db": -26, "spans": ["0-4"]}}))
    assert main(["mix-gains", str(tmp_path / "ref"), "--targets", str(tmp_path / "targets.json")]) == 0
    gains = json.loads(capsys.readouterr().out)
    assert set(gains) == {"bass", "other"} and gains["other"]["gain_db"] is not None
    assert main(["listening-round", str(tmp_path / "ref" / "other.wav"), str(tmp_path / "new" / "other.wav"),
                 "--question", "Pad?", "--out", str(tmp_path / "r")]) == 0
    assert (tmp_path / "r" / "1.wav").is_file()
