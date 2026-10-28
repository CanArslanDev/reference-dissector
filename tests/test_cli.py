import pytest

from refdissect.cli import main


def test_help_lists_the_steps(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for step in ("inventory", "index-library", "separate", "analyze", "melodic", "match", "fit-808", "plan",
                 "live-build", "calibrate", "suite"):
        assert step in out


def test_analyze_passes_the_grid_options(monkeypatch):
    from refdissect import pipeline
    seen = {}
    monkeypatch.setattr(pipeline, "analyze", lambda *a, **kw: seen.update(args=a, kw=kw))
    main(["analyze", "stems", "--work", "w", "--no-tracker", "--bpm", "137", "--meter", "3/4"])
    assert seen["args"] == ("stems", "w")
    assert seen["kw"] == {"bpm_range": (100.0, 180.0), "bpm": 137.0, "meter": "3/4", "tracker": False}
    main(["analyze", "stems", "--work", "w"])
    assert seen["kw"] == {"bpm_range": (100.0, 180.0), "bpm": None, "meter": None, "tracker": True}
    main(["analyze", "stems", "--work", "w", "--no-cache"])
    assert seen["kw"]["cache"] is False
