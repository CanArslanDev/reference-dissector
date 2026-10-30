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
    assert seen["kw"] == {"bpm_range": (100.0, 180.0), "bpm": 137.0, "meter": "3/4", "tracker": False, "cache": True}
    main(["analyze", "stems", "--work", "w"])
    assert seen["kw"] == {"bpm_range": (100.0, 180.0), "bpm": None, "meter": None, "tracker": True, "cache": True}
    main(["analyze", "stems", "--work", "w", "--no-cache"])
    assert seen["kw"]["cache"] is False


def test_the_commands_keep_by_default_and_the_suite_only_when_asked(monkeypatch):
    """analyze and fit-808 keep their 808 results in work/cache unless
    --no-cache; a suite run keeps nothing unless --cache, so its numbers
    and seconds are the code's (--no-cache is the default, and accepted)."""
    from refdissect import pipeline, suite
    seen = {}
    monkeypatch.setattr(pipeline, "fit_808", lambda *a, **kw: seen.update(fit=kw))
    main(["fit-808", "w"])
    assert seen["fit"]["cache"] is True
    main(["fit-808", "w", "--no-cache"])
    assert seen["fit"]["cache"] is False

    def run(*a, **kw):
        seen.update(suite=kw)
        return {"beats": 0, "failed": [], "overall": {"analysis": {}, "plan": {}}}
    monkeypatch.setattr(suite, "run", run)
    for flags, cache in (([], False), (["--cache"], True), (["--no-cache"], False)):
        main(["suite", "run", "d", "--library", "l", "--tag", "t", *flags])
        assert seen["suite"]["cache"] is cache, flags
