import pytest

from refdissect.cli import main


def test_help_lists_the_steps(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for step in ("inventory", "index-library", "separate", "analyze", "melodic", "match", "fit-808", "plan",
                 "live-build", "calibrate"):
        assert step in out
