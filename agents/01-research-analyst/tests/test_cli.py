"""CLI: the review prompt, prompts never polluting the report on stdout, and exit codes."""

import pytest

from research_analyst import cli
from research_analyst.core.config import Settings
from research_analyst.state import Report
from research_analyst.storage import DatabaseUnavailable
from research_analyst.tools.edgar import SecAccessDenied

SETTINGS = Settings(_env_file=None, gcp_project="p", sec_user_agent="app dev@mycompany.io")
STATE = {"report": Report(title="Apple", summary="s", body_markdown="b"), "review_rounds": 0}


def answers(monkeypatch, *typed):
    it = iter(typed)
    monkeypatch.setattr("builtins.input", lambda: next(it))


@pytest.mark.parametrize("typed,expected", [
    (["a"], ("approve", "")),
    (["j"], ("reject", "")),
    (["r", "Shorter summary"], ("revise", "Shorter summary")),
    (["x", "r", "", "r", "Add risks", ], ("revise", "Add risks")),  # bad choice, then empty feedback: asked again
])
def test_review_prompt(monkeypatch, typed, expected):
    answers(monkeypatch, *typed)
    d = cli._ask_review(STATE, SETTINGS)
    assert (d.action, d.feedback) == expected


@pytest.mark.parametrize("typed", [["q"], [""]])
def test_decide_later(monkeypatch, typed):
    answers(monkeypatch, *typed)
    assert cli._ask_review(STATE, SETTINGS) is None


def test_revise_not_offered_when_the_limit_is_used(monkeypatch, capsys):
    answers(monkeypatch, "r", "a")
    d = cli._ask_review({**STATE, "review_rounds": 3}, SETTINGS)
    assert d.action == "approve"
    err = capsys.readouterr().err
    assert "[r]evise" not in err.split("Review (3/3")[-1]


def test_prompts_go_to_stderr_so_a_redirected_report_stays_clean(monkeypatch, capsys):
    answers(monkeypatch, "a")
    cli._ask_review(STATE, SETTINGS)
    out = capsys.readouterr()
    assert out.out == ""                       # nothing on stdout
    assert "[a]pprove" in out.err              # the prompt went to stderr


@pytest.mark.parametrize("error,code,message", [
    (DatabaseUnavailable("cannot connect"), 2, "Database problem"),
    (SecAccessDenied("403"), 1, "SEC refused access"),
    (KeyboardInterrupt(), 130, "resume"),
])
def test_exit_codes(monkeypatch, capsys, error, code, message):
    async def failing(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(cli, "_session", failing)
    assert cli.main(["run", "Analyze Apple"]) == code
    assert message in capsys.readouterr().err


def test_config_error_exit_code_2(monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, gcp_project="p", sec_user_agent="bad"))
    assert cli.main(["list"]) == 2
    assert "Configuration error" in capsys.readouterr().err


def test_subcommand_is_required(capsys):
    with pytest.raises(SystemExit):
        cli.main([])
