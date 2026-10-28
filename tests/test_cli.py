"""Tests for the command-line entry point (no network access)."""

from __future__ import annotations

import pytest

from myfusion import __version__
from myfusion.cli import _print_stages, main
from myfusion.pipeline import FusionResult, JudgeAnalysis, PanelResponse


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # main() reads ./.env, so run from an empty directory.
    monkeypatch.chdir(tmp_path)
    for key in ("OPENROUTER_API_KEY", "FUSION_PANEL_MODELS"):
        monkeypatch.delenv(key, raising=False)


def test_main_without_api_key_returns_2(capsys) -> None:
    assert main(["hello"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_main_with_empty_prompt_returns_2(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main(["   "]) == 2
    assert "the prompt is empty" in capsys.readouterr().err


def test_main_with_empty_panel_returns_2(monkeypatch, capsys) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert main(["hello", "--panel", " , ,"]) == 2
    assert "panel model list is empty" in capsys.readouterr().err


def test_version_flag(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"my-own-fusion {__version__}"


def _result(analysis: JudgeAnalysis) -> FusionResult:
    return FusionResult(
        prompt="q",
        panel=[PanelResponse(model="a/x", content="answer"), PanelResponse(model="b/y", error="boom")],
        analysis=analysis,
        final_answer="final",
        judge_model="j/z",
        synth_model="s/w",
    )


def test_print_stages_lists_sections(capsys) -> None:
    _print_stages(_result(JudgeAnalysis(consensus=["agreed"], raw="{}")))
    out = capsys.readouterr().out

    assert "--- a/x ---\nanswer" in out
    assert "--- b/y ---\n(failed: boom)" in out
    assert "Consensus:\n  - agreed" in out
    assert "Blind spots:\n  (none)" in out
    assert "(raw judge output)" not in out


def test_print_stages_falls_back_to_raw(capsys) -> None:
    _print_stages(_result(JudgeAnalysis(raw="judge rambled")))
    assert "(raw judge output)\njudge rambled" in capsys.readouterr().out
