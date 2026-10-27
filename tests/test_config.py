"""Tests for .env loading and environment-based configuration."""

from __future__ import annotations

import os

import pytest

from myfusion.config import (
    DEFAULT_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PANEL_MODELS,
    DEFAULT_SYNTH_MODEL,
    Config,
    load_dotenv,
    split_models,
)

ENV_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "FUSION_PANEL_MODELS",
    "FUSION_JUDGE_MODEL",
    "FUSION_SYNTH_MODEL",
    "MYFUSION_TEST_A",
    "MYFUSION_TEST_B",
    "MYFUSION_TEST_C",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch restores (or removes) every key that
    # load_dotenv writes straight into os.environ.
    for key in ENV_KEYS:
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_load_dotenv_parses_lines_and_strips_quotes(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment\n"
        "\n"
        "MYFUSION_TEST_A = \"double quoted\"\n"
        "MYFUSION_TEST_B='single quoted'\n"
        "not a key value line\n"
        "MYFUSION_TEST_C=a=b\n",
        encoding="utf-8",
    )

    load_dotenv(env_file)

    assert os.environ["MYFUSION_TEST_A"] == "double quoted"
    assert os.environ["MYFUSION_TEST_B"] == "single quoted"
    assert os.environ["MYFUSION_TEST_C"] == "a=b"


def test_load_dotenv_keeps_existing_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MYFUSION_TEST_A", "from shell")
    env_file = tmp_path / ".env"
    env_file.write_text("MYFUSION_TEST_A=from file\n", encoding="utf-8")

    load_dotenv(env_file)

    assert os.environ["MYFUSION_TEST_A"] == "from shell"


def test_load_dotenv_ignores_missing_file(tmp_path) -> None:
    load_dotenv(tmp_path / "does-not-exist.env")


def test_split_models() -> None:
    assert split_models(" a/x , ,b/y,") == ["a/x", "b/y"]
    assert split_models(" , ") is None
    assert split_models("") is None
    assert split_models(None) is None


def test_from_env_requires_api_key(monkeypatch) -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Config.from_env()

    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        Config.from_env()


def test_from_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    config = Config.from_env()

    assert config.api_key == "test-key"
    assert config.base_url == DEFAULT_BASE_URL
    assert config.panel_models == list(DEFAULT_PANEL_MODELS)
    assert config.judge_model == DEFAULT_JUDGE_MODEL
    assert config.synth_model == DEFAULT_SYNTH_MODEL


def test_from_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("FUSION_PANEL_MODELS", "a/x, b/y")
    monkeypatch.setenv("FUSION_JUDGE_MODEL", "j/z")
    monkeypatch.setenv("FUSION_SYNTH_MODEL", "s/w")

    config = Config.from_env()

    assert config.base_url == "https://example.invalid/v1"
    assert config.panel_models == ["a/x", "b/y"]
    assert config.judge_model == "j/z"
    assert config.synth_model == "s/w"
