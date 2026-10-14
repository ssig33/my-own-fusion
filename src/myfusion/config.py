"""Configuration loading for my-own-fusion.

Configuration is resolved from three layers, in increasing priority:

1. the built-in defaults below,
2. environment variables (optionally seeded from a local ``.env`` file), and
3. explicit command-line flags handled in :mod:`myfusion.cli`.

No secret is ever hardcoded; the API key must come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# A small, cheap, provider-diverse default panel. Diversity is the point of
# Fusion: panel members should disagree often enough to be worth comparing.
# Swap these for whatever is currently available on your account; see
# https://openrouter.ai/models for the live catalogue.
DEFAULT_PANEL_MODELS: tuple[str, ...] = (
    "openai/gpt-6-luna",
    "anthropic/claude-3.5-haiku",
    "google/gemini-2.0-flash-001",
)

# The judge compares the panel responses and emits structured JSON analysis.
DEFAULT_JUDGE_MODEL = "openai/gpt-6-luna"

# The synthesizer reads the analysis and writes the final answer.
DEFAULT_SYNTH_MODEL = "anthropic/claude-3.5-sonnet"


def load_dotenv(path: str | os.PathLike[str] = ".env") -> None:
    """Populate ``os.environ`` from a ``.env`` file if it exists.

    This is a deliberately tiny parser (``KEY=VALUE`` per line, ``#`` comments)
    so the project has no hard dependency on python-dotenv. Existing environment
    variables always win, so exporting a value in your shell overrides the file.
    """

    env_path = Path(path)
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _split_models(value: str | None) -> list[str] | None:
    """Parse a comma-separated model list, dropping empties."""

    if not value:
        return None
    models = [item.strip() for item in value.split(",") if item.strip()]
    return models or None


@dataclass
class Config:
    """Resolved runtime configuration for a single fusion run."""

    api_key: str
    base_url: str = DEFAULT_BASE_URL
    panel_models: list[str] = field(default_factory=lambda: list(DEFAULT_PANEL_MODELS))
    judge_model: str = DEFAULT_JUDGE_MODEL
    synth_model: str = DEFAULT_SYNTH_MODEL
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables.

        Raises:
            ValueError: if ``OPENROUTER_API_KEY`` is missing or empty.
        """

        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or export OPENROUTER_API_KEY in your shell."
            )

        base_url = os.environ.get("OPENROUTER_BASE_URL", DEFAULT_BASE_URL).strip()
        panel = _split_models(os.environ.get("FUSION_PANEL_MODELS"))
        judge = os.environ.get("FUSION_JUDGE_MODEL", "").strip() or DEFAULT_JUDGE_MODEL
        synth = os.environ.get("FUSION_SYNTH_MODEL", "").strip() or DEFAULT_SYNTH_MODEL

        return cls(
            api_key=api_key,
            base_url=base_url,
            panel_models=panel if panel is not None else list(DEFAULT_PANEL_MODELS),
            judge_model=judge,
            synth_model=synth,
        )
