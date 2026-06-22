"""Offline tests for the fusion pipeline.

These tests never hit the network: they inject an ``httpx.MockTransport`` into
the real :class:`OpenRouterClient`, so they exercise the actual request-building
and response-parsing code paths while staying fully deterministic and free.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from myfusion.client import OpenRouterClient
from myfusion.pipeline import _extract_json_object, run_fusion

JUDGE_PAYLOAD = {
    "consensus": ["The sky appears blue due to scattering of light."],
    "contradictions": ["Model A says Rayleigh scattering; Model B blames the ocean."],
    "partial_coverage": ["Only one model mentioned wavelength dependence."],
    "unique_insights": ["Sunsets are red because of longer path length."],
    "blind_spots": ["None discussed atmospheric composition changes."],
}


def _handler(request: httpx.Request) -> httpx.Response:
    """Route mock completions by the model field in the request body."""

    body = json.loads(request.content)
    model = body["model"]
    messages = body["messages"]
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

    if "impartial judge" in system:
        content = json.dumps(JUDGE_PAYLOAD)
    elif "final author" in system:
        content = "The sky is blue because of Rayleigh scattering of sunlight."
    else:
        # A panel member: echo which model answered.
        content = f"Answer from {model}: light scatters in the atmosphere."

    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": content}}]},
    )


def _make_client() -> OpenRouterClient:
    transport = httpx.MockTransport(_handler)
    http_client = httpx.AsyncClient(transport=transport)
    return OpenRouterClient("test-key", "https://example.invalid/v1", http_client=http_client)


def test_full_pipeline_offline() -> None:
    async def scenario() -> None:
        async with _make_client() as client:
            result = await run_fusion(
                client,
                "Why is the sky blue?",
                panel_models=["openai/gpt-4o-mini", "anthropic/claude-3.5-haiku"],
                judge_model="openai/gpt-4o",
                synth_model="anthropic/claude-3.5-sonnet",
            )

        assert len(result.panel) == 2
        assert all(response.ok for response in result.panel)
        assert result.analysis.consensus == JUDGE_PAYLOAD["consensus"]
        assert result.analysis.blind_spots == JUDGE_PAYLOAD["blind_spots"]
        assert "Rayleigh" in result.final_answer

        # The result must round-trip cleanly to JSON for the --json flag.
        as_json = json.dumps(result.to_dict())
        assert "final_answer" in as_json

    asyncio.run(scenario())


def test_panel_failure_is_tolerated() -> None:
    """One broken panel model must not abort the whole run."""

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "broken/model":
            return httpx.Response(500, text="upstream exploded")
        return _handler(request)

    async def scenario() -> None:
        transport = httpx.MockTransport(flaky_handler)
        http_client = httpx.AsyncClient(transport=transport)
        async with OpenRouterClient(
            "k", "https://example.invalid/v1", http_client=http_client
        ) as client:
            result = await run_fusion(
                client,
                "Why is the sky blue?",
                panel_models=["broken/model", "openai/gpt-4o-mini"],
                judge_model="openai/gpt-4o",
                synth_model="anthropic/claude-3.5-sonnet",
            )

        ok = [p for p in result.panel if p.ok]
        failed = [p for p in result.panel if not p.ok]
        assert len(ok) == 1
        assert len(failed) == 1
        assert failed[0].model == "broken/model"
        assert "Rayleigh" in result.final_answer

    asyncio.run(scenario())


def test_extract_json_object_handles_fenced_output() -> None:
    fenced = "Here is the analysis:\n```json\n{\"consensus\": [\"x\"]}\n```\nthanks!"
    parsed = _extract_json_object(fenced)
    assert parsed == {"consensus": ["x"]}

    assert _extract_json_object("not json at all") is None
