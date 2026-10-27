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
from myfusion.pipeline import (
    JudgeAnalysis,
    PanelResponse,
    _extract_json_object,
    _format_analysis_for_synth,
    run_fusion,
    run_judge,
)

JUDGE_PAYLOAD = {
    "consensus": ["The sky appears blue due to scattering of light."],
    "contradictions": ["Model A says Rayleigh scattering; Model B blames the ocean."],
    "partial_coverage": ["Only one model mentioned wavelength dependence."],
    "unique_insights": ["Sunsets are red because of longer path length."],
    "blind_spots": ["None discussed atmospheric composition changes."],
}


def _responses_body(content: str) -> dict:
    """A Responses API body with a reasoning item ahead of the message."""

    return {
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            },
        ]
    }


def _handler(request: httpx.Request) -> httpx.Response:
    """Route mock responses by the model field in the request body."""

    assert request.url.path.endswith("/responses")
    body = json.loads(request.content)
    model = body["model"]
    messages = body["input"]
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

    if "impartial judge" in system:
        content = json.dumps(JUDGE_PAYLOAD)
    elif "final author" in system:
        content = "The sky is blue because of Rayleigh scattering of sunlight."
    else:
        # A panel member: echo which model answered.
        content = f"Answer from {model}: light scatters in the atmosphere."

    return httpx.Response(200, json=_responses_body(content))


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
                panel_models=["openai/gpt-6-luna", "anthropic/claude-3.5-haiku"],
                judge_model="openai/gpt-6-luna",
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
                panel_models=["broken/model", "openai/gpt-6-luna"],
                judge_model="openai/gpt-6-luna",
                synth_model="anthropic/claude-3.5-sonnet",
            )

        ok = [p for p in result.panel if p.ok]
        failed = [p for p in result.panel if not p.ok]
        assert len(ok) == 1
        assert len(failed) == 1
        assert failed[0].model == "broken/model"
        assert "Rayleigh" in result.final_answer

    asyncio.run(scenario())


def test_complete_builds_responses_payload() -> None:
    captured: list[dict] = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=_responses_body('{"consensus": []}'))

    async def scenario() -> list[str]:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(capture))
        async with OpenRouterClient(
            "k", "https://example.invalid/v1", http_client=http_client
        ) as client:
            return [
                await client.complete(
                    "openai/gpt-6-luna",
                    [{"role": "user", "content": "hi"}],
                    temperature=0.2,
                    max_tokens=50,
                    response_format={"type": "json_object"},
                ),
                await client.complete(
                    "anthropic/claude-haiku-5.5",
                    [{"role": "user", "content": "hi"}],
                ),
            ]

    assert asyncio.run(scenario()) == ['{"consensus": []}'] * 2
    assert captured == [
        {
            "model": "openai/gpt-6-luna",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
            "reasoning": {"effort": "none"},
            "max_output_tokens": 50,
            "text": {"format": {"type": "json_object"}},
        },
        {
            "model": "anthropic/claude-haiku-5.5",
            "input": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
        },
    ]


def test_extract_json_object_handles_fenced_output() -> None:
    fenced = "Here is the analysis:\n```json\n{\"consensus\": [\"x\"]}\n```\nthanks!"
    parsed = _extract_json_object(fenced)
    assert parsed == {"consensus": ["x"]}

    assert _extract_json_object("not json at all") is None


def test_run_judge_keeps_raw_text_when_not_json() -> None:
    def prose(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_responses_body("I refuse to emit JSON."))

    async def scenario() -> JudgeAnalysis:
        http_client = httpx.AsyncClient(transport=httpx.MockTransport(prose))
        async with OpenRouterClient(
            "k", "https://example.invalid/v1", http_client=http_client
        ) as client:
            return await run_judge(
                client, "q", [PanelResponse(model="a/x", content="answer")], "j/z"
            )

    analysis = asyncio.run(scenario())
    assert analysis == JudgeAnalysis(raw="I refuse to emit JSON.")
    assert analysis.is_empty


def test_format_analysis_for_synth() -> None:
    structured = _format_analysis_for_synth(
        "q", JudgeAnalysis(consensus=["agreed"], raw="{}")
    )
    assert "JUDGE ANALYSIS:\nConsensus:\n  - agreed\nContradictions: (none)" in structured
    assert "Blind spots: (none)" in structured

    raw_only = _format_analysis_for_synth("q", JudgeAnalysis(raw="judge rambled"))
    assert "JUDGE ANALYSIS (raw):\njudge rambled" in raw_only
