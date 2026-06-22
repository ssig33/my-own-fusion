"""The Fusion pipeline: panel -> judge -> synthesis.

This re-implements, against the raw chat-completions API, the three-stage idea
described by OpenRouter's Fusion plugin:

1. **Panel** - a set of models answers the same prompt in parallel.
2. **Judge** - a single model *compares* (does not merge) the panel answers and
   returns a structured JSON analysis: consensus, contradictions, partial
   coverage, unique insights, and blind spots.
3. **Synthesis** - a final model reads the analysis and writes the best answer.

Each stage is a small coroutine so they can be tested and reused independently.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

from .client import OpenRouterClient, OpenRouterError

# A callback invoked with human-readable progress updates (defaults to no-op).
ProgressFn = Callable[[str], None]


@dataclass
class PanelResponse:
    """One panel member's answer (or the error it produced)."""

    model: str
    content: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class JudgeAnalysis:
    """Structured comparison of the panel answers produced by the judge."""

    consensus: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    partial_coverage: list[str] = field(default_factory=list)
    unique_insights: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    # Raw judge text, kept for debugging / when JSON parsing falls back.
    raw: str = ""

    @classmethod
    def from_payload(cls, payload: dict, raw: str) -> "JudgeAnalysis":
        """Build an analysis from a parsed JSON object, tolerating odd shapes."""

        def as_list(value: object) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)]

        return cls(
            consensus=as_list(payload.get("consensus")),
            contradictions=as_list(payload.get("contradictions")),
            partial_coverage=as_list(payload.get("partial_coverage")),
            unique_insights=as_list(payload.get("unique_insights")),
            blind_spots=as_list(payload.get("blind_spots")),
            raw=raw,
        )


@dataclass
class FusionResult:
    """The full output of a fusion run, suitable for JSON serialization."""

    prompt: str
    panel: list[PanelResponse]
    analysis: JudgeAnalysis
    final_answer: str
    judge_model: str
    synth_model: str

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "panel": [asdict(p) for p in self.panel],
            "analysis": asdict(self.analysis),
            "final_answer": self.final_answer,
            "judge_model": self.judge_model,
            "synth_model": self.synth_model,
        }


def _extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from model output.

    Models sometimes wrap JSON in prose or markdown fences even when asked not
    to, so we first try a strict parse, then fall back to slicing out the
    outermost ``{...}`` span.
    """

    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


async def run_panel(
    client: OpenRouterClient,
    prompt: str,
    models: list[str],
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    progress: ProgressFn | None = None,
) -> list[PanelResponse]:
    """Query every panel model in parallel.

    A single model failing does not abort the run: its error is captured in the
    corresponding :class:`PanelResponse` and the remaining answers proceed.
    """

    say = progress or (lambda _msg: None)
    messages = [{"role": "user", "content": prompt}]

    async def ask(model: str) -> PanelResponse:
        say(f"  - querying panel model: {model}")
        try:
            content = await client.complete(
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            say(f"  - done: {model}")
            return PanelResponse(model=model, content=content)
        except OpenRouterError as exc:
            say(f"  - failed: {model} ({exc})")
            return PanelResponse(model=model, error=str(exc))

    return await asyncio.gather(*(ask(model) for model in models))


JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge in a multi-model deliberation. You will be given "
    "an original prompt and several independent answers from different AI models. "
    "Your job is to COMPARE the answers, not to merge or rewrite them. "
    "Return ONLY a JSON object with these string-array fields:\n"
    '  "consensus": points all or most models agree on (treat as higher confidence)\n'
    '  "contradictions": points where the models directly disagree\n'
    '  "partial_coverage": important sub-topics only some models addressed\n'
    '  "unique_insights": valuable points raised by a single model\n'
    '  "blind_spots": relevant aspects none of the models addressed\n'
    "Each array item is a concise sentence. Do not add other fields or prose."
)


def _format_panel_for_judge(prompt: str, panel: list[PanelResponse]) -> str:
    """Render the panel answers into the judge's user message."""

    blocks = [f"ORIGINAL PROMPT:\n{prompt}\n", "PANEL ANSWERS:"]
    for index, response in enumerate(panel, start=1):
        if response.ok:
            blocks.append(f"\n[Answer {index} - model: {response.model}]\n{response.content}")
        else:
            blocks.append(
                f"\n[Answer {index} - model: {response.model}] (unavailable: {response.error})"
            )
    blocks.append("\nReturn the JSON analysis now.")
    return "\n".join(blocks)


async def run_judge(
    client: OpenRouterClient,
    prompt: str,
    panel: list[PanelResponse],
    judge_model: str,
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    progress: ProgressFn | None = None,
) -> JudgeAnalysis:
    """Have the judge model compare the panel answers and return structured JSON."""

    say = progress or (lambda _msg: None)
    say(f"  - judging with: {judge_model}")
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": _format_panel_for_judge(prompt, panel)},
    ]
    raw = await client.complete(
        judge_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    payload = _extract_json_object(raw)
    if payload is None:
        # Keep the run alive even if the judge ignored the JSON instruction.
        say("  - judge did not return valid JSON; keeping raw text only")
        return JudgeAnalysis(raw=raw)
    return JudgeAnalysis.from_payload(payload, raw=raw)


SYNTH_SYSTEM_PROMPT = (
    "You are the final author in a multi-model deliberation. You are given the "
    "user's original prompt and a judge's structured analysis comparing several "
    "models' answers. Use the analysis to write the single best possible answer: "
    "lean on the consensus, resolve the contradictions, fill the partial coverage "
    "and blind spots, and fold in the unique insights where they help. Write the "
    "answer directly for the user; do not mention the panel, the judge, or this "
    "process."
)


def _format_analysis_for_synth(prompt: str, analysis: JudgeAnalysis) -> str:
    """Render the judge analysis into the synthesizer's user message."""

    def section(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}: (none)"
        bullets = "\n".join(f"  - {item}" for item in items)
        return f"{title}:\n{bullets}"

    if analysis.raw and not any(
        [
            analysis.consensus,
            analysis.contradictions,
            analysis.partial_coverage,
            analysis.unique_insights,
            analysis.blind_spots,
        ]
    ):
        # Fallback: hand the raw judge text to the synthesizer.
        analysis_block = f"JUDGE ANALYSIS (raw):\n{analysis.raw}"
    else:
        analysis_block = "\n".join(
            [
                "JUDGE ANALYSIS:",
                section("Consensus", analysis.consensus),
                section("Contradictions", analysis.contradictions),
                section("Partial coverage", analysis.partial_coverage),
                section("Unique insights", analysis.unique_insights),
                section("Blind spots", analysis.blind_spots),
            ]
        )

    return f"ORIGINAL PROMPT:\n{prompt}\n\n{analysis_block}\n\nWrite the final answer now."


async def run_synthesis(
    client: OpenRouterClient,
    prompt: str,
    analysis: JudgeAnalysis,
    synth_model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    progress: ProgressFn | None = None,
) -> str:
    """Write the final answer from the judge's analysis."""

    say = progress or (lambda _msg: None)
    say(f"  - synthesizing with: {synth_model}")
    messages = [
        {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
        {"role": "user", "content": _format_analysis_for_synth(prompt, analysis)},
    ]
    return await client.complete(
        synth_model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def run_fusion(
    client: OpenRouterClient,
    prompt: str,
    *,
    panel_models: list[str],
    judge_model: str,
    synth_model: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    progress: ProgressFn | None = None,
) -> FusionResult:
    """Run the full panel -> judge -> synthesis pipeline.

    Raises:
        RuntimeError: if every panel model failed (nothing to judge).
    """

    say = progress or (lambda _msg: None)

    say("[1/3] Panel: querying models in parallel...")
    panel = await run_panel(
        client,
        prompt,
        panel_models,
        temperature=temperature,
        max_tokens=max_tokens,
        progress=progress,
    )
    if not any(response.ok for response in panel):
        errors = "; ".join(f"{p.model}: {p.error}" for p in panel)
        raise RuntimeError(f"all panel models failed: {errors}")

    say("[2/3] Judge: comparing answers...")
    analysis = await run_judge(
        client,
        prompt,
        panel,
        judge_model,
        max_tokens=max_tokens,
        progress=progress,
    )

    say("[3/3] Synthesis: writing the final answer...")
    final_answer = await run_synthesis(
        client,
        prompt,
        analysis,
        synth_model,
        temperature=temperature,
        max_tokens=max_tokens,
        progress=progress,
    )

    return FusionResult(
        prompt=prompt,
        panel=panel,
        analysis=analysis,
        final_answer=final_answer,
        judge_model=judge_model,
        synth_model=synth_model,
    )


__all__ = [
    "FusionResult",
    "JudgeAnalysis",
    "PanelResponse",
    "ProgressFn",
    "run_fusion",
    "run_panel",
    "run_judge",
    "run_synthesis",
]
