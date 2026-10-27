"""Command-line interface for my-own-fusion.

Usage examples::

    myfusion "Compare ridge, lasso, and elastic-net regression."
    echo "Are carbon taxes effective?" | myfusion -
    myfusion "..." --panel openai/gpt-6-luna,x-ai/grok-2 --judge openai/gpt-6-luna
    myfusion "..." --json > result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from . import __version__
from .client import OpenRouterClient, OpenRouterError
from .config import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PANEL_MODELS,
    DEFAULT_SYNTH_MODEL,
    Config,
    load_dotenv,
    split_models,
)
from .pipeline import FusionResult, run_fusion


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myfusion",
        description=(
            "Fuse multiple AI models into one answer using a panel -> judge -> "
            "synthesis pipeline (an OpenRouter Fusion proof-of-concept)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send. Use '-' or omit to read from stdin.",
    )
    parser.add_argument(
        "--panel",
        metavar="MODELS",
        help=(
            "Comma-separated panel model slugs "
            f"(default: {','.join(DEFAULT_PANEL_MODELS)})."
        ),
    )
    parser.add_argument(
        "--judge",
        metavar="MODEL",
        help=f"Judge model slug (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--synth",
        metavar="MODEL",
        help=f"Synthesizer model slug (default: {DEFAULT_SYNTH_MODEL}).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the panel and synthesis stages (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional max tokens per completion.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result (panel, analysis, answer) as JSON to stdout.",
    )
    parser.add_argument(
        "--show-stages",
        action="store_true",
        help="Print the panel answers and judge analysis before the final answer.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages on stderr.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"my-own-fusion {__version__}",
    )
    return parser


def _read_prompt(arg: str | None) -> str:
    """Resolve the prompt from the argument or stdin."""

    if arg is None or arg == "-":
        if sys.stdin.isatty():
            raise ValueError("no prompt given (pass it as an argument or via stdin)")
        text = sys.stdin.read().strip()
    else:
        text = arg.strip()
    if not text:
        raise ValueError("the prompt is empty")
    return text


def _print_stages(result: FusionResult) -> None:
    """Render the intermediate panel + judge output in a human-readable form."""

    out = sys.stdout
    out.write("\n===== PANEL ANSWERS =====\n")
    for response in result.panel:
        out.write(f"\n--- {response.model} ---\n")
        out.write(f"{response.content}\n" if response.ok else f"(failed: {response.error})\n")

    analysis = result.analysis
    out.write(f"\n===== JUDGE ANALYSIS ({result.judge_model}) =====\n")
    for title, items in analysis.sections():
        out.write(f"\n{title}:\n")
        if items:
            for item in items:
                out.write(f"  - {item}\n")
        else:
            out.write("  (none)\n")
    if analysis.is_empty and analysis.raw:
        out.write(f"\n(raw judge output)\n{analysis.raw}\n")


async def _run(args: argparse.Namespace, config: Config, prompt: str) -> FusionResult:
    progress = None if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    async with OpenRouterClient(
        config.api_key, config.base_url, timeout=config.timeout
    ) as client:
        return await run_fusion(
            client,
            prompt,
            panel_models=config.panel_models,
            judge_model=config.judge_model,
            synth_model=config.synth_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            progress=progress,
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Seed environment from a local .env file, then resolve configuration.
    load_dotenv()
    try:
        prompt = _read_prompt(args.prompt)
        config = Config.from_env()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # CLI flags override environment/defaults.
    if args.panel:
        config.panel_models = split_models(args.panel) or []
    if args.judge:
        config.judge_model = args.judge
    if args.synth:
        config.synth_model = args.synth
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens

    if not config.panel_models:
        print("error: the panel model list is empty", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(_run(args, config, prompt))
    except (OpenRouterError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return 0

    if args.show_stages:
        _print_stages(result)
        print("\n===== FINAL ANSWER =====\n")

    print(result.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
