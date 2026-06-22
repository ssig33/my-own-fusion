# my-own-fusion

A small, standalone **Proof-of-Concept** of the
[OpenRouter Fusion](https://openrouter.ai/docs/guides/features/plugins/fusion)
idea: instead of trusting a single model, ask **several** models the same
question, have a **judge** model compare their answers, and let a final model
**synthesize** the best possible response from that comparison.

This repo re-implements that pipeline from scratch against the plain
OpenAI-compatible `/chat/completions` API, so you can see exactly how each stage
works and run it yourself with nothing but an API key.

> This is an independent educational PoC. It is **not** the official OpenRouter
> Fusion plugin and is not affiliated with OpenRouter.

## How it works

```
                prompt
                  │
        ┌─────────┼─────────┐         (1) PANEL
        ▼         ▼         ▼          N models answer in parallel
     model A   model B   model C
        │         │         │
        └─────────┼─────────┘
                  ▼                    (2) JUDGE
              judge model              compares (does NOT merge) the answers,
                  │                    emits structured JSON:
                  ▼                      consensus / contradictions /
            JSON analysis                partial_coverage / unique_insights /
                  │                      blind_spots
                  ▼                    (3) SYNTHESIS
            synthesizer model          writes the final answer from the analysis
                  │
                  ▼
            final answer
```

1. **Panel** — each model in `--panel` answers the prompt independently and in
   parallel. If one model errors out, the run continues with the rest.
2. **Judge** — the `--judge` model receives all panel answers and returns a
   structured JSON comparison. Crucially, it *compares* rather than *merges*:
   it highlights where the models agree (higher confidence), where they
   contradict each other, what only some of them covered, what unique points a
   single model raised, and what they all missed.
3. **Synthesis** — the `--synth` model reads that analysis and writes the final
   answer for the user.

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (used for dependency management and running)
- An [OpenRouter API key](https://openrouter.ai/keys) (or any OpenAI-compatible
  endpoint + key)

## Install

```bash
git clone https://github.com/ssig33/my-own-fusion.git
cd my-own-fusion
uv sync
```

## Configure

Copy the example env file and add your key. `.env` is gitignored and will never
be committed:

```bash
cp .env.example .env
# then edit .env and set OPENROUTER_API_KEY=...
```

Alternatively, export it in your shell:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

No secrets are ever hardcoded — the key is read only from the environment.

## Usage

```bash
# Simplest form
uv run myfusion "Compare ridge, lasso, and elastic-net regression. Where does each shine?"

# Read the prompt from stdin
echo "What are the strongest arguments for and against carbon taxes?" | uv run myfusion -

# See every stage (panel answers + judge analysis) before the final answer
uv run myfusion --show-stages "Why do distributed systems prefer idempotent operations?"

# Get the whole run as machine-readable JSON
uv run myfusion --json "Explain CAP theorem" > result.json

# Pick your own models
uv run myfusion \
  --panel "openai/gpt-4o-mini,anthropic/claude-3.5-haiku,google/gemini-2.0-flash-001" \
  --judge "openai/gpt-4o" \
  --synth "anthropic/claude-3.5-sonnet" \
  "Design a rate limiter for a public API."
```

Progress messages are printed to **stderr**; the final answer (or JSON) goes to
**stdout**, so piping and redirection behave as you would expect.

### Options

| Flag | Description |
| --- | --- |
| `prompt` | The prompt. Use `-` or omit it to read from stdin. |
| `--panel MODELS` | Comma-separated panel model slugs. |
| `--judge MODEL` | Judge model slug. |
| `--synth MODEL` | Synthesizer model slug. |
| `--temperature FLOAT` | Sampling temperature for panel + synthesis (default `0.7`). |
| `--max-tokens INT` | Optional max tokens per completion. |
| `--show-stages` | Print panel answers and judge analysis before the final answer. |
| `--json` | Print the full result (panel, analysis, answer) as JSON. |
| `--quiet` | Suppress progress messages on stderr. |
| `--version` | Print the version and exit. |

### Configuration precedence

Lowest to highest priority:

1. built-in defaults (see `src/myfusion/config.py`),
2. environment variables — `OPENROUTER_API_KEY` (required),
   `OPENROUTER_BASE_URL`, `FUSION_PANEL_MODELS`, `FUSION_JUDGE_MODEL`,
   `FUSION_SYNTH_MODEL`,
3. command-line flags.

The default models are intentionally small and cheap. Model availability changes
over time — check <https://openrouter.ai/models> and override the defaults with
the flags above as needed.

## Using it as a library

```python
import asyncio
from myfusion import Config, run_fusion
from myfusion.client import OpenRouterClient

async def main():
    config = Config.from_env()
    async with OpenRouterClient(config.api_key, config.base_url) as client:
        result = await run_fusion(
            client,
            "Why is the sky blue?",
            panel_models=config.panel_models,
            judge_model=config.judge_model,
            synth_model=config.synth_model,
        )
    print(result.final_answer)

asyncio.run(main())
```

## Run the tests

The test suite is fully offline — it injects a mock HTTP transport, so it runs
without an API key and without spending any tokens:

```bash
uv run pytest
```

## Project layout

```
src/myfusion/
  client.py     # async OpenAI-compatible chat-completions client
  config.py     # env/.env configuration loading and defaults
  pipeline.py   # the panel -> judge -> synthesis pipeline
  cli.py        # command-line interface
tests/
  test_pipeline.py
```

## Why bother?

A single model can be confidently wrong. Fusion trades extra completions for
robustness: disagreement between models becomes a signal, the judge surfaces it
explicitly, and the synthesizer writes an answer that accounts for it. It is
overkill for quick questions and worth it when being wrong is expensive.

## License

[MIT](LICENSE)
