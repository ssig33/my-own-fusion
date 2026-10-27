"""my-own-fusion: a standalone PoC of the OpenRouter Fusion pipeline.

The package exposes the building blocks of a multi-model deliberation pipeline:

* a :class:`~myfusion.client.OpenRouterClient` that talks to any OpenAI-compatible
  Responses API endpoint,
* a :func:`~myfusion.pipeline.run_fusion` coroutine that orchestrates the
  panel -> judge -> synthesis stages, and
* a thin command-line interface (``myfusion``).
"""

from importlib.metadata import version

from .config import Config
from .pipeline import FusionResult, JudgeAnalysis, PanelResponse, run_fusion

__version__ = version("my-own-fusion")

__all__ = [
    "Config",
    "FusionResult",
    "JudgeAnalysis",
    "PanelResponse",
    "run_fusion",
    "__version__",
]
