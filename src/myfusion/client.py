"""A minimal async client for OpenAI-compatible chat-completions endpoints.

The client targets OpenRouter (https://openrouter.ai/api/v1) by default, but any
endpoint that speaks the OpenAI ``/chat/completions`` shape works, including a
local mock used by the test suite.
"""

from __future__ import annotations

from typing import Any

import httpx

# Message is a list of {"role": ..., "content": ...} dicts.
Message = dict[str, str]


class OpenRouterError(RuntimeError):
    """Raised when the upstream API returns a non-success response."""


class OpenRouterClient:
    """Thin wrapper around the chat-completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        *,
        timeout: float = 120.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        # Identifying headers are recommended by OpenRouter for attribution.
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ssig33/my-own-fusion",
            "X-Title": "my-own-fusion",
        }
        # Allow dependency injection of the transport so tests can run offline.
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def __aenter__(self) -> "OpenRouterClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying HTTP client if we created it."""

        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        model: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """Run a single chat completion and return the assistant text.

        Args:
            model: Model slug, e.g. ``"openai/gpt-4o-mini"``.
            messages: OpenAI-style message list.
            temperature: Sampling temperature.
            max_tokens: Optional cap on generated tokens.
            response_format: Optional structured-output hint, e.g.
                ``{"type": "json_object"}``.

        Raises:
            OpenRouterError: on HTTP errors or malformed responses.
        """

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self._base_url}/chat/completions"
        try:
            response = await self._client.post(url, headers=self._headers, json=payload)
        except httpx.HTTPError as exc:  # network-level failures
            raise OpenRouterError(f"request to {model} failed: {exc}") from exc

        if response.status_code >= 400:
            raise OpenRouterError(
                f"{model} returned HTTP {response.status_code}: {response.text[:500]}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise OpenRouterError(
                f"could not parse response from {model}: {exc}"
            ) from exc
