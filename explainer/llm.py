"""The Anthropic client, with cost accounting and a testable seam.

Everything the generation stages need from the API, in one place: adaptive
thinking, structured outputs, prompt caching for the static half of a prompt,
typed failures, and a running total in dollars.

**Request building and response handling are pure functions.** ``build_request``
returns the keyword arguments and ``reply_from`` turns a response into a
``Reply``; ``complete`` is the thin part between them that actually calls the
API. That split is not decoration -- it is what lets the facts below be
asserted by a test suite that never spends a cent, and several of them are
facts a future edit would plausibly get wrong:

* Thinking is ``{"type": "adaptive"}``. ``budget_tokens`` is **removed** on
  the current models and sending it returns a 400 -- it is not merely
  deprecated, and it is the single most likely thing to be reintroduced from
  memory. The converse is also true and also a 400: an older model takes the
  budget and rejects adaptive, which is why ``CAPABILITIES`` exists rather
  than one shape for everything.
* ``effort`` goes inside ``output_config``, not at the top level, and only for
  models that accept it at all.
* Structured output is ``output_config["format"]``. The top-level
  ``output_format`` parameter on ``create`` is deprecated.
* The large static half of a prompt goes **first** in ``system`` and carries
  the cache breakpoint; anything that varies goes after it. Caching is a
  prefix match, so one byte early in the prefix invalidates everything after.

Costs are computed from the response's own usage rather than estimated, and
cache reads and writes are priced separately -- a cache read is a tenth of the
input rate and a write is a quarter more, so a run that reports only
``input_tokens`` misstates the bill in both directions.

The client is built lazily. Importing this module must work on a machine with
no API key, because most of the test suite does exactly that.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping

MODEL = "claude-opus-5"
MAX_TOKENS = 16_000
EFFORT = "high"

# Long enough for adaptive thinking on a hard scene.
TIMEOUT_SECONDS = 600.0

#: Above this, the request is streamed. A non-streaming call with a large
#: max_tokens hits the HTTP timeout before the model finishes; streaming and
#: taking the final message avoids it and is what the SDK asks for.
STREAM_ABOVE = 16_000


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float


PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.00, output=25.00),
    "claude-sonnet-5": Price(input=2.00, output=10.00),
    "claude-haiku-4-5": Price(input=1.00, output=5.00),
}


@dataclass(frozen=True)
class Capabilities:
    """What a model will accept, which is not the same for all of them.

    Sending the wrong shape is a 400, not a graceful degradation, so a model
    that cannot take adaptive thinking or an effort level must not be sent
    either. This matters the moment a second model is used for anything --
    comparing generators, or running a cheap one under a repair loop.
    """

    adaptive_thinking: bool = True
    effort: bool = True
    #: For models that still take a fixed thinking budget. None means no
    #: thinking configuration at all.
    thinking_budget: int | None = None


CAPABILITIES: dict[str, Capabilities] = {
    "claude-opus-5": Capabilities(),
    "claude-sonnet-5": Capabilities(),
    # Adaptive thinking and `effort` both arrived after this one: it takes the
    # older fixed budget, and an effort level is rejected outright.
    "claude-haiku-4-5": Capabilities(
        adaptive_thinking=False, effort=False, thinking_budget=4_000
    ),
}


def capabilities(model: str) -> Capabilities:
    """What *model* accepts. An unknown model is assumed to be current."""
    return CAPABILITIES.get(model, Capabilities())

# A cache write costs a quarter more than an ordinary input token; a read costs
# a tenth. Both are why a bill cannot be reconstructed from input_tokens alone.
CACHE_WRITE_RATE = 1.25
CACHE_READ_RATE = 0.10


class LLMError(Exception):
    """Anything that stopped a request from producing usable text."""


class Refused(LLMError):
    """The model declined. Carries the category when the API gave one."""

    def __init__(self, message: str, category: str | None = None):
        super().__init__(message)
        self.category = category


class Truncated(LLMError):
    """Output hit max_tokens. The text is real but incomplete."""


class Unavailable(LLMError):
    """A failure worth retrying later: rate limit, server error, network."""


class BadRequest(LLMError):
    """The request was wrong. Retrying it unchanged will fail again."""


@dataclass(frozen=True)
class Usage:
    """Tokens from one call, or the sum over many."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    calls: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            calls=self.calls + other.calls,
        )

    def dollars(self, model: str = MODEL) -> float:
        price = PRICES.get(model)
        if price is None:
            return 0.0
        per_token = price.input / 1_000_000
        return (
            self.input_tokens * per_token
            + self.cache_write_tokens * per_token * CACHE_WRITE_RATE
            + self.cache_read_tokens * per_token * CACHE_READ_RATE
            + self.output_tokens * price.output / 1_000_000
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of prompt tokens served from cache.

        Zero across repeated calls with the same preamble means something is
        invalidating the prefix, and that is worth noticing rather than paying.
        """
        prompt = self.input_tokens + self.cache_write_tokens + self.cache_read_tokens
        return self.cache_read_tokens / prompt if prompt else 0.0


@dataclass(frozen=True)
class Reply:
    text: str
    usage: Usage
    seconds: float
    model: str
    stop_reason: str
    request_id: str | None = None

    def dollars(self) -> float:
        return self.usage.dollars(self.model)


def build_request(
    prompt: str,
    *,
    model: str = MODEL,
    preamble: str = "",
    system: str = "",
    schema: Mapping[str, Any] | None = None,
    effort: str = EFFORT,
    max_tokens: int = MAX_TOKENS,
) -> dict[str, Any]:
    """The keyword arguments for one ``messages.create`` call.

    *preamble* is the large half that does not change between calls -- the
    Manim guidance and the IR specification are identical on every codegen
    call and dominate the tokens. It goes first and carries the cache
    breakpoint. *system* is whatever varies, and goes after it so it cannot
    invalidate the cached prefix.
    """
    blocks: list[dict[str, Any]] = []
    if preamble:
        blocks.append(
            {
                "type": "text",
                "text": preamble,
                "cache_control": {"type": "ephemeral"},
            }
        )
    if system:
        blocks.append({"type": "text", "text": system})

    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    able = capabilities(model)
    if able.adaptive_thinking:
        # Adaptive, and never budget_tokens: that parameter is removed on the
        # current models and returns a 400.
        request["thinking"] = {"type": "adaptive"}
    elif able.thinking_budget:
        # And the converse for an older one, where adaptive is what fails.
        request["thinking"] = {
            "type": "enabled",
            "budget_tokens": min(able.thinking_budget, max_tokens - 1),
        }

    output_config: dict[str, Any] = {}
    if able.effort:
        output_config["effort"] = effort
    if schema is not None:
        # Not the deprecated top-level output_format.
        output_config["format"] = {"type": "json_schema", "schema": dict(schema)}
    if output_config:
        request["output_config"] = output_config

    if blocks:
        request["system"] = blocks
    return request


def usage_from(raw: Any) -> Usage:
    """Read a response's usage, tolerating the fields it may omit."""

    def field(name: str) -> int:
        return int(getattr(raw, name, 0) or 0)

    return Usage(
        input_tokens=field("input_tokens"),
        output_tokens=field("output_tokens"),
        cache_write_tokens=field("cache_creation_input_tokens"),
        cache_read_tokens=field("cache_read_input_tokens"),
        calls=1,
    )


def reply_from(message: Any, seconds: float = 0.0) -> Reply:
    """Turn a response into a ``Reply``, or raise.

    stop_reason is checked before the content is read. A refusal arrives as a
    perfectly ordinary HTTP 200, so code that goes straight for the text sees
    an empty answer and no reason for it.
    """
    stop_reason = getattr(message, "stop_reason", "") or ""
    model = getattr(message, "model", MODEL) or MODEL
    usage = usage_from(getattr(message, "usage", None))

    if stop_reason == "refusal":
        details = getattr(message, "stop_details", None)
        category = getattr(details, "category", None) if details else None
        explanation = getattr(details, "explanation", None) if details else None
        raise Refused(explanation or f"the model declined ({category or 'no category'})", category)

    text = "".join(
        block.text
        for block in getattr(message, "content", [])
        if getattr(block, "type", "") == "text"
    )

    if stop_reason == "max_tokens":
        raise Truncated(
            f"output hit max_tokens after {usage.output_tokens} tokens; "
            "raise max_tokens or ask for less"
        )

    return Reply(
        text=text,
        usage=usage,
        seconds=seconds,
        model=model,
        stop_reason=stop_reason,
        request_id=getattr(message, "_request_id", None),
    )


def have_key() -> bool:
    """Whether any credential the SDK understands is present in the env."""
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class Client:
    """A thin wrapper that keeps a running total of what it has spent."""

    def __init__(
        self,
        *,
        model: str = MODEL,
        effort: str = EFFORT,
        max_tokens: int = MAX_TOKENS,
        timeout: float = TIMEOUT_SECONDS,
        max_retries: int = 3,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.total = Usage()
        self._client: Any = None

    @property
    def api(self) -> Any:
        """The SDK client, built on first use.

        Lazily, so importing this module -- which most of the test suite does
        -- does not require a credential.
        """
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                timeout=self.timeout, max_retries=self.max_retries
            )
        return self._client

    def dollars(self) -> float:
        return self.total.dollars(self.model)

    def complete(
        self,
        prompt: str,
        *,
        preamble: str = "",
        system: str = "",
        schema: Mapping[str, Any] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> Reply:
        """One call. Raises an LLMError subclass rather than returning junk.

        Thinking is billed as output, so a task that reasons hard can exhaust
        max_tokens before it has written anything -- which is what a layout of
        five scenes did at the default. Raising the ceiling means streaming.
        """
        import anthropic

        tokens = max_tokens or self.max_tokens
        request = build_request(
            prompt,
            model=self.model,
            preamble=preamble,
            system=system,
            schema=schema,
            effort=effort or self.effort,
            max_tokens=tokens,
        )
        started = time.perf_counter()
        try:
            if tokens > STREAM_ABOVE:
                with self.api.messages.stream(**request) as stream:
                    message = stream.get_final_message()
            else:
                message = self.api.messages.create(**request)
        except anthropic.BadRequestError as exc:
            raise BadRequest(str(exc)) from exc
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise BadRequest(f"credential rejected: {exc}") from exc
        except anthropic.NotFoundError as exc:
            raise BadRequest(f"no such model {self.model!r}: {exc}") from exc
        except anthropic.RateLimitError as exc:
            # The SDK already retried; arriving here means it kept failing.
            raise Unavailable(f"rate limited after {self.max_retries} retries: {exc}") from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise Unavailable(f"server error {exc.status_code}: {exc}") from exc
            raise BadRequest(f"api error {exc.status_code}: {exc}") from exc
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise Unavailable(f"could not reach the API: {exc}") from exc

        reply = reply_from(message, time.perf_counter() - started)
        self.total = self.total + reply.usage
        return reply

    def count_tokens(self, prompt: str, *, preamble: str = "", system: str = "") -> int:
        """What a prompt would cost to send, before sending it."""
        request = build_request(
            prompt, model=self.model, preamble=preamble, system=system
        )
        response = self.api.messages.count_tokens(
            model=request["model"],
            messages=request["messages"],
            **({"system": request["system"]} if "system" in request else {}),
        )
        return int(response.input_tokens)

    def spent(self) -> Usage:
        return replace(self.total)
