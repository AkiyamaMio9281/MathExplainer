"""The request shape and the cost arithmetic, without spending anything.

Almost all of this runs with no API key, because the parts most likely to be
wrong are not the network call. They are the request shape -- where `effort`
lives, what `thinking` may contain, which half of the prompt carries the cache
breakpoint -- and the arithmetic that turns four token counts into a number of
dollars.

Several assertions here exist to pin facts a future edit would plausibly get
wrong from memory. `budget_tokens` is the clearest: it reads like a harmless
addition, it is what older code and older documentation both use, and on this
model it returns a 400. A test is cheaper than that round trip.

One test does call the API, marked `api`, and is skipped when no key is set.
"""

from __future__ import annotations

import uuid

import pytest

from explainer import llm


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


def test_thinking_is_adaptive_and_carries_no_budget():
    request = llm.build_request("hello")

    assert request["thinking"] == {"type": "adaptive"}
    # Removed on this model, not deprecated: sending it is a 400.
    assert "budget_tokens" not in str(request)


def test_effort_lives_inside_output_config():
    request = llm.build_request("hello", effort="max")

    assert request["output_config"]["effort"] == "max"
    assert "effort" not in {k for k in request if k != "output_config"}


def test_a_schema_becomes_output_config_format_not_output_format():
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    request = llm.build_request("hello", schema=schema)

    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"] == schema
    # The top-level parameter of that name is deprecated.
    assert "output_format" not in request


def test_no_schema_means_no_format_key():
    assert "format" not in llm.build_request("hello")["output_config"]


def test_the_static_preamble_comes_first_and_carries_the_breakpoint():
    request = llm.build_request("hello", preamble="the manim guide", system="be terse")

    preamble, system = request["system"]
    assert preamble["text"] == "the manim guide"
    assert preamble["cache_control"] == {"type": "ephemeral"}
    # Caching is a prefix match, so the varying half must come after the
    # breakpoint or it invalidates everything cached before it.
    assert system["text"] == "be terse"
    assert "cache_control" not in system


def test_a_prompt_with_no_system_content_sends_no_system_field():
    assert "system" not in llm.build_request("hello")


def test_the_prompt_is_the_user_message():
    request = llm.build_request("explain the chain rule")

    assert request["messages"] == [
        {"role": "user", "content": "explain the chain rule"}
    ]


def test_the_default_model_is_opus_5():
    assert llm.build_request("hello")["model"] == "claude-opus-5"
    assert llm.MODEL in llm.PRICES


# ---------------------------------------------------------------------------
# Reading a response
# ---------------------------------------------------------------------------


class Block:
    def __init__(self, type, text=""):
        self.type = type
        self.text = text


class Details:
    def __init__(self, category=None, explanation=None):
        self.category = category
        self.explanation = explanation


class Response:
    """Enough of a response to exercise reply_from."""

    def __init__(
        self,
        content=(),
        stop_reason="end_turn",
        usage=None,
        model="claude-opus-5",
        stop_details=None,
    ):
        self.content = list(content)
        self.stop_reason = stop_reason
        self.usage = usage
        self.model = model
        self.stop_details = stop_details
        self._request_id = "req_test"


class RawUsage:
    def __init__(self, input_tokens=0, output_tokens=0, cache_creation=0, cache_read=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_creation_input_tokens = cache_creation
        self.cache_read_input_tokens = cache_read


def test_text_blocks_are_joined_and_thinking_is_left_out():
    reply = llm.reply_from(
        Response(
            content=[
                Block("thinking", "reasoning that is not the answer"),
                Block("text", "class Demo"),
                Block("text", "(Scene): ..."),
            ],
            usage=RawUsage(input_tokens=10, output_tokens=5),
        )
    )

    assert reply.text == "class Demo(Scene): ..."
    assert reply.usage.output_tokens == 5


def test_a_refusal_raises_before_the_content_is_read():
    # A refusal arrives as an ordinary 200, so code that reaches straight for
    # the text sees an empty answer and no reason for it.
    with pytest.raises(llm.Refused) as caught:
        llm.reply_from(
            Response(
                content=[],
                stop_reason="refusal",
                stop_details=Details(category="cyber", explanation="declined"),
                usage=RawUsage(),
            )
        )

    assert caught.value.category == "cyber"


def test_hitting_max_tokens_raises_rather_than_returning_half_a_file():
    with pytest.raises(llm.Truncated):
        llm.reply_from(
            Response(
                content=[Block("text", "class Demo(Scene):\n    def constr")],
                stop_reason="max_tokens",
                usage=RawUsage(output_tokens=16000),
            )
        )


def test_stop_details_is_only_read_when_there_was_a_refusal():
    # It is null for every other stop reason, so reading it unguarded is a
    # crash on the ordinary path.
    reply = llm.reply_from(
        Response(content=[Block("text", "fine")], usage=RawUsage(), stop_details=None)
    )
    assert reply.stop_reason == "end_turn"


def test_a_response_with_no_usage_still_produces_a_reply():
    reply = llm.reply_from(Response(content=[Block("text", "x")], usage=None))

    assert reply.text == "x"
    assert reply.usage.calls == 1
    assert reply.usage.input_tokens == 0


def test_the_request_id_is_kept_for_reporting_failures():
    reply = llm.reply_from(Response(content=[Block("text", "x")], usage=RawUsage()))
    assert reply.request_id == "req_test"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_prices_each_kind_of_token_at_its_own_rate():
    usage = llm.Usage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
    )
    # 5 + 25 + 5*1.25 + 5*0.10
    assert usage.dollars("claude-opus-5") == pytest.approx(5 + 25 + 6.25 + 0.5)


def test_a_cache_read_is_a_tenth_of_an_input_token():
    read = llm.Usage(cache_read_tokens=1_000_000).dollars()
    plain = llm.Usage(input_tokens=1_000_000).dollars()

    assert read == pytest.approx(plain * 0.10)


def test_a_bill_cannot_be_reconstructed_from_input_tokens_alone():
    # The point of tracking four counters: two runs with identical
    # input_tokens can cost very different amounts.
    cached = llm.Usage(input_tokens=1000, cache_read_tokens=100_000)
    uncached = llm.Usage(input_tokens=1000, cache_write_tokens=100_000)

    assert uncached.dollars() > cached.dollars() * 5


def test_usage_adds_up_across_calls():
    total = llm.Usage(input_tokens=10, calls=1) + llm.Usage(output_tokens=5, calls=1)

    assert total == llm.Usage(input_tokens=10, output_tokens=5, calls=2)


def test_the_cache_hit_rate_is_the_share_of_prompt_tokens_served_from_cache():
    assert llm.Usage(input_tokens=250, cache_read_tokens=750).cache_hit_rate == 0.75
    # Zero across repeated calls means something is invalidating the prefix.
    assert llm.Usage(input_tokens=1000).cache_hit_rate == 0.0
    assert llm.Usage().cache_hit_rate == 0.0


def test_an_unpriced_model_costs_nothing_rather_than_crashing():
    assert llm.Usage(input_tokens=1000).dollars("some-future-model") == 0.0


def test_a_client_keeps_a_running_total():
    client = llm.Client()
    assert client.dollars() == 0.0

    client.total = client.total + llm.Usage(input_tokens=1_000_000, calls=1)
    assert client.dollars() == pytest.approx(5.0)
    assert client.spent().calls == 1


def test_importing_and_constructing_needs_no_credential(monkeypatch):
    # Most of the suite runs on a machine with no key, so the SDK client is
    # built on first use rather than in __init__.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    client = llm.Client()
    assert client._client is None
    assert not llm.have_key()


# ---------------------------------------------------------------------------
# One real call
# ---------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.slow
def test_a_real_call_returns_text_and_reports_what_it_cost():
    client = llm.Client(max_tokens=1024, effort="low")
    reply = client.complete("Reply with exactly the word: ready")

    assert "ready" in reply.text.lower()
    assert reply.stop_reason == "end_turn"
    assert reply.usage.output_tokens > 0
    assert reply.dollars() > 0
    assert client.dollars() == pytest.approx(reply.dollars())


@pytest.mark.api
@pytest.mark.slow
def test_a_real_structured_call_comes_back_as_valid_json():
    import json

    client = llm.Client(max_tokens=1024, effort="low")
    reply = client.complete(
        "Give the title and a one-sentence summary of the Pythagorean theorem.",
        schema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["title", "summary"],
            "additionalProperties": False,
        },
    )

    parsed = json.loads(reply.text)
    assert set(parsed) == {"title", "summary"}


@pytest.mark.api
@pytest.mark.slow
def test_a_repeated_preamble_is_served_from_cache():
    # The claim this module rests on: the static half of a codegen prompt is
    # identical on every call and dominates the tokens, so it should be paid
    # for once. A silent invalidator -- a timestamp in the preamble, an
    # unsorted dump, a varying tool list -- shows up here as a second call
    # that reads nothing from cache and costs the same as the first.
    #
    # The minimum cacheable prefix is model-dependent (512-4096 tokens), so
    # the preamble has to be genuinely large or nothing caches at all.
    #
    # The nonce is what makes this test repeatable. Cache entries live about
    # five minutes, so a fixed preamble means a second run inside that window
    # starts warm and the "first" call reads instead of writing -- the test
    # fails while the code is perfectly correct. A unique prefix per run
    # guarantees a cold start, and doubles as a demonstration of the
    # prefix-match rule: one changed byte at the front invalidates all of it.
    preamble = f"{uuid.uuid4()} Manim guidance for generating scenes. " * 600
    client = llm.Client(max_tokens=64, effort="low")

    first = client.complete("Reply with the single word: ok", preamble=preamble)
    second = client.complete("Reply with the single word: ok", preamble=preamble)

    assert first.usage.cache_write_tokens > 1000
    assert second.usage.cache_read_tokens > 1000
    assert second.usage.cache_hit_rate > 0.9
    # Measured at roughly a twelfth; a wide bound leaves room for pricing to
    # move without turning this into a flake.
    assert second.dollars() < first.dollars() / 3
