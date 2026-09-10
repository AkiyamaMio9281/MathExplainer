"""Prompt assembly and the fenced-block extractor.

The extractor gets most of the attention because it is where a correct file
turns into a failed one. A model that wraps its answer in prose, tags the
fence `py`, or forgets the fence entirely has still produced working code;
every shape the extractor mishandles spends a repair round fixing nothing.

The prompt tests assert the split the cost argument rests on -- the invariant
half in the cached preamble, the scene in the message -- and that the scene
goes over as the same JSON the validator reads, so there is only ever one
description of a scene in the system.

One test renders a generated file for real, marked `api` and `slow`.
"""

from __future__ import annotations

import json

import pytest

from explainer import codegen, ir, llm


def scene(**kwargs) -> ir.Scene:
    base = dict(
        id="intro",
        narration="Every right triangle hides one simple relationship.",
        objects=(
            ir.Text(id="title", content="Pythagoras", position=(0.0, 3.0), font_size=48),
            ir.Polygon(
                id="tri",
                points=((-3.0, -1.0), (0.0, -1.0), (0.0, 1.5)),
                color="BLUE",
            ),
        ),
        steps=(
            ir.Step(action=ir.Action.WRITE, target="title", duration=1.5),
            ir.Step(action=ir.Action.CREATE, target="tri", duration=2.0),
            ir.Step(action=ir.Action.WAIT, duration=1.0),
        ),
    )
    base.update(kwargs)
    return ir.Scene(**base)


# ---------------------------------------------------------------------------
# The extractor
# ---------------------------------------------------------------------------

FILE = "from manim import *\n\n\nclass Demo(Scene):\n    def construct(self):\n        pass"


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param(f"```python\n{FILE}\n```", id="tagged fence"),
        pytest.param(f"```py\n{FILE}\n```", id="short language tag"),
        pytest.param(f"```\n{FILE}\n```", id="untagged fence"),
        pytest.param(f"Here you go:\n\n```python\n{FILE}\n```", id="prose before"),
        pytest.param(
            f"```python\n{FILE}\n```\n\nThat renders the triangle.", id="prose after"
        ),
        pytest.param(FILE, id="no fence at all"),
        pytest.param(f"  ```python\n{FILE}\n```  ", id="surrounding whitespace"),
    ],
)
def test_the_file_survives_however_it_is_wrapped(reply):
    assert codegen.extract_code(reply) == FILE


def test_the_longest_block_wins_not_the_first():
    # A reply that opens with a one-line snippet before the file would
    # otherwise hand back the snippet, and the snippet compiles.
    reply = f"First, the import:\n\n```python\nfrom manim import *\n```\n\nThen:\n\n```python\n{FILE}\n```"

    assert codegen.extract_code(reply) == FILE


def test_an_empty_reply_extracts_to_nothing_rather_than_raising():
    # L0 will reject it in 0.07 ms with a message the model can act on, which
    # is a better failure than an exception here.
    assert codegen.extract_code("") == ""


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_scene_goes_over_as_the_json_the_validator_reads():
    prompt = codegen.scene_prompt(scene(), title="The Pythagorean Theorem")
    body = prompt[prompt.index("{") : prompt.rindex("}") + 1]
    document = json.loads(body)

    assert document["title"] == "The Pythagorean Theorem"
    assert [o["id"] for o in document["scene"]["objects"]] == ["title", "tri"]
    assert document["scene"]["steps"][0] == {
        "action": "write",
        "duration": 1.5,
        "target": "title",
    }


def test_the_scene_json_is_exactly_what_the_ir_round_trips():
    # One description of a scene in the system, not two. A second renderer
    # here is a second thing to drift.
    subject = scene()
    prompt = codegen.scene_prompt(subject)
    body = json.loads(prompt[prompt.index("{") : prompt.rindex("}") + 1])

    assert body["scene"] == ir.Document(scenes=(subject,)).to_dict()["scenes"][0]


def test_narration_is_sent_as_context_and_marked_as_not_being_on_screen():
    prompt = codegen.scene_prompt(scene())

    assert "Every right triangle hides" in prompt
    assert "do not put it on screen" in prompt


def test_a_scene_without_narration_says_nothing_about_it():
    prompt = codegen.scene_prompt(scene(narration=""))

    assert "do not put it on screen" not in prompt


def test_the_preamble_carries_the_vocabulary_and_the_scene_does_not():
    # The cost argument for the split: the invariant half is several thousand
    # tokens and is cached, the varying half is one scene.
    assert "Object types" in codegen.PREAMBLE
    assert "mathtex" in codegen.PREAMBLE
    assert "exactly one class inheriting from `Scene`" in codegen.PREAMBLE
    assert "Object types" not in codegen.scene_prompt(scene())


def test_the_preamble_is_large_enough_to_be_worth_caching():
    # Below the model's minimum cacheable prefix nothing caches, silently.
    assert len(codegen.PREAMBLE) > 2000


def test_the_preamble_states_the_single_scene_class_rule():
    # validate.check_syntax rejects a second Scene subclass at L0, so the
    # prompt has to say so or that rejection is a repair round about nothing.
    assert "More than one is rejected" in codegen.PREAMBLE


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def test_the_repair_prompt_carries_the_code_and_the_whole_traceback():
    prompt = codegen.repair_prompt(FILE, "NameError: name 'NoSuchThing' is not defined")

    assert FILE in prompt
    assert "NameError: name 'NoSuchThing' is not defined" in prompt


def test_a_long_traceback_is_trimmed_to_its_tail():
    error = "filler\n" * 5000 + "ZeroDivisionError: division by zero"
    prompt = codegen.repair_prompt(FILE, error, limit=200)

    # The tail is where the exception is; the head is stack frames.
    assert "ZeroDivisionError" in prompt
    assert prompt.count("filler") < 100


def test_repair_uses_the_same_preamble_so_the_cache_still_applies():
    # A repair round that changes the preamble pays for it again.
    calls = []

    class Recorder(llm.Client):
        def complete(self, prompt, **kwargs):
            calls.append(kwargs.get("preamble"))
            return llm.Reply(
                text=f"```python\n{FILE}\n```",
                usage=llm.Usage(calls=1),
                seconds=0.0,
                model=llm.MODEL,
                stop_reason="end_turn",
            )

    client = Recorder()
    codegen.generate(client, scene())
    codegen.repair(client, FILE, "boom")

    assert calls == [codegen.PREAMBLE, codegen.PREAMBLE]


def test_generate_returns_the_extracted_code_and_the_reply():
    class Stub(llm.Client):
        def complete(self, prompt, **kwargs):
            return llm.Reply(
                text=f"Sure:\n```python\n{FILE}\n```",
                usage=llm.Usage(output_tokens=42, calls=1),
                seconds=1.0,
                model=llm.MODEL,
                stop_reason="end_turn",
            )

    code, reply = codegen.generate(Stub(), scene())

    assert code == FILE
    assert reply.usage.output_tokens == 42


# ---------------------------------------------------------------------------
# One real generation, rendered
# ---------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.slow
def test_a_generated_scene_passes_validation(tmp_path):
    from explainer.validate import Level, validate

    code, reply = codegen.generate(
        llm.Client(), scene(), title="The Pythagorean Theorem"
    )
    result = validate(tmp_path, code, up_to=Level.DRYRUN)

    assert result.ok, f"{result.error}\n\n--- generated ---\n{code}"
    assert result.scene_name
    print(f"\ngenerated {len(code)} chars for ${reply.dollars():.4f}")
