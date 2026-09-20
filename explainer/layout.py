"""A lesson plan to a Scene IR document.

The layout stage. It is handed the pedagogy -- what each scene teaches and the
words spoken over it -- and decides what is on screen and when. It does not
get to change the words: narration is carried across from the plan verbatim
rather than asked for and echoed back, which is what makes "repair the layout
without the explanation changing underneath it" true by construction rather
than by hoping.

That has a second consequence worth stating. The plan fixes the narration, and
the narration fixes how long the scene runs, so **step durations are derived
from the words rather than chosen**. A scene whose animation runs three times
its narration is the most common thing this stage gets wrong when it is not
told, and it was the first thing the API produced when tried.

## What the schema can carry, and why it is not the whole vocabulary

Structured outputs compile the schema into a grammar, and that grammar has a
size limit this format runs into. Measured against the API, each as its own
400:

* `minItems` is accepted only as 0 or 1; anything else is rejected by value.
* `maxItems` is not supported at all.
* `additionalProperties` must be present on every object, and must be `false`
  -- an open object is not an option.
* A tagged union of all nine object types is "too large", at either document
  or scene level. So is eight, and so is seven. Six compiles, at about 3.5 kB
  of schema.
* Merging the nine into one object with every field optional is "too complex"
  -- the cost is in the optional properties, not the nesting.

So the generator works in six of the nine types. `axes` and `plot` stay in the
IR and remain valid in a hand-written or simplified document; they are simply
outside what this stage will produce, which sits well with the escalation
ladder already calling them the most failure-prone types. `rectangle` is
omitted because a polygon expresses one.

Everything the schema cannot say -- that a point is exactly two numbers, that
a circle needs a radius -- is said by `ir.parse_document` and the rules, which
already check it and name the field. The schema pins the vocabulary; the
parser pins the arithmetic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Iterable

from . import ir, ir_rules, llm, plan as planning
from .ir import Issue, Severity

#: The object types this stage will produce. Six, because the seventh does not
#: compile -- see the module docstring. Ordered by how much of a lesson they
#: carry.
GENERATED_TYPES = ("text", "mathtex", "polygon", "circle", "line", "arrow")

#: How far the animation may run from the narration it is spoken over. The
#: IR's own pacing rule, so this stage aims at what the next one checks.
PACING_TOLERANCE = ir_rules.PACING_TOLERANCE

#: A whole document at once, and thinking is billed as output: a five-scene
#: layout exhausted the client's default before it had written anything.
#: Above llm.STREAM_ABOVE this streams, which is what makes the larger ceiling
#: safe rather than a timeout.
MAX_TOKENS = 32_000


def schema() -> dict[str, Any]:
    """The document schema this stage generates against.

    Narration is removed rather than requested: it comes from the plan, and
    asking for it back would spend output tokens on a copy that might not be
    one.
    """
    document = ir.document_schema()
    scene = document["properties"]["scenes"]["items"]
    scene["properties"].pop("narration", None)
    scene["required"] = [f for f in scene["required"] if f != "narration"]
    scene["properties"]["objects"]["items"] = {
        "anyOf": [ir.object_schema(ir.OBJECT_TYPES[name]) for name in GENERATED_TYPES]
    }
    return document


PREAMBLE = f"""\
You lay out mathematical explainer videos. You are given a lesson plan -- what
each scene teaches and the words spoken over it -- and you return a
declarative description of what is on screen and when.

You do not write the narration. It is given, and it is already final.

# The frame

Manim units. The frame is 14.222 by 8.0, so x runs from -7.111 to 7.111 and y
from -4.0 to 4.0, with [0, 0] at the centre.

**Keep everything inside x from -6.6 to 6.6 and y from -3.6 to 3.6.** An
object placed at the edge spills past it and is cut off in the video, which
renders without any error at all.

Estimate what things occupy before placing them. A line of text is about
0.25 units wide per character and 0.5 tall, both scaled by font_size/36; a
formula about 0.16 per character and 0.6 tall. Two things whose boxes overlap
are unreadable, and overlapping text is the single most common way a generated
scene looks broken. Put a title at the top, a formula in the middle, a caption
below -- separated in y, not stacked at the same point.

# Objects

Every object has an `id` and a `type`. Ids are lowercase identifiers --
letters, digits and underscores, starting with a letter -- and unique within a
scene. They become variable names, so `right_triangle`, not `Right Triangle`.

- `text`: content, position. Optional font_size (default 36), color.
- `mathtex`: content (LaTeX, no dollar signs), position. Optional font_size
  (default 48), color.
- `polygon`: points, three or more [x, y]. Optional color, fill_opacity.
- `circle`: center, radius. Optional color, fill_opacity.
- `line`: start, end. Optional color, stroke_width.
- `arrow`: start, end. Optional color.

Colours are names: WHITE, BLUE, YELLOW, GREEN, RED, ORANGE, PURPLE, GREY.

# Steps

`steps` is the timeline. An object is not on screen until a step introduces
it. Every step has an `action` and a `duration` in seconds.

- `write` -- text and mathtex only. Draws it on stroke by stroke.
- `create` -- shapes only, never text. Draws it on.
- `fade_in` -- anything.
- `fade_out` -- anything. The object is gone afterwards; do not act on it
  again.
- `transform` with `into` -- morphs the target into another declared object.
  The target is consumed and `into` is left in its place, already on screen.
- `move` with `to` -- translates it to [x, y]. Check the new position is still
  inside the frame.
- `indicate` -- a brief flash of attention. The object stays.
- `wait` -- holds the frame. No target.

Never act on an object before something introduces it. Never act on one after
`fade_out`, or after it has been the target of a `transform`.

# Timing

**The narration decides how long a scene runs.** It is read at 150 words a
minute, so a scene of N words runs N/150*60 seconds, and the step durations
must add up to about that -- within {PACING_TOLERANCE:.0%}. Each scene's word
count is given below. Work out the budget first, then spend it across the
steps: an animation that finishes in a third of the narration leaves the
viewer watching a still frame, and one that runs twice as long is worse.

A `wait` at the end is a legitimate way to spend what is left over.

# Scenes

Return one scene per scene in the plan, in the same order, **with the same
`id`**. The ids are how the narration is matched back on.
"""


def plan_prompt(lesson: planning.LessonPlan) -> str:
    """The user half: the plan, with each scene's timing budget worked out."""
    lines = [
        f"Lay out this lesson: {lesson.title}",
        f"Audience: {lesson.audience}" if lesson.audience else "",
        "",
    ]
    for scene in lesson.scenes:
        words = len(scene.narration.split())
        lines += [
            f"## {scene.id}",
            f"Teaches: {scene.beat}",
            f"Narration ({words} words, so the steps must total about "
            f"{scene.spoken_seconds:.1f}s):",
            scene.narration,
            "",
        ]
    return "\n".join(line for line in lines if line is not None)


def retry_prompt(document: ir.Document, issues: Iterable[Issue]) -> str:
    """Ask for a corrected document, given what the rules said.

    The issues go back verbatim, each already carrying the path that produced
    it, so the fix does not have to be located first.
    """
    return "\n".join(
        [
            "This layout was rejected. Return a corrected one, keeping the same",
            "scene ids and the same order.",
            "",
            "```json",
            json.dumps(document.to_dict(), indent=1),
            "```",
            "",
            "It was rejected for:",
            "",
            *(f"- {issue}" for issue in issues),
        ]
    )


def carry_narration(
    document: ir.Document, lesson: planning.LessonPlan
) -> tuple[ir.Document, tuple[Issue, ...]]:
    """Put the plan's narration onto the document, matched by scene id.

    The words are the plan's and stay the plan's. This is also where a layout
    that renamed or dropped a scene is caught: without a matching id there is
    nothing to attach, and the lesson would render a scene with no narration
    to be timed against.
    """
    issues: list[Issue] = []
    narration = {scene.id: scene.narration for scene in lesson.scenes}
    seen = set()

    scenes = []
    for index, scene in enumerate(document.scenes):
        if scene.id in narration:
            seen.add(scene.id)
            scenes.append(replace(scene, narration=narration[scene.id]))
        else:
            issues.append(
                Issue(
                    "matches-plan",
                    f"scene {scene.id!r} is not in the plan, so it has no narration",
                    f"scenes[{index}]",
                )
            )
            scenes.append(scene)

    for missing in (s.id for s in lesson.scenes if s.id not in seen):
        issues.append(
            Issue("matches-plan", f"the plan's scene {missing!r} was not laid out", "$")
        )

    return replace(document, scenes=tuple(scenes)), tuple(issues)


def check(document: ir.Document, lesson: planning.LessonPlan) -> tuple[Issue, ...]:
    """Everything wrong with this layout, including what the plan asked for."""
    _, mismatches = carry_narration(document, lesson)
    return mismatches + ir_rules.check_document(document)


def errors(issues: Iterable[Issue]) -> tuple[Issue, ...]:
    return tuple(i for i in issues if i.severity is Severity.ERROR)


def report(issues: Iterable[Issue]) -> str:
    return "\n".join(str(i) for i in issues)


def retry(
    client: llm.Client,
    lesson: planning.LessonPlan,
    document: ir.Document,
    issues: Iterable[Issue],
    *,
    effort: str | None = None,
) -> tuple[ir.Document | None, tuple[Issue, ...], llm.Reply]:
    """Ask for the same layout again, with what was wrong with it.

    The same preamble and the same schema, so the cached prefix survives and
    the correction costs a fraction of the first attempt.
    """
    reply = client.complete(
        retry_prompt(document, issues),
        preamble=PREAMBLE,
        schema=schema(),
        effort=effort,
        max_tokens=MAX_TOKENS,
    )
    return _read(reply, lesson)


def make(
    client: llm.Client,
    lesson: planning.LessonPlan,
    *,
    effort: str | None = None,
) -> tuple[ir.Document | None, tuple[Issue, ...], llm.Reply]:
    """Lay out *lesson*. Returns the document, its issues, and the cost."""
    reply = client.complete(
        plan_prompt(lesson),
        preamble=PREAMBLE,
        schema=schema(),
        effort=effort,
        max_tokens=MAX_TOKENS,
    )
    return _read(reply, lesson)


def _read(
    reply: llm.Reply, lesson: planning.LessonPlan
) -> tuple[ir.Document | None, tuple[Issue, ...], llm.Reply]:
    parsed = ir.parse_json(reply.text)
    if parsed.document is None:
        return None, parsed.issues, reply

    document = replace(parsed.document, title=lesson.title or parsed.document.title)
    document, mismatches = carry_narration(document, lesson)
    issues = parsed.issues + mismatches + ir_rules.check_document(document)
    return document, issues, reply
