"""A one-line prompt to a lesson plan.

The first stage, and the only one concerned with pedagogy: what to teach, in
what order, and in what words. It says nothing about what appears on screen --
that is the Scene IR's job, and keeping the two apart is what lets a layout be
repaired without the explanation quietly changing underneath it.

The plan comes back through structured outputs, so the shape is guaranteed by
the API rather than parsed out of prose. What is *not* guaranteed is that a
schema-valid plan is a usable one, and that gap is what `check` is for. Four
things can be wrong in a plan that validates perfectly:

* **Ids that are not identifiers.** A scene id becomes a working directory and
  survives into the IR, so model-supplied text ends up building a path. It is
  also the one field here that has a security consequence rather than an
  aesthetic one.
* **Too many scenes.** Every scene is a codegen call, a validation, and a
  render. A plan with twenty of them is a budget being spent, and it is
  cheaper to reject it here than to notice halfway through.
* **Narration that implies an impossible scene.** The IR's `scene-length` rule
  wants 5 to 90 seconds, which at 150 words a minute is 13 to 225 words. Those
  bounds are computed from the IR's own constants rather than written down
  here, because the round numbers are wrong in a way that matters: twelve
  words reads in 4.8 seconds, so a plan sitting on a hand-written lower bound
  would produce scenes the next stage warns about.
* **A lesson of one scene.** That is an answer, not an explanation.

Rejecting those here rather than downstream is the same argument the tier
ladder makes about code: the cheapest check that can catch a defect should.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from . import ir, ir_rules, llm
from .ir import Issue, Severity

#: Words per minute, taken from the IR's pacing rule rather than repeated, so
#: the two stages cannot disagree about what a second of narration is.
WORDS_PER_MINUTE = ir_rules.WORDS_PER_MINUTE

#: The IR's scene-length bounds, converted to words. Derived rather than
#: written down: the round numbers are wrong. Twelve words reads in 4.8
#: seconds, which is under the five the IR rules allow, so a plan sitting on
#: its own lower bound would produce scenes the next stage warns about.
MIN_WORDS = math.ceil(ir_rules.MIN_SCENE_SECONDS * WORDS_PER_MINUTE / 60)
MAX_WORDS = math.floor(ir_rules.MAX_SCENE_SECONDS * WORDS_PER_MINUTE / 60)

#: Each scene costs a generation, a validation and a render.
MIN_SCENES = 2
MAX_SCENES = 8

RULE_NAMES = frozenset(
    {"non-empty", "unique-ids", "safe-ids", "scene-count", "narration-length", "has-title"}
)


@dataclass(frozen=True)
class Beat:
    """One scene, as pedagogy rather than as layout."""

    id: str
    beat: str  # what this scene teaches, for the IR stage to lay out
    narration: str  # the words spoken over it
    #: How long this narration takes to say, once something has said it. Zero
    #: until then, which is the honest value -- an unmeasured scene has an
    #: estimate, not a duration.
    seconds: float = 0.0

    @property
    def measured(self) -> bool:
        return self.seconds > 0.0

    @property
    def spoken_seconds(self) -> float:
        """The scene's length: measured if it has been, estimated otherwise.

        Both callers -- the layout's timing budget and the pacing rule -- want
        the best number available rather than specifically the estimate, so the
        substitution happens here instead of at each of them.
        """
        if self.measured:
            return self.seconds
        return len(self.narration.split()) / WORDS_PER_MINUTE * 60.0


@dataclass(frozen=True)
class LessonPlan:
    title: str = ""
    audience: str = ""
    scenes: tuple[Beat, ...] = ()

    @property
    def spoken_seconds(self) -> float:
        return sum(scene.spoken_seconds for scene in self.scenes)

    @property
    def measured(self) -> bool:
        """Whether every scene's length is known rather than predicted."""
        return bool(self.scenes) and all(scene.measured for scene in self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "audience": self.audience,
            "scenes": [
                {"id": s.id, "beat": s.beat, "narration": s.narration}
                for s in self.scenes
            ],
        }


SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "audience": {"type": "string"},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "beat": {"type": "string"},
                    "narration": {"type": "string"},
                },
                "required": ["id", "beat", "narration"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "audience", "scenes"],
    "additionalProperties": False,
}

# Counts are stated in the prompt rather than as minItems/maxItems in the
# schema: the rules below enforce them either way, and a schema keyword the
# API declines to accept would be a 400 on every call.
PREAMBLE = f"""\
You plan short mathematical explainer videos. Given a topic, you return the
pedagogy: what to teach, in what order, and the words spoken over each part.

You are not describing what appears on screen. A later stage decides the
layout. Say what each scene is *for*, and write the narration.

# Shape

Between {MIN_SCENES} and {MAX_SCENES} scenes. Each one carries a single idea,
and the sequence should build: state the thing, show why it is true, land the
conclusion. A lesson of one scene is an answer rather than an explanation.

Each scene has:

- `id` -- a short lowercase identifier, letters, digits and underscores only,
  starting with a letter. It becomes a directory name and a variable name, so
  `right_triangle` is fine and `Right Triangle` is not.
- `beat` -- one sentence saying what this scene teaches. The layout stage
  reads this to decide what to draw, so name the objects it should show.
- `narration` -- the words spoken over the scene.

# Narration

Between {MIN_WORDS} and {MAX_WORDS} words per scene. It is read aloud at about
{WORDS_PER_MINUTE:.0f} words a minute, and the animation is timed to match, so
the length is what sets how long the scene runs.

Write it to be heard. No LaTeX, no symbols that only work written down: say
"a squared plus b squared" rather than "a^2 + b^2". No "as you can see" or "in
this diagram" -- the viewer is listening, and the visuals may change.

Prefer the concrete over the general. One worked instance a viewer can hold
beats a correct statement they cannot picture.
"""


def parse(data: Any) -> tuple[LessonPlan | None, tuple[Issue, ...]]:
    """Type a plan, collecting anything wrong rather than raising."""
    issues: list[Issue] = []
    if not isinstance(data, Mapping):
        return None, (Issue("plan-shape", "the plan must be an object", "$"),)

    title = data.get("title", "")
    audience = data.get("audience", "")
    if not isinstance(title, str):
        issues.append(Issue("plan-shape", "title must be a string", "$.title"))
        title = ""
    if not isinstance(audience, str):
        issues.append(Issue("plan-shape", "audience must be a string", "$.audience"))
        audience = ""

    raw = data.get("scenes", [])
    if not isinstance(raw, list):
        return None, (*issues, Issue("plan-shape", "scenes must be a list", "$.scenes"))

    beats = []
    for index, entry in enumerate(raw):
        where = f"scenes[{index}]"
        if not isinstance(entry, Mapping):
            issues.append(Issue("plan-shape", "a scene must be an object", where))
            continue
        missing = [f for f in ("id", "beat", "narration") if not isinstance(entry.get(f), str)]
        if missing:
            issues.append(
                Issue("plan-shape", f"a scene needs {', '.join(missing)}", where)
            )
            continue
        beats.append(
            Beat(id=entry["id"], beat=entry["beat"], narration=entry["narration"])
        )

    return LessonPlan(title=title, audience=audience, scenes=tuple(beats)), tuple(issues)


def check(plan: LessonPlan) -> tuple[Issue, ...]:
    """What can be wrong with a plan the schema already accepted."""
    issues: list[Issue] = []

    if not plan.title.strip():
        issues.append(Issue("has-title", "a lesson needs a title", "$.title"))

    if not plan.scenes:
        issues.append(Issue("non-empty", "a plan needs at least one scene", "$"))
        return tuple(issues)

    if len(plan.scenes) < MIN_SCENES:
        issues.append(
            Issue(
                "scene-count",
                f"{len(plan.scenes)} scene is an answer rather than an explanation; "
                f"want at least {MIN_SCENES}",
                "$.scenes",
            )
        )
    elif len(plan.scenes) > MAX_SCENES:
        # Each one is a generation, a validation and a render.
        issues.append(
            Issue(
                "scene-count",
                f"{len(plan.scenes)} scenes is more than the {MAX_SCENES} this "
                "pipeline will render in one run",
                "$.scenes",
            )
        )

    seen: dict[str, int] = {}
    for index, scene in enumerate(plan.scenes):
        where = f"scenes[{index}]"
        if scene.id in seen:
            issues.append(
                Issue(
                    "unique-ids",
                    f"duplicate scene id {scene.id!r}, first used at scenes[{seen[scene.id]}]",
                    where,
                )
            )
        else:
            seen[scene.id] = index

        if not ir.is_safe_id(scene.id):
            issues.append(
                Issue(
                    "safe-ids",
                    f"scene id {scene.id!r} must be a lowercase identifier: it "
                    "names a directory on disk",
                    where,
                )
            )

        words = len(scene.narration.split())
        if words < MIN_WORDS:
            issues.append(
                Issue(
                    "narration-length",
                    f"{words} words reads in {scene.spoken_seconds:.1f}s, under the "
                    "5s the IR rules allow for a scene",
                    where,
                    Severity.WARNING,
                )
            )
        elif words > MAX_WORDS:
            issues.append(
                Issue(
                    "narration-length",
                    f"{words} words reads in {scene.spoken_seconds:.1f}s, over the "
                    "90s the IR rules allow for a scene",
                    where,
                    Severity.WARNING,
                )
            )

    return tuple(issues)


def errors(issues: Iterable[Issue]) -> tuple[Issue, ...]:
    return tuple(i for i in issues if i.severity is Severity.ERROR)


def report(issues: Iterable[Issue]) -> str:
    return "\n".join(str(i) for i in issues)


def make(
    client: llm.Client, prompt: str, *, effort: str | None = None
) -> tuple[LessonPlan | None, tuple[Issue, ...], llm.Reply]:
    """Plan a lesson for *prompt*. Returns the plan, its issues, and the cost."""
    import json

    reply = client.complete(
        f"Plan a lesson for this: {prompt}",
        preamble=PREAMBLE,
        schema=SCHEMA,
        effort=effort,
    )
    try:
        data = json.loads(reply.text)
    except ValueError as exc:
        return None, (Issue("plan-shape", f"invalid JSON: {exc}", "$"),), reply

    plan, issues = parse(data)
    if plan is None:
        return None, issues, reply
    return plan, issues + check(plan), reply
