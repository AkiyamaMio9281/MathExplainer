"""Validation rules over a typed Scene IR document.

Pure functions, no LLM and no subprocess: these run in microseconds, and every
defect caught here is one that would otherwise survive a clean compile and
surface only when a human watched the video.

`ir.py` owns the rules that decide whether a typed object can be built at all
-- `known-type`, `known-action`, `required-fields`, `field-shape`. This module
owns the seven that need a built document in order to run.

The two that cost more than a few lines are `introduced-before-use` and
`not-after-removal`, and they share one walk of the timeline. `walk` replays
the steps while tracking which objects are on screen and which have been
removed, yielding a `Moment` per step. Commit 16's `no-overlap` needs the same
replay -- the set of *simultaneously visible* objects is exactly what a
bounding-box comparison needs -- so it is built once here and extended there
with placement.

Two deliberate choices in that walk:

* **It models what the renderer would do, not what the rules wish it did.**
  Re-introducing an object after `fade_out` is reported by
  `not-after-removal`, and the walk still puts it back on screen, because that
  is what would actually happen in the output. Rules describe defects; the
  walk describes reality.
* **`not-after-removal` reports once per object.** A scene that keeps acting
  on something it faded out would otherwise emit one issue per subsequent
  step, and a repair prompt buried in twelve copies of the same sentence is
  worse than one that names the problem once.

A rule never reports what another rule already owns. A step whose target is
not declared is `reference-integrity`'s business, so the timeline and
`action-applies` skip it rather than adding a second complaint about the same
character.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

from . import ir
from .ir import Issue, Severity

#: Everything this module can emit. Metrics record which rule caught which
#: defect, and the test suite asserts that each of these has a case.
RULE_NAMES = frozenset(
    {
        "non-empty",
        "unique-ids",
        "reference-integrity",
        "introduced-before-use",
        "not-after-removal",
        "positive-duration",
        "action-applies",
    }
)

#: `write` draws a glyph outline stroke by stroke, so it means nothing on a
#: shape; `create` is the converse.
TEXTUAL = (ir.Text, ir.MathTex)


# ---------------------------------------------------------------------------
# The timeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Moment:
    """The scene either side of one step."""

    index: int
    step: ir.Step
    visible_before: frozenset[str]
    visible_after: frozenset[str]
    removed_before: frozenset[str]


def walk(scene: ir.Scene) -> Iterator[Moment]:
    """Replay *scene*'s steps, tracking what is on screen at each one."""
    visible: set[str] = set()
    removed: set[str] = set()

    for index, step in enumerate(scene.steps):
        before = frozenset(visible)
        removed_before = frozenset(removed)
        target = step.target

        if step.action is ir.Action.TRANSFORM:
            # transform consumes its target and leaves `into` in its place, so
            # `into` needs no separate introduction.
            if target:
                visible.discard(target)
                removed.add(target)
            if step.into:
                visible.add(step.into)
        elif step.action is ir.Action.FADE_OUT:
            if target:
                visible.discard(target)
                removed.add(target)
        elif step.action in ir.INTRODUCTIONS and target:
            visible.add(target)
            # Reported by not-after-removal, but still shown: the renderer
            # would put it back, and the following steps should be judged
            # against a screen that has it.
            removed.discard(target)

        yield Moment(index, step, before, frozenset(visible), removed_before)


def visible_at(scene: ir.Scene) -> Iterator[tuple[Moment, frozenset[str]]]:
    """Each moment with the objects on screen while its step plays.

    An introduction is on screen during its own step, so this is the state
    *after* the step is applied -- which is what a layout check wants.
    """
    for moment in walk(scene):
        yield moment, moment.visible_after


# ---------------------------------------------------------------------------
# Document rules
# ---------------------------------------------------------------------------


def document_non_empty(document: ir.Document) -> Iterable[Issue]:
    if not document.scenes:
        yield Issue("non-empty", "a document needs at least one scene", "$")


def unique_scene_ids(document: ir.Document) -> Iterable[Issue]:
    # Scene ids name working directories and output files, so a duplicate is
    # not a cosmetic clash: the second render would overwrite the first.
    seen: dict[str, int] = {}
    for index, scene in enumerate(document.scenes):
        if scene.id in seen:
            yield Issue(
                "unique-ids",
                f"duplicate scene id {scene.id!r}, first used at scenes[{seen[scene.id]}]",
                f"scenes[{index}]",
            )
        else:
            seen[scene.id] = index


# ---------------------------------------------------------------------------
# Scene rules
# ---------------------------------------------------------------------------


def non_empty(scene: ir.Scene, where: str) -> Iterable[Issue]:
    if not scene.objects:
        yield Issue("non-empty", "a scene needs at least one object", where)
    if not scene.steps:
        yield Issue("non-empty", "a scene needs at least one step", where)


def unique_ids(scene: ir.Scene, where: str) -> Iterable[Issue]:
    seen: dict[str, int] = {}
    for index, obj in enumerate(scene.objects):
        if obj.id in seen:
            yield Issue(
                "unique-ids",
                f"duplicate object id {obj.id!r}, first declared at objects[{seen[obj.id]}]",
                f"{where}.objects[{index}]",
            )
        else:
            seen[obj.id] = index


def reference_integrity(scene: ir.Scene, where: str) -> Iterable[Issue]:
    declared = {obj.id: obj for obj in scene.objects}

    for index, obj in enumerate(scene.objects):
        if not isinstance(obj, ir.Plot):
            continue
        referenced = declared.get(obj.axes)
        if referenced is None:
            yield Issue(
                "reference-integrity",
                f"plot references axes {obj.axes!r}, which is not declared",
                f"{where}.objects[{index}]",
            )
        elif not isinstance(referenced, ir.Axes):
            yield Issue(
                "reference-integrity",
                f"plot references {obj.axes!r}, which is a "
                f"{type(referenced).TYPE} rather than an axes",
                f"{where}.objects[{index}]",
            )

    for index, step in enumerate(scene.steps):
        for field in ("target", "into"):
            reference = getattr(step, field)
            if reference is not None and reference not in declared:
                yield Issue(
                    "reference-integrity",
                    f"{step.action.value} {field} {reference!r} is not a declared object",
                    f"{where}.steps[{index}]",
                )


def positive_duration(scene: ir.Scene, where: str) -> Iterable[Issue]:
    for index, step in enumerate(scene.steps):
        if step.duration <= 0:
            yield Issue(
                "positive-duration",
                f"duration must be greater than 0, got {step.duration}",
                f"{where}.steps[{index}]",
            )


def action_applies(scene: ir.Scene, where: str) -> Iterable[Issue]:
    declared = {obj.id: obj for obj in scene.objects}

    for index, step in enumerate(scene.steps):
        obj = declared.get(step.target) if step.target else None
        if obj is None:
            continue  # reference-integrity owns this one
        kind = type(obj).TYPE
        if step.action is ir.Action.WRITE and not isinstance(obj, TEXTUAL):
            yield Issue(
                "action-applies",
                f"write applies to text and mathtex, not to {kind}; use create",
                f"{where}.steps[{index}]",
            )
        elif step.action is ir.Action.CREATE and isinstance(obj, TEXTUAL):
            yield Issue(
                "action-applies",
                f"create does not apply to {kind}; use write",
                f"{where}.steps[{index}]",
            )


def timeline(scene: ir.Scene, where: str) -> Iterable[Issue]:
    """`introduced-before-use` and `not-after-removal`, from one replay."""
    declared = {obj.id for obj in scene.objects}
    reported: set[str] = set()

    for moment in walk(scene):
        step = moment.step
        target = step.target
        if target is None or target not in declared:
            continue  # reference-integrity owns undeclared targets

        location = f"{where}.steps[{moment.index}]"
        if target in moment.removed_before:
            if target not in reported:
                reported.add(target)
                yield Issue(
                    "not-after-removal",
                    f"{target!r} was removed earlier, so {step.action.value} "
                    "cannot apply to it",
                    location,
                )
        elif step.action not in ir.INTRODUCTIONS and target not in moment.visible_before:
            yield Issue(
                "introduced-before-use",
                f"{step.action.value} acts on {target!r} before anything introduces it",
                location,
            )


DocumentRule = Callable[[ir.Document], Iterable[Issue]]
SceneRule = Callable[[ir.Scene, str], Iterable[Issue]]

DOCUMENT_RULES: tuple[DocumentRule, ...] = (document_non_empty, unique_scene_ids)

SCENE_RULES: tuple[SceneRule, ...] = (
    non_empty,
    unique_ids,
    reference_integrity,
    positive_duration,
    action_applies,
    timeline,
)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def check_scene(scene: ir.Scene, where: str = "$") -> tuple[Issue, ...]:
    issues: list[Issue] = []
    for rule in SCENE_RULES:
        issues.extend(rule(scene, where))
    return tuple(issues)


def check_document(document: ir.Document) -> tuple[Issue, ...]:
    """Every issue in *document*, in rule order within each scene."""
    issues: list[Issue] = []
    for rule in DOCUMENT_RULES:
        issues.extend(rule(document))
    for index, scene in enumerate(document.scenes):
        issues.extend(check_scene(scene, f"scenes[{index}]"))
    return tuple(issues)


def errors(issues: Iterable[Issue]) -> tuple[Issue, ...]:
    return tuple(i for i in issues if i.severity is Severity.ERROR)


def report(issues: Iterable[Issue]) -> str:
    """The issues as text, for feeding back to the model."""
    return "\n".join(str(i) for i in issues)
