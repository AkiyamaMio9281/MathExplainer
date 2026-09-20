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

The four warnings are layout and timing checks. They do not block a render --
they are recorded, and may drive `simplify_ir` -- because the geometry is
estimated rather than computed: exact text extents need a font engine this
module deliberately does not load.

**The estimator's constants are measured, not guessed.** The first draft of
SCENE_IR.md put text at `0.55 * font_size/36` units per character and one unit
tall, which is roughly twice Manim's real extents in both directions; under
those numbers the title in the specification's own example came out 16.9 units
wide against a 14.2-unit frame, so `in-frame` would have rejected the canonical
document. Measured against ManimCE 0.21.0, a character is about 0.23 units
wide per `font_size/36` and a line 0.24 to 0.49 tall depending on ascenders and
descenders. The values below sit just above the measured means, because an
estimate that is slightly too large turns a near-miss into a warning, while one
that is too large by a factor of two turns every scene into one.

The margins are asymmetric, and the bottom one is not about spill. The
finished lesson carries burned-in subtitles, measured at y -3.25 to -2.52, so
anything the layout puts there is covered by the very words it was timed
against. The top margin went the other way: at 3.6 it warned about titles
reaching 3.68 against a frame edge of 4.0, none of which were clipped.

Axes are the other correction. SCENE_IR.md derived their size from `x_range`
and `y_range`; Manim does not -- it sizes axes from the frame and uses the
range for tick labels, so an axes is 12 by 6 whatever range it carries. That
also settles `no-overlap`: an axes is a backdrop covering most of the frame,
and objects are meant to be drawn over it, so axes and plots are left out of
the overlap comparison. Including them would fire on nearly every scene that
plots anything, and the rule exists to catch overlapping text.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Mapping

from . import ir
from .ir import Issue, Severity

#: Rules that reject a document.
ERROR_RULES = frozenset(
    {
        "non-empty",
        "unique-ids",
        "reference-integrity",
        "introduced-before-use",
        "not-after-removal",
        "positive-duration",
        "action-applies",
        "safe-ids",
    }
)

#: Rules that are recorded and may drive simplification, but do not block a
#: render. All four rest on estimated geometry or on a rule of thumb.
WARNING_RULES = frozenset({"in-frame", "no-overlap", "pacing", "scene-length"})

#: Everything this module can emit. Metrics record which rule caught which
#: defect, and the test suite asserts that each of these has a case.
RULE_NAMES = ERROR_RULES | WARNING_RULES

# Frame geometry, from manim's default config and independent of render
# quality: -ql changes pixels, not units.
FRAME_WIDTH = 14.222
FRAME_HEIGHT = 8.0

#: The margins in-frame enforces, and they are not symmetric.
#:
#: Horizontally and at the top, the margin exists because an estimated box
#: that just fits tends to spill in the render. The top was 3.6 and that was
#: too tight: on a real run it warned five times about titles reaching 3.68
#: against a frame edge of 4.0, none of which were clipped in the video. 3.8
#: keeps a fifth of a unit of slack and stops crying wolf.
#:
#: The bottom is a different quantity entirely. Subtitles are burned into the
#: finished lesson, and measured on an 854x480 render a two-line cue occupies
#: y -3.25 to -2.52. Anything the layout puts down there is covered by the
#: words it was timed against. The band is reserved whether or not this
#: particular run burns subtitles: they are the default, and a layout that is
#: only correct with them switched off is a layout that breaks by default.
MARGIN_X = 6.6
MARGIN_TOP = 3.8
MARGIN_BOTTOM = 2.4

#: Where the burned-in subtitles sit, measured rather than assumed.
SUBTITLE_BAND = (-3.25, -2.52)

# Text extents, measured against ManimCE 0.21.0 and expressed per font_size/36.
# See the module docstring for why these are not the numbers SCENE_IR.md first
# carried.
TEXT_WIDTH_PER_CHAR = 0.25  # measured mean 0.233, max 0.257
TEXT_HEIGHT = 0.50  # measured 0.241 to 0.486, by ascender and descender
MATHTEX_WIDTH_PER_CHAR = 0.16  # measured mean 0.134, max 0.186
MATHTEX_HEIGHT = 0.60  # measured 0.169 to 0.772; fractions are tall

# Manim sizes axes from the frame, not from x_range/y_range.
AXES_WIDTH = 12.0
AXES_HEIGHT = 6.0

#: A normal explainer pace, and how far narration may drift from animation.
WORDS_PER_MINUTE = 150.0
PACING_TOLERANCE = 0.30

#: Below the first a scene reads as a glitch; above the second it loses the
#: viewer.
MIN_SCENE_SECONDS = 5.0
MAX_SCENE_SECONDS = 90.0

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


def safe_scene_ids(document: ir.Document) -> Iterable[Issue]:
    # A scene id becomes a working directory. Model-supplied text used to
    # build a path is how a lesson writes outside the directory it was given.
    for index, scene in enumerate(document.scenes):
        if not ir.is_safe_id(scene.id):
            yield Issue(
                "safe-ids",
                f"scene id {scene.id!r} must be a lowercase identifier: it "
                "names a directory on disk",
                f"scenes[{index}]",
            )


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


def safe_ids(scene: ir.Scene, where: str) -> Iterable[Issue]:
    # An object id becomes a local variable in the generated file. Lowercase
    # keeps it clear of Manim's CamelCase classes and ALLCAPS colours; the
    # keyword check keeps it clear of Python itself.
    for index, obj in enumerate(scene.objects):
        if not ir.is_safe_id(obj.id):
            yield Issue(
                "safe-ids",
                f"object id {obj.id!r} must be a lowercase identifier: it "
                "becomes a variable name in the generated code",
                f"{where}.objects[{index}]",
            )


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


# ---------------------------------------------------------------------------
# Estimated geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned bounding box in Manim units."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def around(cls, centre: ir.Point, width: float, height: float) -> "Box":
        x, y = centre
        return cls(x - width / 2, y - height / 2, x + width / 2, y + height / 2)

    @classmethod
    def containing(cls, points: Iterable[ir.Point]) -> "Box":
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return cls(min(xs), min(ys), max(xs), max(ys))

    @property
    def centre(self) -> ir.Point:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def moved_to(self, centre: ir.Point) -> "Box":
        """The same box with its centre at *centre*.

        A ``move`` translates whatever the object is, so modelling it as a
        translation of the box works for every type without asking which of
        position, center, start or points it happens to carry.
        """
        return Box.around(centre, self.width, self.height)

    def intersects(self, other: "Box") -> bool:
        """True when the boxes share area. Touching edges do not count."""
        return (
            self.min_x < other.max_x
            and other.min_x < self.max_x
            and self.min_y < other.max_y
            and other.min_y < self.max_y
        )

    def fits(self, side: float, top: float, bottom: float) -> bool:
        """Whether the box is inside the usable frame.

        Three numbers rather than two: the bottom is reserved for subtitles,
        so the frame an object may occupy is not centred on the origin.
        """
        return (
            -side <= self.min_x
            and self.max_x <= side
            and -bottom <= self.min_y
            and self.max_y <= top
        )

    def outside(self, side: float, top: float, bottom: float) -> str:
        """Which edge it crossed, for a message worth acting on."""
        if self.min_x < -side or self.max_x > side:
            return f"past the side margin of {side}"
        if self.max_y > top:
            return f"above the top margin of {top}"
        return f"into the subtitle band below y {-bottom}"


def bounding_box(
    obj: ir.SceneObject, declared: Mapping[str, ir.SceneObject] | None = None
) -> Box | None:
    """An estimate of what *obj* occupies, or None when it cannot be placed."""
    if isinstance(obj, ir.Text):
        scale = obj.font_size / 36.0
        return Box.around(
            obj.position,
            TEXT_WIDTH_PER_CHAR * len(obj.content) * scale,
            TEXT_HEIGHT * scale,
        )
    if isinstance(obj, ir.MathTex):
        scale = obj.font_size / 36.0
        return Box.around(
            obj.position,
            MATHTEX_WIDTH_PER_CHAR * len(obj.content) * scale,
            MATHTEX_HEIGHT * scale,
        )
    if isinstance(obj, ir.Polygon):
        return Box.containing(obj.points)
    if isinstance(obj, ir.Circle):
        return Box.around(obj.center, obj.radius * 2, obj.radius * 2)
    if isinstance(obj, ir.Rectangle):
        return Box.around(obj.center, obj.width, obj.height)
    if isinstance(obj, (ir.Line, ir.Arrow)):
        return Box.containing((obj.start, obj.end))
    if isinstance(obj, ir.Axes):
        return Box.around(obj.position, AXES_WIDTH, AXES_HEIGHT)
    if isinstance(obj, ir.Plot):
        # A plot is drawn on its axes and bounded by them.
        axes = (declared or {}).get(obj.axes)
        if not isinstance(axes, ir.Axes):
            return None  # reference-integrity owns this
        return bounding_box(axes)
    return None


def _boxes(scene: ir.Scene) -> dict[str, Box]:
    declared = {obj.id: obj for obj in scene.objects}
    placed = {}
    for obj in scene.objects:
        box = bounding_box(obj, declared)
        if box is not None:
            placed[obj.id] = box
    return placed


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def in_frame(scene: ir.Scene, where: str) -> Iterable[Issue]:
    boxes = _boxes(scene)
    reported: set[str] = set()

    for index, obj in enumerate(scene.objects):
        box = boxes.get(obj.id)
        if box is not None and not box.fits(MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM):
            reported.add(obj.id)
            yield Issue(
                "in-frame",
                f"{obj.id!r} extends to x {box.min_x:.2f}..{box.max_x:.2f}, "
                f"y {box.min_y:.2f}..{box.max_y:.2f}, "
                + box.outside(MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM),
                f"{where}.objects[{index}]",
                Severity.WARNING,
            )

    # A move can push something off the edge that was declared inside it.
    for index, step in enumerate(scene.steps):
        if step.action is not ir.Action.MOVE or step.to is None:
            continue
        box = boxes.get(step.target)
        if box is None or step.target in reported:
            continue
        moved = box.moved_to(step.to)
        if not moved.fits(MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM):
            reported.add(step.target)
            yield Issue(
                "in-frame",
                f"moving {step.target!r} to {step.to} puts it "
                + moved.outside(MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM),
                f"{where}.steps[{index}]",
                Severity.WARNING,
            )


def no_overlap(scene: ir.Scene, where: str) -> Iterable[Issue]:
    declared = {obj.id: obj for obj in scene.objects}
    boxes = {
        obj_id: box
        for obj_id, box in _boxes(scene).items()
        # Text against text, and nothing else. The rule exists because
        # overlapping text is the commonest way a generated scene looks
        # broken, and every other pairing turned out to be either normal or
        # the content itself: an axes is a backdrop drawn over by design, a
        # label sits on the shape it labels, an arrow points *at* something,
        # and in geometry two overlapping shapes are usually the argument --
        # the squares in a Pythagorean figure sit on the triangle's sides.
        # Measured on a real layout, every one of the five warnings raised was
        # of that kind, and a false warning is not free: warnings drive
        # simplify_ir, so it tears apart a scene that was right.
        if isinstance(declared[obj_id], TEXTUAL)
    }
    reported: set[tuple[str, str]] = set()

    for moment in walk(scene):
        step = moment.step
        if step.action is ir.Action.MOVE and step.to is not None and step.target in boxes:
            boxes[step.target] = boxes[step.target].moved_to(step.to)

        present = sorted(i for i in moment.visible_after if i in boxes)
        for first, second in itertools.combinations(present, 2):
            if (first, second) in reported:
                continue
            if boxes[first].intersects(boxes[second]):
                reported.add((first, second))
                yield Issue(
                    "no-overlap",
                    f"{first!r} and {second!r} are on screen together and their "
                    "estimated bounding boxes overlap",
                    f"{where}.steps[{moment.index}]",
                    Severity.WARNING,
                )


def pacing(scene: ir.Scene, where: str) -> Iterable[Issue]:
    animated = sum(step.duration for step in scene.steps)
    if animated <= 0:
        return  # non-empty and positive-duration own that
    spoken = len(scene.narration.split()) / WORDS_PER_MINUTE * 60.0
    if abs(spoken - animated) > PACING_TOLERANCE * animated:
        yield Issue(
            "pacing",
            f"narration reads in about {spoken:.1f}s but the animation runs "
            f"{animated:.1f}s",
            where,
            Severity.WARNING,
        )


def scene_length(scene: ir.Scene, where: str) -> Iterable[Issue]:
    if not scene.steps:
        return  # non-empty owns that
    total = sum(step.duration for step in scene.steps)
    if total < MIN_SCENE_SECONDS:
        yield Issue(
            "scene-length",
            f"the scene runs {total:.1f}s, under the {MIN_SCENE_SECONDS:.0f}s "
            "below which it reads as a glitch",
            where,
            Severity.WARNING,
        )
    elif total > MAX_SCENE_SECONDS:
        yield Issue(
            "scene-length",
            f"the scene runs {total:.1f}s, over the {MAX_SCENE_SECONDS:.0f}s "
            "beyond which it loses the viewer",
            where,
            Severity.WARNING,
        )


DocumentRule = Callable[[ir.Document], Iterable[Issue]]
SceneRule = Callable[[ir.Scene, str], Iterable[Issue]]

DOCUMENT_RULES: tuple[DocumentRule, ...] = (
    document_non_empty,
    unique_scene_ids,
    safe_scene_ids,
)

SCENE_RULES: tuple[SceneRule, ...] = (
    non_empty,
    unique_ids,
    safe_ids,
    reference_integrity,
    positive_duration,
    action_applies,
    timeline,
    in_frame,
    no_overlap,
    pacing,
    scene_length,
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
