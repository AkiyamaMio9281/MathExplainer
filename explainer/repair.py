"""The rung between repairing code and dropping a scene.

When a scene will not compile, the response depends on how many times it has
already failed, and each rung changes a *different* layer. Repairing the code
asks the model to fix the file it wrote. Simplifying the IR asks the scene for
less and regenerates from there. Dropping it records why and carries on.

`repair_code` is the other rung and lives in `codegen.py`, because it reuses
the same cached preamble and the same fenced-block extractor; it is re-exported
here so the two rungs can be reached from one place. This module is
`simplify_ir`.

## What simplification promises

SCENE_IR.md says the last reduction "always produces something renderable, so
the ladder terminates". That is the important claim, and this module makes a
stronger one: **every rung returns either a scene with no rule errors at all,
or None.** Not just the last one.

The reason is cost. Each rung is followed by a codegen call and a validation,
so a rung that hands back an IR the rules would reject has spent a model call
to learn something the rules knew for free. Every reduction therefore runs
through `normalise`, which fixes what can be fixed and drops what cannot:
actions that do not apply to their target are corrected rather than removed,
steps acting on something not on screen are dropped, ids that could not be a
directory or a variable take their objects with them.

`None` means the scene cannot be saved -- there is nothing left that could be
drawn -- and the caller should drop it and say so.

## Order

The four reductions are the ones SCENE_IR.md specifies, in its order:

1. Drop `plot` and `axes`, the most failure-prone types.
2. Unwind `transform` into a `fade_out` and a `fade_in`.
3. Keep at most three objects, the ones the timeline uses most.
4. Flatten to introductions and a wait.

A reduction that changes nothing is skipped rather than counted. A scene with
no plots should not spend a rung, a codegen call and a render discovering
that.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from . import codegen, ir, ir_rules, llm
from .codegen import repair as repair_code  # noqa: F401  -- the other rung

#: At most this many objects survive reduction 3.
KEEP_OBJECTS = 3

#: No step is allowed to be shorter than this once durations are repaired.
MIN_STEP = 0.4

#: Bounds for the introductions a flattened scene is built from.
MIN_INTRO = 0.6
MAX_INTRO = 2.0


@dataclass(frozen=True)
class Simplification:
    """One rung: what came back, and which reduction produced it."""

    scene: ir.Scene | None
    stage: int
    what: str

    @property
    def exhausted(self) -> bool:
        return self.scene is None


# ---------------------------------------------------------------------------
# Normalisation: what every reduction runs through
# ---------------------------------------------------------------------------


def usable_objects(scene: ir.Scene) -> tuple[ir.SceneObject, ...]:
    """The objects that could survive to a rendered file.

    An id that is not a lowercase identifier cannot be a directory or a Python
    local, and a duplicate cannot be told from the object it collides with, so
    neither is worth carrying further. First declaration wins.
    """
    kept: list[ir.SceneObject] = []
    seen: set[str] = set()
    for obj in scene.objects:
        if obj.id in seen or not ir.is_safe_id(obj.id):
            continue
        seen.add(obj.id)
        kept.append(obj)
    return tuple(kept)


def fix_actions(scene: ir.Scene) -> ir.Scene:
    """Correct `write` on a shape and `create` on text rather than drop them.

    The step is what the scene wanted; only the verb was wrong.
    """
    declared = {obj.id: obj for obj in scene.objects}
    steps = []
    for step in scene.steps:
        obj = declared.get(step.target) if step.target else None
        if obj is not None:
            textual = isinstance(obj, ir_rules.TEXTUAL)
            if step.action is ir.Action.WRITE and not textual:
                step = replace(step, action=ir.Action.CREATE)
            elif step.action is ir.Action.CREATE and textual:
                step = replace(step, action=ir.Action.WRITE)
        steps.append(step)
    return replace(scene, steps=tuple(steps))


def prune_timeline(scene: ir.Scene) -> ir.Scene:
    """Drop steps that could not play: the timeline, enforced by removal.

    Dropping an object leaves steps pointing at nothing, and dropping the
    `transform` that introduced an object leaves everything after it acting on
    something never shown. Walking the timeline and removing what cannot play
    is what keeps a reduction from trading one rule failure for another.
    """
    declared = {obj.id for obj in scene.objects}
    visible: set[str] = set()
    removed: set[str] = set()
    kept: list[ir.Step] = []

    for step in scene.steps:
        if step.action is ir.Action.WAIT:
            kept.append(step)
            continue

        target = step.target
        if not target or target not in declared or target in removed:
            continue

        if step.action is ir.Action.TRANSFORM:
            into = step.into
            if (
                not into
                or into not in declared
                or into in removed
                or into in visible
                or target not in visible
            ):
                continue
            visible.discard(target)
            removed.add(target)
            visible.add(into)
        elif step.action is ir.Action.FADE_OUT:
            if target not in visible:
                continue
            visible.discard(target)
            removed.add(target)
        elif step.action in ir.INTRODUCTIONS:
            if target in visible:
                continue  # already on screen; a second introduction is a no-op
            visible.add(target)
        else:  # move, indicate
            if target not in visible:
                continue

        kept.append(step)

    return replace(scene, steps=tuple(kept))


def clamp_durations(scene: ir.Scene) -> ir.Scene:
    return replace(
        scene,
        steps=tuple(
            replace(s, duration=max(s.duration, MIN_STEP)) for s in scene.steps
        ),
    )


def normalise(scene: ir.Scene) -> ir.Scene | None:
    """Make *scene* satisfy the rules, or report that it cannot."""
    objects = usable_objects(scene)
    if not objects:
        return None

    current = replace(scene, objects=objects)
    current = clamp_durations(prune_timeline(fix_actions(current)))
    if not current.steps:
        # Everything was dropped. There are still objects, so the scene can be
        # rebuilt as introductions rather than lost.
        current = _introduce_everything(current)
    return current


# ---------------------------------------------------------------------------
# The four reductions
# ---------------------------------------------------------------------------


def _without(scene: ir.Scene, dropped: set[str]) -> ir.Scene:
    if not dropped:
        return scene
    return replace(
        scene,
        objects=tuple(o for o in scene.objects if o.id not in dropped),
        steps=tuple(
            s
            for s in scene.steps
            if (s.target or "") not in dropped and (s.into or "") not in dropped
        ),
    )


def drop_plots(scene: ir.Scene) -> ir.Scene:
    """Reduction 1: lose the graphs.

    Named first by SCENE_IR.md as the most failure-prone type, and the layout
    stage already declines to generate them.
    """
    return _without(
        scene, {o.id for o in scene.objects if isinstance(o, (ir.Axes, ir.Plot))}
    )


def unwind_transforms(scene: ir.Scene) -> ir.Scene:
    """Reduction 2: a transform becomes a fade out and a fade in.

    The same beat, in two animations that cannot fail together.
    """
    steps: list[ir.Step] = []
    for step in scene.steps:
        if step.action is ir.Action.TRANSFORM and step.into:
            half = max(step.duration / 2, MIN_STEP)
            steps.append(
                replace(step, action=ir.Action.FADE_OUT, duration=half, into=None)
            )
            steps.append(
                ir.Step(action=ir.Action.FADE_IN, target=step.into, duration=half)
            )
        else:
            steps.append(step)
    return replace(scene, steps=tuple(steps))


def keep_busiest(scene: ir.Scene, limit: int = KEEP_OBJECTS) -> ir.Scene:
    """Reduction 3: keep the objects the timeline spends the most on."""
    if len(scene.objects) <= limit:
        return scene

    uses: Counter[str] = Counter()
    for step in scene.steps:
        for reference in (step.target, step.into):
            if reference:
                uses[reference] += 1

    order = sorted(
        scene.objects, key=lambda o: (-uses[o.id], scene.objects.index(o))
    )
    keep = {o.id for o in order[:limit]}
    return _without(scene, {o.id for o in scene.objects} - keep)


def _introduce_everything(scene: ir.Scene) -> ir.Scene:
    """One introduction per object, then a wait, timed to the narration.

    The reduction that always works: no ordering to get wrong, no object
    referred to after it leaves, nothing to transform into anything.
    """
    objects = scene.objects
    target = max(
        len(scene.narration.split()) / ir_rules.WORDS_PER_MINUTE * 60.0,
        ir_rules.MIN_SCENE_SECONDS,
    )
    intro = min(max(target / (len(objects) + 1), MIN_INTRO), MAX_INTRO)

    steps = [
        ir.Step(
            action=(
                ir.Action.WRITE
                if isinstance(obj, ir_rules.TEXTUAL)
                else ir.Action.CREATE
            ),
            target=obj.id,
            duration=round(intro, 2),
        )
        for obj in objects
    ]
    steps.append(
        ir.Step(
            action=ir.Action.WAIT,
            duration=round(max(target - intro * len(objects), MIN_STEP), 2),
        )
    )
    return replace(scene, steps=tuple(steps))


def is_flat(scene: ir.Scene) -> bool:
    """Whether *scene* is already one introduction per object, and waits."""
    introductions = [s for s in scene.steps if s.action is not ir.Action.WAIT]
    return (
        all(s.action in ir.INTRODUCTIONS for s in introductions)
        and {s.target for s in introductions} == {o.id for o in scene.objects}
        and len(introductions) == len(scene.objects)
    )


def flatten(scene: ir.Scene) -> ir.Scene:
    """Reduction 4, and the reason the ladder terminates.

    Idempotent on purpose. A scene that is already introductions and a wait
    would otherwise be rebuilt into an equivalent one with different
    durations, which reads as a reduction and costs a rung, a codegen call and
    a render without asking for anything less.
    """
    if is_flat(scene):
        return scene
    return _introduce_everything(scene)


STAGES: tuple[tuple[str, Callable[[ir.Scene], ir.Scene]], ...] = (
    ("drop plots and axes", drop_plots),
    ("unwind transforms", unwind_transforms),
    (f"keep the {KEEP_OBJECTS} busiest objects", keep_busiest),
    ("flatten to introductions", flatten),
)


def simplify(scene: ir.Scene, from_stage: int = 0) -> Simplification:
    """The next reduction that actually changes something.

    A reduction that is a no-op on this scene is skipped rather than counted.
    A scene with no plots should not spend a rung, a codegen call and a render
    finding that out.
    """
    current = normalise(scene)
    if current is None:
        return Simplification(None, from_stage, "nothing renderable to reduce")

    for index in range(max(from_stage, 0), len(STAGES)):
        what, reduce = STAGES[index]
        reduced = normalise(reduce(current))
        if reduced is None:
            return Simplification(None, index, f"{what}: nothing renderable left")
        if reduced != current:
            return Simplification(reduced, index, what)

    return Simplification(None, len(STAGES), "already as simple as it goes")


# ---------------------------------------------------------------------------
# The escalation ladder
# ---------------------------------------------------------------------------

#: Repairs allowed per version of the IR, feeding the traceback back each time.
REPAIR_ROUNDS = 3

#: Simplifications allowed per scene. Each one starts a fresh cycle, so the
#: worst case is (1 + SIMPLIFY_ROUNDS) * (1 + REPAIR_ROUNDS) model calls --
#: twelve at the defaults. That number is why the agent carries a budget on top
#: of these bounds rather than trusting them alone.
SIMPLIFY_ROUNDS = 2

RENDERED = "rendered"
DROPPED = "dropped"

CODEGEN = "codegen"
REPAIRED = "repair"
STOPPED = "stopped"
SIMPLIFIED = "simplify"
RENDER = "render"


@dataclass(frozen=True)
class Attempt:
    """One action and what came of it. The raw material for the metrics."""

    kind: str
    stage: int  # which simplification the scene is on; 0 is what was asked for
    round: int  # repair round within that stage
    ok: bool
    error: str = ""
    level: str = ""  # the tier that rejected it, when one did
    seconds: float = 0.0
    usage: llm.Usage = field(default_factory=llm.Usage)
    detail: str = ""


@dataclass(frozen=True)
class Outcome:
    scene_id: str
    status: str
    scene: ir.Scene
    attempts: tuple[Attempt, ...] = ()
    code: str = ""
    check: Any = None
    video: Any = None
    usage: llm.Usage = field(default_factory=llm.Usage)
    seconds: float = 0.0
    error: str = ""

    @property
    def rendered(self) -> bool:
        return self.status == RENDERED

    @property
    def repairs(self) -> int:
        return sum(1 for a in self.attempts if a.kind == REPAIRED)

    @property
    def simplifications(self) -> int:
        return sum(1 for a in self.attempts if a.kind == SIMPLIFIED and a.ok)

    @property
    def calls(self) -> int:
        """Model calls. A simplification is arithmetic, not a call."""
        return sum(1 for a in self.attempts if a.kind in (CODEGEN, REPAIRED))

    def trail(self) -> str:
        return " -> ".join(
            a.kind + ("" if a.ok else " x") + (f"({a.detail})" if a.detail else "")
            for a in self.attempts
        )


@dataclass(frozen=True)
class Tools:
    """The four actions the ladder drives.

    Injectable, so the control flow can be tested without a model, a
    subprocess or a renderer. The ordering and the bounds are the whole point
    of this module, and neither needs manim to be exercised.
    """

    generate: Callable[..., tuple[str, llm.Reply]]
    repair: Callable[..., tuple[str, llm.Reply]]
    validate: Callable[..., Any]
    render: Callable[..., Any]


def default_tools() -> Tools:
    from .validate import Level
    from .validate import render as render_file
    from .validate import validate as validate_code

    return Tools(
        generate=lambda client, scene, title: codegen.generate(
            client, scene, title=title
        ),
        repair=lambda client, code, error: codegen.repair(client, code, error),
        validate=lambda workdir, code: validate_code(workdir, code, up_to=Level.DRYRUN),
        render=lambda workdir, name, quality: render_file(
            workdir, name, quality=quality
        ),
    )


def climb(
    scene: ir.Scene,
    workdir: Any,
    client: Any,
    *,
    title: str = "",
    quality: str = "l",
    repair_rounds: int = REPAIR_ROUNDS,
    simplify_rounds: int = SIMPLIFY_ROUNDS,
    stop: Callable[[], str] | None = None,
    tools: Tools | None = None,
) -> Outcome:
    """Take one scene as far up the ladder as it needs, and no further.

    Generate, validate, and on failure repair the code with the traceback fed
    back verbatim. When the repairs run out, simplify the IR and start again
    from a scene that asks for less. When the simplifications run out, drop the
    scene and say why: a lesson missing one scene and carrying a note about it
    is a usable result, and an exception four minutes into a render is not.

    Every rung is bounded and every action is recorded. `Outcome.attempts` is
    the log the metrics are computed from, and it is kept whether the scene
    rendered or not -- a scene that took three repairs is as interesting as one
    that was dropped.

    *stop* is checked before every model call and aborts the scene when it
    returns a reason. Bounds alone are not a cap: twelve calls a scene is a
    known number only if the caller knows how many scenes there are, and the
    ladder does not. Without this, a budget checked between scenes can be
    overshot by a whole scene; with it, by at most one call.
    """
    tools = tools or default_tools()
    started = time.perf_counter()
    attempts: list[Attempt] = []
    usage = llm.Usage()
    current = scene
    stage = 0
    simplifications = 0

    def finish(status: str, *, error: str = "", code: str = "", check=None, video=None):
        return Outcome(
            scene_id=scene.id,
            status=status,
            scene=current,
            attempts=tuple(attempts),
            code=code,
            check=check,
            video=video,
            usage=usage,
            seconds=time.perf_counter() - started,
            error=error,
        )

    while True:
        code = ""
        check = None
        failure = ""

        for round_number in range(repair_rounds + 1):
            kind = CODEGEN if round_number == 0 else REPAIRED
            if stop and (reason := stop()):
                attempts.append(Attempt(STOPPED, stage, round_number, False, error=reason))
                return finish(DROPPED, error=reason, code=code)
            call_started = time.perf_counter()
            try:
                if round_number == 0:
                    code, reply = tools.generate(client, current, title)
                else:
                    code, reply = tools.repair(client, code, failure)
            except llm.LLMError as exc:
                # Truncation especially: a scene that asks for less produces a
                # shorter file, so simplifying is the useful answer rather than
                # asking the same question again.
                attempts.append(
                    Attempt(
                        kind,
                        stage,
                        round_number,
                        False,
                        error=f"{type(exc).__name__}: {exc}",
                        seconds=time.perf_counter() - call_started,
                    )
                )
                failure = str(exc)
                check = None
                break

            usage = usage + reply.usage
            check = tools.validate(workdir, code)
            failure = getattr(check, "error", "") or ""
            attempts.append(
                Attempt(
                    kind,
                    stage,
                    round_number,
                    bool(getattr(check, "ok", False)),
                    error=failure,
                    level=getattr(getattr(check, "level", None), "name", ""),
                    seconds=time.perf_counter() - call_started,
                    usage=reply.usage,
                )
            )
            if getattr(check, "ok", False):
                break

        if check is not None and getattr(check, "ok", False):
            produced = tools.render(
                workdir, getattr(check, "scene_name", "") or "", quality
            )
            if getattr(produced, "ok", False):
                return finish(
                    RENDERED,
                    code=code,
                    check=produced,
                    video=getattr(produced, "output", None),
                )
            # A scene that runs but will not rasterise is usually asking for
            # too much, and that is what simplification is for -- not another
            # repair of code which already executes.
            failure = getattr(produced, "error", "") or "the render failed"
            attempts.append(
                Attempt(RENDER, stage, 0, False, error=failure, level="RENDER")
            )

        if simplifications >= simplify_rounds:
            return finish(
                DROPPED,
                error=(
                    f"still failing after {simplifications} simplification(s) "
                    f"and {repair_rounds} repairs a time: {failure}"
                ),
                code=code,
            )

        reduced = simplify(current, from_stage=stage)
        attempts.append(
            Attempt(SIMPLIFIED, stage, 0, not reduced.exhausted, detail=reduced.what)
        )
        if reduced.exhausted:
            return finish(
                DROPPED,
                error=f"cannot be simplified further ({reduced.what}): {failure}",
                code=code,
            )
        current, stage = reduced.scene, reduced.stage + 1
        simplifications += 1
