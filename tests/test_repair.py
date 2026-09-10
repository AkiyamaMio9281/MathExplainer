"""Simplification: what each reduction does, and what every rung guarantees.

The claim worth testing hardest is not any individual reduction. It is that
**every rung hands back either a scene with no rule errors or nothing at all**
-- stronger than SCENE_IR.md's promise about the last reduction, and load
bearing for the same reason the tier ladder is: a rung is followed by a codegen
call and a render, so one that returns an IR the rules would reject has spent a
model call learning what the rules knew for free.

`test_every_rung_of_every_broken_scene_is_clean` asserts it over a battery of
scenes broken in different ways, at every rung, which is the closest thing to a
proof available without generating scenes at random.
"""

from __future__ import annotations

import pytest

from explainer import ir, ir_rules, repair


def text(id="t", position=(0.0, 0.0), content="x"):
    return ir.Text(id=id, content=content, position=position)


def circle(id="c"):
    return ir.Circle(id=id, center=(0.0, 0.0), radius=1.0)


def axes(id="ax"):
    return ir.Axes(id=id, x_range=(-1.0, 1.0, 1.0), y_range=(-1.0, 1.0, 1.0))


def plot(id="p", axes_id="ax"):
    return ir.Plot(id=id, axes=axes_id, expression="x")


def step(action, target=None, duration=1.0, **extra):
    return ir.Step(action=ir.Action(action), duration=duration, target=target, **extra)


def scene(objects=(), steps=(), id="s", narration="word " * 20):
    return ir.Scene(
        id=id, narration=narration, objects=tuple(objects), steps=tuple(steps)
    )


def errors_of(subject: ir.Scene):
    return ir_rules.errors(ir_rules.check_scene(subject))


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------

BROKEN = {
    "a plot whose axes went missing": scene(
        [text(), plot(axes_id="gone")], [step("write", "t"), step("create", "p")]
    ),
    "acting on something after it faded out": scene(
        [text()],
        [step("write", "t"), step("fade_out", "t"), step("indicate", "t")],
    ),
    "acting on something never introduced": scene(
        [text(), circle()], [step("indicate", "t"), step("move", "c", to=(1.0, 1.0))]
    ),
    "write on a shape and create on text": scene(
        [text(), circle()], [step("write", "c"), step("create", "t")]
    ),
    "a transform into an object that is not declared": scene(
        [text()], [step("write", "t"), step("transform", "t", into="ghost")]
    ),
    "steps pointing at nothing at all": scene(
        [text()], [step("write", "ghost"), step("indicate", "nobody")]
    ),
    "durations of zero and below": scene(
        [text()], [step("write", "t", duration=0.0), step("wait", duration=-3.0)]
    ),
    "duplicate and unusable ids": scene(
        [text(id="t"), text(id="t"), text(id="Bad"), text(id="../x")],
        [step("write", "t")],
    ),
    "eight objects and a tangle": scene(
        [text(id=f"t{n}") for n in range(8)],
        [step("write", f"t{n}") for n in range(8)]
        + [step("transform", "t0", into="t1"), step("indicate", "t0")],
    ),
    "no steps at all": scene([text(), circle()], []),
}


@pytest.mark.parametrize("subject", BROKEN.values(), ids=list(BROKEN))
def test_every_rung_of_every_broken_scene_is_clean(subject):
    # Walk the whole ladder, asserting the invariant at each rung rather than
    # only at the end.
    stage = 0
    rungs = 0
    while True:
        result = repair.simplify(subject, from_stage=stage)
        if result.exhausted:
            break
        assert errors_of(result.scene) == (), (
            f"rung {result.stage} ({result.what}) left "
            f"{ir_rules.report(errors_of(result.scene))}"
        )
        subject, stage, rungs = result.scene, result.stage + 1, rungs + 1
        assert rungs <= len(repair.STAGES), "the ladder did not terminate"


def test_the_ladder_runs_out_rather_than_looping():
    subject = scene([text()], [step("write", "t")])
    result = repair.simplify(subject, from_stage=len(repair.STAGES))

    assert result.exhausted
    assert "as simple as it goes" in result.what


def test_a_scene_with_nothing_usable_cannot_be_saved():
    # Every id is unusable, so there is nothing that could be drawn.
    result = repair.simplify(scene([text(id="Bad"), text(id="../x")], []))

    assert result.exhausted
    assert "nothing renderable" in result.what


# ---------------------------------------------------------------------------
# The reductions, one at a time
# ---------------------------------------------------------------------------


def test_reduction_one_drops_graphs_and_the_steps_that_used_them():
    subject = scene(
        [text(), axes(), plot()],
        [step("write", "t"), step("create", "ax"), step("create", "p")],
    )
    result = repair.simplify(subject)

    assert result.stage == 0
    assert [o.id for o in result.scene.objects] == ["t"]
    assert [s.target for s in result.scene.steps] == ["t"]


def test_reduction_two_turns_a_transform_into_a_fade_out_and_a_fade_in():
    subject = scene(
        [text("a"), text("b")],
        [step("write", "a"), step("transform", "a", into="b", duration=2.0)],
    )
    result = repair.simplify(subject)

    assert result.stage == 1
    actions = [s.action for s in result.scene.steps]
    assert actions == [ir.Action.WRITE, ir.Action.FADE_OUT, ir.Action.FADE_IN]
    # The beat keeps its length, split across the two halves.
    assert sum(s.duration for s in result.scene.steps[1:]) == pytest.approx(2.0)


def test_reduction_three_keeps_the_objects_the_timeline_spends_most_on():
    subject = scene(
        [text(f"t{n}") for n in range(5)],
        [step("write", f"t{n}") for n in range(5)]
        + [step("indicate", "t3"), step("indicate", "t3"), step("indicate", "t1")],
    )
    result = repair.simplify(subject, from_stage=2)

    assert result.stage == 2
    kept = {o.id for o in result.scene.objects}
    assert len(kept) == repair.KEEP_OBJECTS
    assert {"t3", "t1"} <= kept


def test_reduction_four_is_one_introduction_each_and_a_wait():
    subject = scene(
        [text("a"), circle("b")],
        [step("write", "a"), step("indicate", "a"), step("create", "b")],
    )
    result = repair.simplify(subject, from_stage=3)

    assert result.stage == 3
    actions = [(s.action, s.target) for s in result.scene.steps]
    assert actions == [
        (ir.Action.WRITE, "a"),
        (ir.Action.CREATE, "b"),
        (ir.Action.WAIT, None),
    ]


def test_flattening_times_itself_to_the_narration():
    subject = scene(
        [text("a", (0.0, 2.0)), text("b", (0.0, -2.0))], [], narration="word " * 45
    )  # 18s of narration
    flattened = repair.flatten(subject)

    total = sum(s.duration for s in flattened.steps)
    assert total == pytest.approx(18.0, abs=18.0 * ir_rules.PACING_TOLERANCE)
    assert ir_rules.check_scene(flattened) == ()


def test_flattening_does_not_move_anything():
    # Simplification reduces what a scene asks for, not where it puts it. Two
    # labels on top of each other stay on top of each other, and the warning
    # survives -- repositioning is not one of the four reductions.
    subject = scene([text("a"), text("b")], [], narration="word " * 45)
    flattened = repair.flatten(subject)

    assert errors_of(flattened) == ()
    assert {i.rule for i in ir_rules.check_scene(flattened)} == {"no-overlap"}


def test_flattening_is_idempotent():
    # A scene already in the flat form is not rebuilt: doing so would read as
    # a reduction and cost a rung without asking for anything less.
    once = repair.flatten(scene([text("a"), circle("b")], [], narration="word " * 20))
    assert repair.flatten(once) == once


def test_flattening_a_silent_scene_still_clears_the_minimum_length():
    flattened = repair.flatten(scene([text()], [], narration=""))

    assert sum(s.duration for s in flattened.steps) >= ir_rules.MIN_SCENE_SECONDS


# ---------------------------------------------------------------------------
# Rungs that change nothing are not spent
# ---------------------------------------------------------------------------


def test_a_reduction_that_changes_nothing_is_skipped():
    # No plots and no transforms, so the first rung that does anything is the
    # third. Spending two codegen calls to discover that is the thing being
    # avoided.
    subject = scene(
        [text(f"t{n}") for n in range(6)], [step("write", f"t{n}") for n in range(6)]
    )
    result = repair.simplify(subject)

    assert result.stage == 2
    assert len(result.scene.objects) == repair.KEEP_OBJECTS


def test_a_scene_that_is_already_minimal_exhausts_immediately():
    subject = scene([text()], [step("write", "t", duration=8.0)])
    result = repair.simplify(subject)

    assert result.exhausted


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_a_wrong_verb_is_corrected_rather_than_the_step_dropped():
    # The step is what the scene wanted; only the verb was wrong.
    normalised = repair.normalise(
        scene([circle()], [step("write", "c", duration=8.0)])
    )

    assert [s.action for s in normalised.steps] == [ir.Action.CREATE]


def test_a_step_that_could_not_play_is_dropped():
    normalised = repair.normalise(
        scene([text()], [step("indicate", "t"), step("write", "t", duration=8.0)])
    )

    assert [s.action for s in normalised.steps] == [ir.Action.WRITE]


def test_an_unusable_id_takes_its_object_with_it():
    normalised = repair.normalise(
        scene([text(id="good"), text(id="Bad"), text(id="good")], [step("write", "good")])
    )

    assert [o.id for o in normalised.objects] == ["good"]


def test_normalising_a_scene_with_no_steps_left_rebuilds_them():
    # Everything was dropped, but there are still objects, so the scene is
    # rebuilt rather than lost.
    normalised = repair.normalise(scene([text(), circle()], [step("indicate", "t")]))

    assert len(normalised.steps) == 3
    assert ir_rules.check_scene(normalised) == ()


def test_repair_code_is_reachable_from_here():
    # The other rung. It lives in codegen because it reuses the same cached
    # preamble and extractor.
    from explainer import codegen

    assert repair.repair_code is codegen.repair
