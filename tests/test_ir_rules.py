"""One case per rule, each violating exactly that rule and nothing else.

"Exactly that rule" is the part worth insisting on. A fixture that trips three
rules at once cannot tell you which one caught the defect, and the metric in
ARCHITECTURE.md -- which IR rule caught which defect -- is only meaningful if
each rule is known to fire on its own. So every case below asserts the *set*
of rule names emitted, not merely that the expected one is among them.

`test_every_rule_has_a_case` closes the loop: it fails if a rule is added to
RULE_NAMES without a case here, so the coverage claim cannot rot.

Documents are built from the dataclasses rather than parsed from JSON. These
rules take a typed document as their input, and going through the parser would
make a rules test fail for parser reasons.
"""

from __future__ import annotations

import pytest

from explainer import ir, ir_rules


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


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


def scene(objects=(), steps=(), id="s"):
    return ir.Scene(id=id, objects=tuple(objects), steps=tuple(steps))


def doc(*scenes):
    return ir.Document(scenes=tuple(scenes))


def names(issues):
    return {issue.rule for issue in issues}


# ---------------------------------------------------------------------------
# One document per rule
# ---------------------------------------------------------------------------

VIOLATIONS = (
    ("non-empty", "a document with no scenes", doc()),
    ("non-empty", "a scene with no objects and no steps", doc(scene())),
    (
        "unique-ids",
        "two objects sharing an id",
        doc(scene([text("t"), text("t")], [step("write", "t")])),
    ),
    (
        "unique-ids",
        "two scenes sharing an id",
        doc(
            scene([text()], [step("write", "t")], id="same"),
            scene([text()], [step("write", "t")], id="same"),
        ),
    ),
    (
        "reference-integrity",
        "a step targeting an object that was never declared",
        doc(scene([text()], [step("write", "t"), step("indicate", "ghost")])),
    ),
    (
        "reference-integrity",
        "a transform into an object that was never declared",
        doc(scene([text()], [step("write", "t"), step("transform", "t", into="ghost")])),
    ),
    (
        "reference-integrity",
        "a plot whose axes id is not declared",
        doc(scene([plot(axes_id="nope")], [step("create", "p")])),
    ),
    (
        "reference-integrity",
        "a plot whose axes id names something that is not an axes",
        doc(scene([text(), plot(axes_id="t")], [step("write", "t"), step("create", "p")])),
    ),
    (
        "introduced-before-use",
        "indicating an object before anything introduces it",
        doc(scene([text()], [step("indicate", "t"), step("write", "t")])),
    ),
    (
        "introduced-before-use",
        "fading out an object that was never on screen",
        doc(scene([text()], [step("fade_out", "t")])),
    ),
    (
        "not-after-removal",
        "acting on an object after fade_out",
        doc(
            scene(
                [text()],
                [step("write", "t"), step("fade_out", "t"), step("indicate", "t")],
            )
        ),
    ),
    (
        "not-after-removal",
        "acting on an object after it was transformed away",
        doc(
            scene(
                [text("a"), text("b")],
                [
                    step("write", "a"),
                    step("transform", "a", into="b"),
                    step("indicate", "a"),
                ],
            )
        ),
    ),
    (
        "positive-duration",
        "a step with a duration of zero",
        doc(scene([text()], [step("write", "t", duration=0.0)])),
    ),
    (
        "positive-duration",
        "a step with a negative duration",
        doc(scene([text()], [step("write", "t", duration=-1.0)])),
    ),
    (
        "action-applies",
        "writing a shape",
        doc(scene([circle()], [step("write", "c")])),
    ),
    (
        "action-applies",
        "creating text",
        doc(scene([text()], [step("create", "t")])),
    ),
)


@pytest.mark.parametrize(
    "rule,document",
    [(rule, document) for rule, _, document in VIOLATIONS],
    ids=[description for _, description, _ in VIOLATIONS],
)
def test_each_case_trips_exactly_its_own_rule(rule, document):
    issues = ir_rules.check_document(document)
    assert issues, "the case did not trip any rule"
    assert names(issues) == {rule}


def test_every_rule_has_a_case():
    # Two halves, and both are needed. The first fails when a rule is added to
    # RULE_NAMES without a case here. The second runs the cases through the
    # real entry point, so it also fails when a rule is written and declared
    # but never registered in SCENE_RULES -- which a comparison of two
    # constants cannot see.
    assert {rule for rule, _, _ in VIOLATIONS} == ir_rules.RULE_NAMES

    emitted = set()
    for _, _, document in VIOLATIONS:
        emitted |= names(ir_rules.check_document(document))
    assert emitted == ir_rules.RULE_NAMES


# ---------------------------------------------------------------------------
# Valid documents produce nothing
# ---------------------------------------------------------------------------


def test_a_document_using_the_whole_vocabulary_is_clean():
    document = doc(
        scene(
            [text("title"), circle("c"), axes("ax"), plot("p", "ax"), text("other")],
            [
                step("write", "title"),
                step("create", "c", duration=2.0),
                step("create", "ax"),
                step("create", "p"),
                step("fade_in", "other"),
                step("move", "c", to=(1.0, 1.0)),
                step("indicate", "title"),
                step("transform", "title", into="other"),
                step("fade_out", "c"),
                step("wait"),
            ],
        )
    )
    assert ir_rules.check_document(document) == ()


def test_the_example_in_the_specification_is_clean():
    # docs/SCENE_IR.md shows this document; if the rules reject it, one of the
    # two is wrong and it matters which.
    result = ir.parse_document(
        {
            "version": "1",
            "title": "The Pythagorean Theorem",
            "scenes": [
                {
                    "id": "intro",
                    "narration": "Every right triangle hides one simple relationship "
                    "between its sides.",
                    "objects": [
                        {
                            "id": "title",
                            "type": "text",
                            "content": "The Pythagorean Theorem",
                            "position": [0, 3],
                            "font_size": 48,
                        },
                        {
                            "id": "tri",
                            "type": "polygon",
                            "points": [[-3, -1], [0, -1], [0, 1.5]],
                            "color": "BLUE",
                        },
                        {
                            "id": "eq",
                            "type": "mathtex",
                            "content": "a^2 + b^2 = c^2",
                            "position": [3.2, 0],
                            "font_size": 56,
                        },
                    ],
                    "steps": [
                        {"action": "write", "target": "title", "duration": 1.5},
                        {"action": "create", "target": "tri", "duration": 2.0},
                        {"action": "write", "target": "eq", "duration": 1.5},
                        {"action": "indicate", "target": "eq", "duration": 1.0},
                        {"action": "wait", "duration": 1.0},
                    ],
                }
            ],
        }
    )
    assert result.ok, result.report()
    assert ir_rules.check_document(result.document) == ()


# ---------------------------------------------------------------------------
# The timeline walk
# ---------------------------------------------------------------------------


def test_transform_introduces_its_target_without_a_separate_introduction():
    # `into` is put on screen by the transform, so acting on it afterwards is
    # not introduced-before-use.
    document = doc(
        scene(
            [text("a"), text("b")],
            [step("write", "a"), step("transform", "a", into="b"), step("indicate", "b")],
        )
    )
    assert ir_rules.check_document(document) == ()


def test_reintroducing_an_object_clears_its_removal():
    # Asserted on the walk rather than through the rules, because the
    # report-once guard makes this invisible in the issue list: without the
    # clear, step 3 would still be judged against a screen that has removed
    # the object it is currently showing. Moment is published for commit 16's
    # layout checks, so it has to be faithful even where no rule reads it.
    subject = scene(
        [text()],
        [
            step("write", "t"),
            step("fade_out", "t"),
            step("write", "t"),
            step("indicate", "t"),
        ],
    )
    removed = [m.removed_before for m in ir_rules.walk(subject)]

    assert removed == [
        frozenset(),
        frozenset(),
        frozenset({"t"}),
        frozenset(),  # the re-introduction put it back
    ]


def test_the_walk_puts_a_reintroduced_object_back_on_screen():
    # Re-introducing after fade_out is reported once, and the walk still shows
    # the object again -- so the indicate that follows is judged against a
    # screen that has it, and does not also fail.
    document = doc(
        scene(
            [text()],
            [
                step("write", "t"),
                step("fade_out", "t"),
                step("write", "t"),
                step("indicate", "t"),
            ],
        )
    )
    issues = ir_rules.check_document(document)

    assert names(issues) == {"not-after-removal"}
    assert len(issues) == 1


def test_not_after_removal_reports_once_however_many_steps_follow():
    document = doc(
        scene(
            [text()],
            [
                step("write", "t"),
                step("fade_out", "t"),
                step("indicate", "t"),
                step("indicate", "t"),
                step("move", "t", to=(1.0, 1.0)),
            ],
        )
    )
    issues = ir_rules.check_document(document)

    assert len(issues) == 1
    assert issues[0].rule == "not-after-removal"


def test_an_introduction_is_on_screen_during_its_own_step():
    subject = scene([text("a"), text("b")], [step("write", "a"), step("write", "b")])
    moments = [(m.index, visible) for m, visible in ir_rules.visible_at(subject)]

    assert moments == [(0, frozenset({"a"})), (1, frozenset({"a", "b"}))]


def test_the_walk_tracks_removal_across_steps():
    subject = scene(
        [text("a"), text("b")],
        [step("write", "a"), step("write", "b"), step("fade_out", "a"), step("wait")],
    )
    states = [m.visible_after for m in ir_rules.walk(subject)]

    assert states == [
        frozenset({"a"}),
        frozenset({"a", "b"}),
        frozenset({"b"}),
        frozenset({"b"}),
    ]


# ---------------------------------------------------------------------------
# Issues are located and reportable
# ---------------------------------------------------------------------------


def test_an_issue_points_at_the_step_that_caused_it():
    document = doc(
        scene([text()], [step("write", "t"), step("wait"), step("indicate", "ghost")])
    )
    (issue,) = ir_rules.check_document(document)

    assert issue.where == "scenes[0].steps[2]"
    assert "ghost" in issue.message


def test_a_rule_does_not_repeat_what_another_rule_owns():
    # An undeclared target is reference-integrity's business alone: the
    # timeline and action-applies both skip it rather than piling on.
    document = doc(scene([text()], [step("write", "t"), step("write", "ghost")]))
    issues = ir_rules.check_document(document)

    assert len(issues) == 1
    assert issues[0].rule == "reference-integrity"


def test_report_renders_every_issue_on_its_own_line():
    document = doc(
        scene([text()], [step("write", "t", duration=0.0), step("indicate", "ghost")])
    )
    lines = ir_rules.report(ir_rules.check_document(document)).splitlines()

    assert len(lines) == 2
    assert all("scenes[0].steps[" in line for line in lines)
