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


def errors_of(document):
    """Only the errors.

    The error cases below are built for one rule each and pay no attention to
    layout or timing, so most of them also trip a warning. Filtering keeps each
    case about the rule it was written for; the warnings have their own table.
    """
    return ir_rules.errors(ir_rules.check_document(document))


def paced(seconds):
    """Narration that reads in about *seconds* at the assumed 150 wpm."""
    return " ".join(["word"] * round(seconds * ir_rules.WORDS_PER_MINUTE / 60))


def timed_scene(objects, steps, narration=None, id="s"):
    """A scene whose narration matches its animation, so pacing stays quiet."""
    total = sum(step.duration for step in steps)
    return ir.Scene(
        id=id,
        narration=paced(total) if narration is None else narration,
        objects=tuple(objects),
        steps=tuple(steps),
    )


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
        "safe-ids",
        "a scene id that would escape the working directory",
        doc(scene([text()], [step("write", "t")], id="../../elsewhere")),
    ),
    (
        "safe-ids",
        "an object id that is a Python keyword",
        doc(scene([text(id="class")], [step("write", "class")])),
    ),
    (
        "safe-ids",
        "an object id that shadows a Manim class",
        doc(scene([text(id="Text")], [step("write", "Text")])),
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
    issues = errors_of(document)
    assert issues, "the case did not trip any rule"
    assert names(issues) == {rule}


def test_every_error_rule_has_a_case():
    # Two halves, and both are needed. The first fails when a rule is added to
    # ERROR_RULES without a case here. The second runs the cases through the
    # real entry point, so it also fails when a rule is written and declared
    # but never registered in SCENE_RULES -- which a comparison of two
    # constants cannot see.
    assert {rule for rule, _, _ in VIOLATIONS} == ir_rules.ERROR_RULES

    emitted = set()
    for _, _, document in VIOLATIONS:
        emitted |= names(errors_of(document))
    assert emitted == ir_rules.ERROR_RULES


# ---------------------------------------------------------------------------
# One document per warning
# ---------------------------------------------------------------------------

WARNINGS = (
    (
        "in-frame",
        "an object declared past the margin",
        doc(timed_scene([text(position=(7.0, 0.0))], [step("write", "t", duration=6.0)])),
    ),
    (
        "in-frame",
        "a move that carries an object off the edge",
        doc(
            timed_scene(
                [text()],
                [step("write", "t", duration=3.0), step("move", "t", 3.0, to=(7.0, 0.0))],
            )
        ),
    ),
    (
        "no-overlap",
        "two labels on top of each other",
        doc(
            timed_scene(
                [text("a", (0.0, 0.0), "hello"), text("b", (0.1, 0.0), "hello")],
                [step("write", "a", duration=3.0), step("write", "b", duration=3.0)],
            )
        ),
    ),
    (
        "pacing",
        "narration far shorter than the animation",
        doc(
            timed_scene(
                [text()], [step("write", "t", duration=6.0)], narration="two words"
            )
        ),
    ),
    (
        "scene-length",
        "a scene too short to read as anything but a glitch",
        doc(timed_scene([text()], [step("write", "t", duration=1.0)])),
    ),
    (
        "scene-length",
        "a scene long enough to lose the viewer",
        doc(timed_scene([text()], [step("write", "t", duration=100.0)])),
    ),
)


@pytest.mark.parametrize(
    "rule,document",
    [(rule, document) for rule, _, document in WARNINGS],
    ids=[description for _, description, _ in WARNINGS],
)
def test_each_warning_case_trips_exactly_its_own_warning(rule, document):
    issues = ir_rules.check_document(document)
    assert ir_rules.errors(issues) == (), "a warning case tripped an error rule"
    assert names(issues) == {rule}


def test_every_warning_rule_has_a_case():
    assert {rule for rule, _, _ in WARNINGS} == ir_rules.WARNING_RULES

    emitted = set()
    for _, _, document in WARNINGS:
        emitted |= names(ir_rules.check_document(document))
    assert emitted == ir_rules.WARNING_RULES


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
    assert errors_of(document) == ()


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
    issues = ir_rules.check_document(result.document)
    assert ir_rules.errors(issues) == ()

    # The geometry warnings are the ones this example would have failed under
    # the constants SCENE_IR.md first carried: they made this very title 16.9
    # units wide against a 14.2-unit frame. Measured constants put it at 7.67,
    # against manim's 7.72.
    assert not [i for i in issues if i.rule in ("in-frame", "no-overlap")]


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
    assert errors_of(document) == ()


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
    issues = errors_of(document)

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
    issues = errors_of(document)

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
    (issue,) = errors_of(document)

    assert issue.where == "scenes[0].steps[2]"
    assert "ghost" in issue.message


def test_a_rule_does_not_repeat_what_another_rule_owns():
    # An undeclared target is reference-integrity's business alone: the
    # timeline and action-applies both skip it rather than piling on.
    document = doc(scene([text()], [step("write", "t"), step("write", "ghost")]))
    issues = errors_of(document)

    assert len(issues) == 1
    assert issues[0].rule == "reference-integrity"


def test_report_renders_every_issue_on_its_own_line():
    document = doc(
        scene([text()], [step("write", "t", duration=0.0), step("indicate", "ghost")])
    )
    lines = ir_rules.report(errors_of(document)).splitlines()

    assert len(lines) == 2
    assert all("scenes[0].steps[" in line for line in lines)


# ---------------------------------------------------------------------------
# Estimated geometry
# ---------------------------------------------------------------------------


def test_shapes_are_measured_rather_than_estimated():
    # Manim draws shapes in scene units one to one, so nothing here is an
    # approximation and the boxes are exact.
    assert ir_rules.bounding_box(ir.Circle(id="c", center=(1.0, 2.0), radius=1.5)) == (
        ir_rules.Box(-0.5, 0.5, 2.5, 3.5)
    )
    assert ir_rules.bounding_box(
        ir.Rectangle(id="r", center=(0.0, 0.0), width=3.0, height=2.0)
    ) == ir_rules.Box(-1.5, -1.0, 1.5, 1.0)
    assert ir_rules.bounding_box(
        ir.Polygon(id="p", points=((-3.0, -1.0), (0.0, -1.0), (0.0, 1.5)))
    ) == ir_rules.Box(-3.0, -1.0, 0.0, 1.5)
    assert ir_rules.bounding_box(
        ir.Line(id="l", start=(-1.0, -3.0), end=(1.0, -3.0))
    ) == ir_rules.Box(-1.0, -3.0, 1.0, -3.0)


def test_axes_are_sized_from_the_frame_not_from_their_range():
    # SCENE_IR.md derived the size from x_range/y_range; manim does not, and
    # uses the range for tick labels instead.
    narrow = ir.Axes(id="a", x_range=(-1.0, 1.0, 0.5), y_range=(-1.0, 1.0, 0.5))
    wide = ir.Axes(id="b", x_range=(-100.0, 100.0, 10.0), y_range=(-50.0, 50.0, 10.0))

    assert ir_rules.bounding_box(narrow) == ir_rules.bounding_box(wide)
    assert ir_rules.bounding_box(narrow) == ir_rules.Box(-6.0, -3.0, 6.0, 3.0)


def test_a_plot_is_bounded_by_the_axes_it_is_drawn_on():
    declared = {"ax": axes("ax"), "p": plot("p", "ax")}
    assert ir_rules.bounding_box(declared["p"], declared) == ir_rules.bounding_box(
        declared["ax"]
    )


def test_a_plot_whose_axes_cannot_be_resolved_has_no_box():
    # reference-integrity owns the complaint; the estimator just declines.
    assert ir_rules.bounding_box(plot("p", "missing"), {}) is None


def test_a_box_moves_by_its_centre():
    box = ir_rules.Box(-1.0, -1.0, 1.0, 1.0)
    moved = box.moved_to((5.0, 2.0))

    assert moved.centre == (5.0, 2.0)
    assert (moved.width, moved.height) == (box.width, box.height)


def test_boxes_that_only_touch_do_not_overlap():
    left = ir_rules.Box(0.0, 0.0, 1.0, 1.0)
    right = ir_rules.Box(1.0, 0.0, 2.0, 1.0)

    assert not left.intersects(right)
    assert left.intersects(ir_rules.Box(0.99, 0.0, 2.0, 1.0))


def test_axes_and_plots_are_left_out_of_the_overlap_check():
    # An axes is 12 by 6 -- most of the frame -- and objects are meant to be
    # drawn over it. Counting it would fire on nearly every plotted scene.
    document = doc(
        timed_scene(
            [
                # Shifted up: an axes is six units tall and the usable band is
                # six and a fifth, so one centred on the origin reaches into
                # the subtitles. See the test below.
                ir.Axes(
                    id="ax",
                    x_range=(-1.0, 1.0, 1.0),
                    y_range=(-1.0, 1.0, 1.0),
                    position=(0.0, 0.8),
                ),
                plot("p", "ax"),
                text("t", (0.0, 0.0), "label"),
            ],
            [
                step("create", "ax", duration=2.0),
                step("create", "p", duration=2.0),
                step("write", "t", duration=2.0),
            ],
        )
    )
    assert names(ir_rules.check_document(document)) == set()


def test_an_overlapping_pair_is_reported_once_however_long_it_lasts():
    document = doc(
        timed_scene(
            [text("a", (0.0, 0.0), "hello"), text("b", (0.1, 0.0), "hello")],
            [
                step("write", "a", duration=2.0),
                step("write", "b", duration=2.0),
                step("indicate", "a", duration=1.0),
                step("wait", duration=1.0),
            ],
        )
    )
    issues = [i for i in ir_rules.check_document(document) if i.rule == "no-overlap"]

    assert len(issues) == 1
    assert issues[0].where == "scenes[0].steps[1]"


def test_objects_never_on_screen_together_do_not_overlap():
    document = doc(
        timed_scene(
            [text("a", (0.0, 0.0), "hello"), text("b", (0.0, 0.0), "hello")],
            [
                step("write", "a", duration=2.0),
                step("fade_out", "a", duration=2.0),
                step("write", "b", duration=2.0),
            ],
        )
    )
    assert names(ir_rules.check_document(document)) == set()


def test_a_move_is_taken_into_account_by_the_overlap_check():
    # Declared apart, moved on top of each other.
    document = doc(
        timed_scene(
            [text("a", (-3.0, 0.0), "hello"), text("b", (3.0, 0.0), "hello")],
            [
                step("write", "a", duration=2.0),
                step("write", "b", duration=2.0),
                step("move", "b", duration=2.0, to=(-3.0, 0.0)),
            ],
        )
    )
    assert names(ir_rules.check_document(document)) == {"no-overlap"}


@pytest.mark.slow
def test_the_estimator_tracks_what_manim_actually_produces():
    # The constants are measured, and this is what keeps them measured: it
    # fails if manim's text metrics move, and it would have failed on the
    # constants SCENE_IR.md first carried, which were out by a factor of two.
    import manim

    for content, size in (
        ("x", 36),
        ("hello", 36),
        ("derivative", 48),
        ("The Pythagorean Theorem", 48),
    ):
        box = ir_rules.bounding_box(
            ir.Text(id="t", content=content, position=(0.0, 0.0), font_size=size)
        )
        real = manim.Text(content, font_size=size)
        assert 0.8 <= box.width / real.width <= 1.5, (content, size, box.width, real.width)
        assert 0.9 <= box.height / real.height <= 2.5, (content, size)

    # Char count is a weak proxy for LaTeX -- \frac{d}{dx} is twelve characters
    # that render narrow -- so mathtex gets looser bounds on purpose.
    formula = "a^2 + b^2 = c^2"
    box = ir_rules.bounding_box(
        ir.MathTex(id="m", content=formula, position=(0.0, 0.0), font_size=48)
    )
    real = manim.MathTex(formula, font_size=48)
    assert 0.5 <= box.width / real.width <= 2.5
    assert 0.9 <= box.height / real.height <= 3.0

    real_axes = manim.Axes(x_range=[-3, 3, 1], y_range=[-2, 2, 1])
    box = ir_rules.bounding_box(axes("ax"))
    assert box.width == pytest.approx(real_axes.width, abs=0.2)
    assert box.height == pytest.approx(real_axes.height, abs=0.2)


def test_a_label_sitting_on_the_shape_it_labels_is_not_an_overlap():
    # Measured on a real layout: every no-overlap warning it raised was a
    # label or an arrow against the polygon it referred to. A warning drives
    # simplify_ir, so a false one costs a scene that was right.
    document = doc(
        timed_scene(
            [
                ir.Polygon(id="tri", points=((-2.0, -1.0), (2.0, -1.0), (2.0, 2.0))),
                text("label", (0.0, 0.0), "hypotenuse"),
            ],
            [step("create", "tri", duration=3.0), step("write", "label", duration=3.0)],
        )
    )
    assert names(ir_rules.check_document(document)) == set()


def test_two_shapes_overlapping_is_the_content_not_a_defect():
    # The squares in a Pythagorean figure sit on the triangle's sides.
    document = doc(
        timed_scene(
            [
                ir.Polygon(id="tri", points=((-2.0, -1.0), (2.0, -1.0), (2.0, 2.0))),
                ir.Rectangle(id="sq", center=(0.0, 0.0), width=3.0, height=3.0),
            ],
            [step("create", "tri", duration=3.0), step("create", "sq", duration=3.0)],
        )
    )
    assert names(ir_rules.check_document(document)) == set()


def test_overlapping_text_is_still_caught():
    document = doc(
        timed_scene(
            [text("a", (0.0, 0.0), "hello"), ir.MathTex(id="b", content="x+y", position=(0.0, 0.0))],
            [step("write", "a", duration=3.0), step("write", "b", duration=3.0)],
        )
    )
    assert names(ir_rules.check_document(document)) == {"no-overlap"}


def test_the_bottom_margin_is_the_subtitle_band_rather_than_the_frame_edge():
    # Measured on an 854x480 render: a two-line cue occupies y -3.25 to -2.52,
    # so anything below -2.4 is covered by the words it was timed against.
    low, high = ir_rules.SUBTITLE_BAND
    assert high < -ir_rules.MARGIN_BOTTOM < 0
    assert low > -4.0


def test_an_axes_centred_on_the_origin_now_reaches_into_the_subtitles():
    # Six units tall against a usable six and a fifth. This is a true warning,
    # not a regression: the bottom of that axes is where the narration goes.
    document = doc(
        timed_scene([axes("ax")], [step("create", "ax", duration=6.0)])
    )
    assert names(ir_rules.check_document(document)) == {"in-frame"}


def test_a_title_that_used_to_be_warned_about_is_left_alone():
    # A real run warned five times about titles reaching y 3.68 against a
    # frame edge of 4.0, and not one of them was clipped in the video.
    title = ir.Text(id="title", content="Every number breaks into primes",
                    position=(0.0, 3.4), font_size=42)
    box = ir_rules.bounding_box(title)

    assert box.max_y > 3.6  # what the old margin rejected
    assert box.fits(ir_rules.MARGIN_X, ir_rules.MARGIN_TOP, ir_rules.MARGIN_BOTTOM)


def test_the_message_names_the_edge_that_was_crossed():
    # "outside the margin" is not actionable; "into the subtitle band" is.
    low = doc(timed_scene([text("t", (0.0, -3.0))], [step("write", "t", duration=6.0)]))
    high = doc(timed_scene([text("t", (0.0, 3.9))], [step("write", "t", duration=6.0)]))
    wide = doc(timed_scene([text("t", (6.9, 0.0))], [step("write", "t", duration=6.0)]))

    assert "subtitle band" in ir_rules.check_document(low)[0].message
    assert "top margin" in ir_rules.check_document(high)[0].message
    assert "side margin" in ir_rules.check_document(wide)[0].message
