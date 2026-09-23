"""The layout stage: what the schema can carry, and what it must not change.

Two properties matter more than the rest and are asserted directly.

The narration is the plan's. It is not requested from the model and echoed
back, it is attached afterwards by scene id, so "the layout can be repaired
without the explanation changing underneath it" holds by construction. A test
asserts a model that rewrites the words has no effect on the lesson.

And the schema is limited on purpose. The API compiles it into a grammar with
a size limit that a nine-way tagged union exceeds; the constants here are what
was measured, and a test pins the boundary so a future edit that widens the
vocabulary discovers the ceiling here rather than as a 400 in a live run.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from explainer import ir, ir_rules, layout, llm, plan as planning


def lesson(*, words=40, ids=("intro", "proof")) -> planning.LessonPlan:
    return planning.LessonPlan(
        title="A Lesson",
        audience="beginners",
        scenes=tuple(
            planning.Beat(id=i, beat=f"Teach {i}.", narration="word " * words)
            for i in ids
        ),
    )


def document(*ids: str, narration="something else entirely") -> ir.Document:
    return ir.Document(
        title="A Lesson",
        scenes=tuple(
            ir.Scene(
                id=i,
                narration=narration,
                objects=(ir.Text(id="t", content="x", position=(0.0, 0.0)),),
                steps=(ir.Step(action=ir.Action.WRITE, target="t", duration=16.0),),
            )
            for i in ids
        ),
    )


# ---------------------------------------------------------------------------
# The narration belongs to the plan
# ---------------------------------------------------------------------------


def test_the_plans_words_replace_whatever_the_layout_returned():
    carried, issues = layout.carry_narration(document("intro", "proof"), lesson())

    assert issues == ()
    assert all(s.narration == "word " * 40 for s in carried.scenes)


def test_narration_is_not_in_the_schema_at_all():
    # Not asked for and echoed back: attached afterwards, so it cannot drift.
    scene = layout.schema()["properties"]["scenes"]["items"]

    assert "narration" not in scene["properties"]
    assert "narration" not in scene["required"]


def test_a_renamed_scene_is_reported_from_both_sides():
    # A layout that renames a scene loses one and invents another, and both
    # halves are worth saying: the invented one has no narration to be timed
    # against, and the lost one is a beat the lesson no longer teaches.
    _, issues = layout.carry_narration(document("intro", "invented"), lesson())

    assert [i.rule for i in issues] == ["matches-plan", "matches-plan"]
    assert "invented" in issues[0].message
    assert "proof" in issues[1].message


def test_a_scene_the_plan_asked_for_and_did_not_get_is_reported():
    _, issues = layout.carry_narration(document("intro"), lesson())

    assert [i.rule for i in issues] == ["matches-plan"]
    assert "proof" in issues[0].message


def test_matching_the_plan_is_an_error_not_a_warning():
    # Without a matching id there is nothing to attach, and the scene would
    # render with no narration to have been timed against.
    _, issues = layout.carry_narration(document("wrong"), lesson(ids=("intro",)))

    assert all(i.severity is ir.Severity.ERROR for i in issues)


# ---------------------------------------------------------------------------
# The schema, and the ceiling it sits under
# ---------------------------------------------------------------------------


def test_the_generated_vocabulary_is_a_subset_of_the_ir_vocabulary():
    assert set(layout.GENERATED_TYPES) < set(ir.OBJECT_TYPES)
    # axes and plot stay valid IR -- a hand-written document may use them --
    # they are only outside what this stage produces.
    assert {"axes", "plot"} <= set(ir.OBJECT_TYPES)
    assert {"axes", "plot"}.isdisjoint(layout.GENERATED_TYPES)


def test_the_schema_stays_under_the_size_that_was_measured_to_compile():
    # Six types compile at about 3.5 kB; seven do not. Widening the vocabulary
    # should fail here rather than as a 400 halfway through a live run.
    assert len(json.dumps(layout.schema())) < 3800


def test_every_generated_type_offers_a_variant_with_its_required_fields():
    variants = {
        v["properties"]["type"]["enum"][0]: v
        for v in layout.schema()["properties"]["scenes"]["items"]["properties"][
            "objects"
        ]["items"]["anyOf"]
    }

    assert set(variants) == set(layout.GENERATED_TYPES)
    assert set(variants["circle"]["required"]) == {"type", "id", "center", "radius"}
    assert set(variants["text"]["required"]) == {"type", "id", "content", "position"}


def test_the_schema_is_shaped_the_way_structured_outputs_demand():
    # Verified against the API: additionalProperties must be present and false,
    # minItems only 0 or 1, maxItems unsupported at all.
    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
            assert node.get("minItems", 0) in (0, 1)
            assert "maxItems" not in node
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(layout.schema())


def test_colours_are_an_enum_so_an_invented_one_cannot_be_returned():
    variant = next(
        v
        for v in layout.schema()["properties"]["scenes"]["items"]["properties"][
            "objects"
        ]["items"]["anyOf"]
        if v["properties"]["type"]["enum"] == ["text"]
    )
    assert variant["properties"]["color"]["enum"] == list(ir.COLORS)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_prompt_works_out_each_scenes_timing_budget():
    # The model chose its own durations when it was not told, and produced an
    # animation three times its narration.
    prompt = layout.plan_prompt(lesson(words=50))

    assert "50 words" in prompt
    assert f"{50 / ir_rules.WORDS_PER_MINUTE * 60:.1f}s" in prompt
    assert "must total about" in prompt


def test_a_spoken_scene_is_given_its_measured_length_not_an_estimate():
    # Once the narration has been synthesised the audio is fixed, so the
    # animation is being fitted to it. Handing the model a word-count estimate
    # here would be asking it to fit a length that is no longer the length.
    plan = lesson(words=50)
    spoken = replace(
        plan, scenes=(replace(plan.scenes[0], seconds=31.7),)
    )
    prompt = layout.plan_prompt(spoken)

    assert "31.7s" in prompt
    assert f"{50 / ir_rules.WORDS_PER_MINUTE * 60:.1f}s" not in prompt
    assert "must total about" not in prompt
    assert "must fill" in prompt


def test_the_preamble_states_the_frame_margin_and_the_pacing_tolerance():
    assert "-6.6 to 6.6" in layout.PREAMBLE
    assert "-2.4 to 3.8" in layout.PREAMBLE
    assert "30%" in layout.PREAMBLE


def test_the_preamble_tells_the_model_the_bottom_band_is_taken():
    # Subtitles are burned in below y = -2.5. A layout that does not know that
    # puts a caption where the words describing it will be.
    assert "bottom of the frame is not yours" in layout.PREAMBLE
    assert "-2.5" in layout.PREAMBLE


def test_the_preamble_quotes_the_margins_the_rules_actually_enforce():
    # Aiming at one number and being judged against another is how a stage
    # produces layouts that are rejected for following its own instructions.
    assert str(ir_rules.MARGIN_X) in layout.PREAMBLE
    assert str(ir_rules.MARGIN_TOP) in layout.PREAMBLE
    assert str(ir_rules.MARGIN_BOTTOM) in layout.PREAMBLE


def test_the_preamble_offers_the_extent_estimates_the_rules_will_judge_by():
    # The same measured constants no-overlap and in-frame use, so the stage
    # aims at what the next one checks.
    assert "0.25 units wide per character" in layout.PREAMBLE
    assert "0.5 tall" in layout.PREAMBLE


def test_the_preamble_only_offers_the_types_the_schema_accepts():
    for name in layout.GENERATED_TYPES:
        assert f"- `{name}`" in layout.PREAMBLE
    for name in ("axes", "plot", "rectangle"):
        assert f"- `{name}`" not in layout.PREAMBLE


def test_the_retry_prompt_carries_the_document_and_the_issues_verbatim():
    subject = document("intro")
    issues = ir_rules.check_document(subject)
    prompt = layout.retry_prompt(subject, issues)

    assert '"id": "intro"' in prompt
    for issue in issues:
        assert str(issue) in prompt


# ---------------------------------------------------------------------------
# make
# ---------------------------------------------------------------------------


class Stub(llm.Client):
    def __init__(self, payload):
        super().__init__()
        self.payload = payload
        self.kwargs = None

    def complete(self, prompt, **kwargs):
        self.kwargs = kwargs
        return llm.Reply(
            text=json.dumps(self.payload),
            usage=llm.Usage(calls=1),
            seconds=0.0,
            model=llm.MODEL,
            stop_reason="end_turn",
        )


def test_make_attaches_the_plans_narration_over_whatever_came_back():
    stub = Stub(document("intro", "proof", narration="the model's own words").to_dict())

    made, issues, _ = layout.make(stub, lesson())

    assert made is not None
    assert all(s.narration == "word " * 40 for s in made.scenes)
    assert stub.kwargs["preamble"] is layout.PREAMBLE


def test_make_reports_rule_failures_alongside_plan_mismatches():
    stub = Stub(document("intro", "proof").to_dict())

    _, issues, _ = layout.make(stub, lesson())

    # 16s of animation against 16s of narration is fine; the two scenes each
    # declare one object and one step, so nothing structural is wrong.
    assert layout.errors(issues) == (), layout.report(issues)


def test_make_reports_a_reply_that_is_not_json():
    class Junk(llm.Client):
        def complete(self, prompt, **kwargs):
            return llm.Reply(
                text="Certainly!",
                usage=llm.Usage(calls=1),
                seconds=0.0,
                model=llm.MODEL,
                stop_reason="end_turn",
            )

    made, issues, _ = layout.make(Junk(), lesson())

    assert made is None
    assert issues[0].rule == "document-shape"


def test_make_keeps_the_plans_title():
    stub = Stub({"version": "1", "title": "whatever", "scenes": []})

    made, _, _ = layout.make(stub, lesson())

    assert made.title == "A Lesson"


# ---------------------------------------------------------------------------
# A real layout
# ---------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.slow
def test_a_real_layout_satisfies_the_rules_it_was_told_about():
    subject = planning.LessonPlan(
        title="Right Triangles",
        audience="beginners",
        scenes=(
            planning.Beat(
                id="statement",
                beat="Show a right triangle and name its three sides.",
                narration=(
                    "A right triangle has one square corner. The two short sides "
                    "meet there, and the long side across from it is called the "
                    "hypotenuse, which is always the longest of the three."
                ),
            ),
            planning.Beat(
                id="relation",
                beat="Show the equation relating the three side lengths.",
                narration=(
                    "Square the two short sides and add them together. That total "
                    "is exactly the square of the hypotenuse, and it holds for "
                    "every right triangle you can draw."
                ),
            ),
        ),
    )

    made, issues, reply = layout.make(llm.Client(), subject)

    print(f"\n${reply.dollars():.4f}  {layout.report(issues) or 'no issues'}")
    for scene, beat in zip(made.scenes, subject.scenes):
        kinds = [type(o).TYPE for o in scene.objects]
        total = sum(s.duration for s in scene.steps)
        print(
            f"  {scene.id}: {len(scene.objects)} objects {kinds}, "
            f"{len(scene.steps)} steps, {total:.1f}s animated vs "
            f"{beat.spoken_seconds:.1f}s spoken"
        )

    assert made is not None
    assert layout.errors(issues) == (), layout.report(issues)
    assert [s.id for s in made.scenes] == ["statement", "relation"]
    # The words are the plan's, unchanged.
    assert made.scenes[0].narration == subject.scenes[0].narration


# ---------------------------------------------------------------------------
# Fitting the animation to the voice
# ---------------------------------------------------------------------------


def timed(*durations: float, scene_id="intro") -> ir.Document:
    return ir.Document(
        title="A Lesson",
        scenes=(
            ir.Scene(
                id=scene_id,
                narration="word " * 40,
                objects=(ir.Text(id="t", content="x", position=(0.0, 0.0)),),
                steps=tuple(
                    ir.Step(action=ir.Action.WAIT, duration=d) for d in durations
                ),
            ),
        ),
    )


def spoken(seconds: float, ids=("intro",)) -> planning.LessonPlan:
    plan = lesson(ids=ids)
    return replace(plan, scenes=tuple(replace(b, seconds=seconds) for b in plan.scenes))


def test_a_scene_that_undershoots_its_narration_is_stretched_to_meet_it():
    # Measured on a real run: told 46.9s, produced 44.5s, and every scene was
    # short. Sixteen seconds of voice ran past the end of the lesson.
    fitted, issues = layout.fit_to_speech(timed(20.0, 24.5), spoken(46.9))

    assert issues == ()
    assert sum(s.duration for s in fitted.scenes[0].steps) == pytest.approx(46.9, abs=0.02)


def test_an_animation_longer_than_its_narration_is_squeezed(fitted=None):
    fitted, _ = layout.fit_to_speech(timed(30.0, 30.0), spoken(40.0))

    assert sum(s.duration for s in fitted.scenes[0].steps) == pytest.approx(40.0, abs=0.02)


def test_which_beat_gets_more_time_stays_the_layouts_decision():
    # Uniform, so the model's relative choices survive. Putting the whole
    # difference into the last step would finish the animation early and leave
    # the viewer on a still frame while the voice caught up.
    fitted, _ = layout.fit_to_speech(timed(10.0, 30.0), spoken(60.0))
    steps = [s.duration for s in fitted.scenes[0].steps]

    assert steps[1] / steps[0] == pytest.approx(3.0)
    assert steps == pytest.approx([15.0, 45.0])


def test_a_scene_too_far_out_to_scale_is_reported_rather_than_stretched():
    # Ten seconds of animation under a minute of narration is not a rounding
    # difference, and stretching it six-fold would be slow motion.
    fitted, issues = layout.fit_to_speech(timed(10.0), spoken(60.0))

    assert [i.rule for i in issues] == ["fits-narration"]
    assert issues[0].severity is ir.Severity.WARNING
    assert [s.duration for s in fitted.scenes[0].steps] == [10.0]


def test_nothing_is_scaled_when_the_narration_was_never_spoken():
    # Without audio the word count is an estimate, and fitting an animation to
    # an estimate to two decimal places would be false precision.
    fitted, issues = layout.fit_to_speech(timed(20.0, 24.5), lesson())

    assert issues == ()
    assert [s.duration for s in fitted.scenes[0].steps] == [20.0, 24.5]


def test_a_scene_the_plan_does_not_have_is_left_alone():
    fitted, issues = layout.fit_to_speech(timed(20.0, scene_id="invented"), spoken(46.9))

    assert issues == ()
    assert [s.duration for s in fitted.scenes[0].steps] == [20.0]


def test_a_scene_with_no_time_in_it_is_not_divided_by_zero():
    fitted, issues = layout.fit_to_speech(timed(0.0, 0.0), spoken(46.9))

    assert issues == ()
    assert [s.duration for s in fitted.scenes[0].steps] == [0.0, 0.0]


def read_back(document: ir.Document, plan: planning.LessonPlan):
    reply = llm.Reply(
        text=document.to_json(),
        usage=llm.Usage(),
        seconds=0.0,
        model=llm.MODEL,
        stop_reason="end_turn",
    )
    return layout._read(reply, plan)


def test_the_fit_is_applied_before_the_rules_judge_the_document():
    # The fitted document is the one that gets rendered, so judging the
    # unfitted one would be judging a draft.
    #
    # `pacing` is the rule that notices. Seventy-eight words estimate to about
    # thirty seconds, so the rule accepts roughly 21s to 39s. This layout asks
    # for 55s, which it rejects -- and fits to 30s, which it accepts. Running
    # the rules first would report a scene that no longer exists.
    plan = replace(
        lesson(words=78, ids=("intro",)),
        scenes=(replace(lesson(words=78, ids=("intro",)).scenes[0], seconds=30.0),),
    )
    document, issues, _ = read_back(timed(25.0, 30.0), plan)

    assert sum(s.duration for s in document.scenes[0].steps) == pytest.approx(30.0, abs=0.02)
    assert "pacing" not in [i.rule for i in issues]


def test_the_rules_still_catch_a_scene_the_fit_could_not_rescue():
    # The converse, so the test above cannot pass by the rule never firing.
    plan = replace(
        lesson(words=78, ids=("intro",)),
        scenes=(replace(lesson(words=78, ids=("intro",)).scenes[0], seconds=30.0),),
    )
    # 5s against 30s of narration is a sixfold stretch: outside the fit's
    # range, so it stays as it is and `pacing` sees it.
    _, issues, _ = read_back(timed(5.0), plan)

    assert "fits-narration" in [i.rule for i in issues]
    assert "pacing" in [i.rule for i in issues]
