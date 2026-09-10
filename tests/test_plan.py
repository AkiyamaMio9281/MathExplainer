"""What can be wrong with a plan the schema already accepted.

Structured outputs guarantee the shape, so the tests that matter are not about
parsing. They are about the four things a perfectly schema-valid plan can get
wrong -- ids that are not identifiers, a scene count that is a budget, and
narration whose length implies a scene the IR rules will refuse.

That last one is the point of checking here at all. The bounds are derived
from `ir_rules.scene_length`, and a test asserts the two agree: if they ever
drift, the planner starts producing lessons that die one stage later, after a
model call.
"""

from __future__ import annotations

import json

import pytest

from explainer import ir, ir_rules, llm, plan


def beat(id="intro", words=60, beat_text="State the theorem."):
    return plan.Beat(id=id, beat=beat_text, narration="word " * words)


def lesson(*beats, title="A Lesson", audience="beginners"):
    return plan.LessonPlan(
        title=title, audience=audience, scenes=beats or (beat(), beat(id="proof"))
    )


def names(issues):
    return {issue.rule for issue in issues}


# ---------------------------------------------------------------------------
# The bounds this stage shares with the IR rules
# ---------------------------------------------------------------------------


def test_the_narration_bounds_match_what_the_ir_rules_will_accept():
    # Derived, not guessed. If these drift, the planner starts producing
    # lessons that are rejected one stage later, after a model call.
    assert plan.WORDS_PER_MINUTE == ir_rules.WORDS_PER_MINUTE
    assert plan.MIN_WORDS / plan.WORDS_PER_MINUTE * 60 == pytest.approx(
        ir_rules.MIN_SCENE_SECONDS, abs=0.5
    )
    assert plan.MAX_WORDS / plan.WORDS_PER_MINUTE * 60 == pytest.approx(
        ir_rules.MAX_SCENE_SECONDS, abs=0.5
    )


def test_a_scene_at_each_bound_is_a_scene_the_ir_rules_would_allow():
    for words in (plan.MIN_WORDS, plan.MAX_WORDS):
        seconds = plan.Beat(id="s", beat="b", narration="w " * words).spoken_seconds
        assert ir_rules.MIN_SCENE_SECONDS <= seconds <= ir_rules.MAX_SCENE_SECONDS


def test_an_id_the_planner_accepts_is_one_the_ir_rules_accept():
    assert plan.check(lesson(beat(id="right_triangle"), beat(id="proof"))) == ()
    assert ir.is_safe_id("right_triangle")


# ---------------------------------------------------------------------------
# One case per rule
# ---------------------------------------------------------------------------


def test_a_sound_plan_produces_nothing():
    assert plan.check(lesson()) == ()


@pytest.mark.parametrize(
    "rule,subject",
    [
        ("has-title", lesson(title="   ")),
        ("non-empty", plan.LessonPlan(title="t")),
        ("scene-count", lesson(beat())),
        (
            "scene-count",
            lesson(*[beat(id=f"s{n}") for n in range(plan.MAX_SCENES + 1)]),
        ),
        ("unique-ids", lesson(beat(id="same"), beat(id="same"))),
        ("safe-ids", lesson(beat(id="../../elsewhere"), beat(id="ok"))),
        ("safe-ids", lesson(beat(id="Intro"), beat(id="ok"))),
        ("narration-length", lesson(beat(words=3), beat(id="b"))),
        ("narration-length", lesson(beat(words=400), beat(id="b"))),
    ],
    ids=[
        "a plan with no title",
        "a plan with no scenes",
        "a lesson of one scene",
        "more scenes than the pipeline will render",
        "two scenes sharing an id",
        "a scene id that would escape the working directory",
        "a scene id that is not lowercase",
        "narration too short to fill a scene",
        "narration too long for one scene",
    ],
)
def test_each_case_trips_exactly_its_own_rule(rule, subject):
    issues = plan.check(subject)

    assert issues, "the case did not trip any rule"
    assert names(issues) == {rule}


def test_every_rule_has_a_case():
    covered = set()
    for _, subject in [
        ("has-title", lesson(title="   ")),
        ("non-empty", plan.LessonPlan(title="t")),
        ("scene-count", lesson(beat())),
        ("unique-ids", lesson(beat(id="same"), beat(id="same"))),
        ("safe-ids", lesson(beat(id="Intro"), beat(id="ok"))),
        ("narration-length", lesson(beat(words=3), beat(id="b"))),
    ]:
        covered |= names(plan.check(subject))
    assert covered == plan.RULE_NAMES


def test_a_path_traversing_id_is_an_error_not_a_warning():
    # It names a directory the pipeline writes into, so it is the one rule
    # here with a consequence beyond a scene looking wrong.
    (issue,) = plan.check(lesson(beat(id="../../etc"), beat(id="ok")))

    assert issue.severity is ir.Severity.ERROR
    assert "directory" in issue.message


def test_narration_length_is_a_warning_because_the_ir_stage_can_still_fix_it():
    (issue,) = plan.check(lesson(beat(words=3), beat(id="b")))
    assert issue.severity is ir.Severity.WARNING


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_a_well_formed_plan_parses():
    parsed, issues = plan.parse(
        {
            "title": "Derivatives",
            "audience": "first-year calculus",
            "scenes": [
                {"id": "intro", "beat": "State it.", "narration": "Here we go."}
            ],
        }
    )

    assert issues == ()
    assert parsed.scenes[0].id == "intro"
    assert parsed.title == "Derivatives"


def test_a_plan_round_trips_through_to_dict():
    original = lesson()
    parsed, issues = plan.parse(original.to_dict())

    assert issues == ()
    assert parsed == original


def test_a_scene_missing_a_field_is_reported_and_dropped():
    parsed, issues = plan.parse(
        {"title": "t", "audience": "a", "scenes": [{"id": "x"}, {"id": "y", "beat": "b", "narration": "n"}]}
    )

    assert [i.rule for i in issues] == ["plan-shape"]
    assert "beat" in issues[0].message and "narration" in issues[0].message
    assert [s.id for s in parsed.scenes] == ["y"]


@pytest.mark.parametrize("data", ["a string", 3, None, []], ids=str)
def test_a_plan_that_is_not_an_object_yields_nothing(data):
    parsed, issues = plan.parse(data)

    assert parsed is None
    assert issues[0].rule == "plan-shape"


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


def test_the_schema_is_strict_enough_for_structured_outputs():
    # additionalProperties false and an explicit required list are what the
    # API needs to guarantee the shape.
    assert plan.SCHEMA["additionalProperties"] is False
    assert set(plan.SCHEMA["required"]) == {"title", "audience", "scenes"}
    item = plan.SCHEMA["properties"]["scenes"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"id", "beat", "narration"}


def test_the_schema_travels_in_output_config_format():
    request = llm.build_request("x", schema=plan.SCHEMA)
    assert request["output_config"]["format"]["schema"] == plan.SCHEMA


def test_the_preamble_states_the_bounds_the_rules_enforce():
    # A prompt that does not say the limits produces plans that trip them.
    assert str(plan.MAX_SCENES) in plan.PREAMBLE
    assert str(plan.MAX_WORDS) in plan.PREAMBLE
    assert "lowercase identifier" in plan.PREAMBLE
    # Narration is heard, not read.
    assert "No LaTeX" in plan.PREAMBLE


def test_make_sends_the_schema_and_checks_what_comes_back():
    class Stub(llm.Client):
        def complete(self, prompt, **kwargs):
            assert kwargs["schema"] is plan.SCHEMA
            assert kwargs["preamble"] is plan.PREAMBLE
            return llm.Reply(
                text=json.dumps(lesson().to_dict()),
                usage=llm.Usage(calls=1),
                seconds=0.0,
                model=llm.MODEL,
                stop_reason="end_turn",
            )

    made, issues, reply = plan.make(Stub(), "explain the chain rule")

    assert issues == ()
    assert len(made.scenes) == 2
    assert reply.usage.calls == 1


def test_a_reply_that_is_not_json_is_reported_rather_than_raised():
    class Stub(llm.Client):
        def complete(self, prompt, **kwargs):
            return llm.Reply(
                text="I'd be glad to help!",
                usage=llm.Usage(calls=1),
                seconds=0.0,
                model=llm.MODEL,
                stop_reason="end_turn",
            )

    made, issues, _ = plan.make(Stub(), "x")

    assert made is None
    assert issues[0].rule == "plan-shape"


# ---------------------------------------------------------------------------
# A real plan
# ---------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.slow
def test_a_real_plan_satisfies_its_own_rules():
    made, issues, reply = plan.make(
        llm.Client(), "explain why the derivative of sin is cos"
    )

    print(f"\n{made.title} -- {made.audience}")
    for scene in made.scenes:
        print(f"  {scene.id}: {scene.beat} ({scene.spoken_seconds:.0f}s)")
    print(f"  {plan.report(issues) or 'no issues'}  ${reply.dollars():.4f}")

    assert made is not None
    assert plan.errors(issues) == (), plan.report(issues)
    assert plan.MIN_SCENES <= len(made.scenes) <= plan.MAX_SCENES
    # Every id must survive into the IR stage, where it becomes a directory.
    assert all(ir.is_safe_id(s.id) for s in made.scenes)
