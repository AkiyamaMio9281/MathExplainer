"""The escalation ladder: what it does in what order, and where it stops.

None of this needs a model, a subprocess or a renderer. The ladder's four
actions are injected, so a fake that fails a set number of times exercises
exactly what is interesting -- the ordering and the bounds -- in
milliseconds.

Two properties carry the design. Each rung changes a *different* layer, so the
order matters and is asserted as a sequence rather than as a set. And every
rung is bounded, so a scene the model cannot get right costs a known number of
calls rather than an overnight bill: at the defaults the worst case is twelve,
and that ceiling is asserted rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from explainer import ir, llm, repair


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """Enough of validate.Check for the ladder."""

    ok: bool
    error: str = ""
    scene_name: str = "Demo"
    output: Path | None = None

    @property
    def level(self):
        return type("L", (), {"name": "DRYRUN"})()


class Fake:
    """Fails the first *failures* validations, then passes.

    Counts everything, so a test can assert on calls rather than on output.
    """

    def __init__(self, failures=0, *, render_fails=0, raises=None, never_passes=False):
        self.failures = failures
        self.render_fails = render_fails
        self.raises = raises
        self.never_passes = never_passes
        self.generated: list[ir.Scene] = []
        self.repaired: list[str] = []
        self.rendered = 0
        self.validations = 0

    # -- the four tools ----------------------------------------------------

    def generate(self, client, scene, title):
        if self.raises:
            raise self.raises
        self.generated.append(scene)
        return "code v%d" % len(self.generated), self._reply()

    def repair(self, client, code, error):
        self.repaired.append(error)
        return code + "+", self._reply()

    def validate(self, workdir, code):
        self.validations += 1
        if self.never_passes or self.validations <= self.failures:
            return Check(False, error=f"NameError on validation {self.validations}")
        return Check(True)

    def render(self, workdir, name, quality):
        self.rendered += 1
        if self.rendered <= self.render_fails:
            return Check(False, error="could not rasterise")
        return Check(True, output=Path("scene.mp4"))

    def tools(self) -> repair.Tools:
        return repair.Tools(self.generate, self.repair, self.validate, self.render)

    @staticmethod
    def _reply():
        return llm.Reply(
            text="",
            usage=llm.Usage(input_tokens=10, output_tokens=10, calls=1),
            seconds=0.0,
            model=llm.MODEL,
            stop_reason="end_turn",
        )


def scene(objects=6) -> ir.Scene:
    """A scene with enough in it that every reduction has something to do."""
    return ir.Scene(
        id="s",
        narration="word " * 30,
        objects=(
            ir.Axes(id="ax", x_range=(-1.0, 1.0, 1.0), y_range=(-1.0, 1.0, 1.0)),
            ir.Plot(id="p", axes="ax", expression="x"),
            *[
                ir.Text(id=f"t{n}", content="x", position=(0.0, float(n)))
                for n in range(objects)
            ],
        ),
        steps=(
            ir.Step(action=ir.Action.CREATE, target="ax", duration=1.0),
            ir.Step(action=ir.Action.CREATE, target="p", duration=1.0),
            *[
                ir.Step(action=ir.Action.WRITE, target=f"t{n}", duration=1.0)
                for n in range(objects)
            ],
            ir.Step(action=ir.Action.TRANSFORM, target="t0", into="t1", duration=1.0),
        ),
    )


def climb(fake: Fake, subject=None, **kwargs) -> repair.Outcome:
    return repair.climb(
        subject or scene(), Path("."), object(), tools=fake.tools(), **kwargs
    )


def kinds(outcome: repair.Outcome) -> list[str]:
    return [a.kind for a in outcome.attempts]


# ---------------------------------------------------------------------------
# The cheap path stays cheap
# ---------------------------------------------------------------------------


def test_a_scene_that_works_costs_one_call():
    fake = Fake(failures=0)
    outcome = climb(fake)

    assert outcome.rendered
    assert kinds(outcome) == ["codegen"]
    assert outcome.calls == 1
    assert outcome.repairs == 0 and outcome.simplifications == 0
    assert fake.rendered == 1


def test_the_video_and_the_code_come_back_with_the_outcome():
    outcome = climb(Fake())

    assert outcome.video == Path("scene.mp4")
    assert outcome.code == "code v1"
    assert outcome.usage.calls == 1


# ---------------------------------------------------------------------------
# Repair before simplification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_repairs_happen_before_anything_is_simplified(failures):
    fake = Fake(failures=failures)
    outcome = climb(fake)

    assert outcome.rendered
    assert kinds(outcome) == ["codegen"] + ["repair"] * failures
    assert outcome.simplifications == 0
    assert outcome.calls == failures + 1


def test_the_traceback_goes_back_verbatim_each_round():
    fake = Fake(failures=2)
    climb(fake)

    # Not a summary, and not the first error repeated: what the checker said
    # this time.
    assert fake.repaired == [
        "NameError on validation 1",
        "NameError on validation 2",
    ]


def test_repairs_stop_at_the_bound_and_escalate():
    fake = Fake(never_passes=True)
    outcome = climb(fake, repair_rounds=3, simplify_rounds=0)

    assert not outcome.rendered
    assert outcome.repairs == 3
    assert kinds(outcome) == ["codegen", "repair", "repair", "repair"]


# ---------------------------------------------------------------------------
# Then simplification, then dropping
# ---------------------------------------------------------------------------


def test_the_ladder_repairs_then_simplifies_then_drops():
    # The shape the architecture describes, asserted as a sequence: each rung
    # changes a different layer, so the order is the design.
    fake = Fake(never_passes=True)
    outcome = climb(fake, repair_rounds=2, simplify_rounds=2)

    assert kinds(outcome) == [
        "codegen", "repair", "repair", "simplify",
        "codegen", "repair", "repair", "simplify",
        "codegen", "repair", "repair",
    ]
    assert outcome.status == repair.DROPPED
    assert "still failing after 2 simplification(s)" in outcome.error


def test_each_simplification_regenerates_from_the_simpler_scene():
    fake = Fake(never_passes=True)
    climb(fake, repair_rounds=0, simplify_rounds=2)

    # Three codegen calls, each on a scene asking for strictly less.
    assert len(fake.generated) == 3
    sizes = [len(s.objects) for s in fake.generated]
    assert sizes == sorted(sizes, reverse=True) and sizes[0] > sizes[-1]


def test_a_simplification_is_not_a_model_call():
    fake = Fake(never_passes=True)
    outcome = climb(fake, repair_rounds=1, simplify_rounds=2)

    assert outcome.simplifications == 2
    assert outcome.calls == 6  # three cycles of codegen plus one repair
    assert outcome.usage.calls == 6


def test_dropping_says_what_was_tried_and_what_failed():
    outcome = climb(Fake(never_passes=True), repair_rounds=1, simplify_rounds=1)

    assert outcome.status == repair.DROPPED
    assert "NameError" in outcome.error
    assert "simplification" in outcome.error


def test_a_scene_that_cannot_be_reduced_is_dropped_early():
    # One object, one step: nothing to drop, so the ladder stops before
    # spending its simplification budget.
    minimal = ir.Scene(
        id="s",
        narration="word " * 20,
        objects=(ir.Text(id="t", content="x", position=(0.0, 0.0)),),
        steps=(ir.Step(action=ir.Action.WRITE, target="t", duration=8.0),),
    )
    outcome = climb(Fake(never_passes=True), minimal, repair_rounds=1, simplify_rounds=2)

    assert outcome.status == repair.DROPPED
    assert "cannot be simplified further" in outcome.error
    assert outcome.simplifications == 0
    assert outcome.calls == 2  # one cycle, not three


# ---------------------------------------------------------------------------
# The bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "repairs,simplifies", [(0, 0), (1, 1), (3, 2), (5, 3)]
)
def test_no_run_exceeds_its_own_ceiling(repairs, simplifies):
    fake = Fake(never_passes=True)
    outcome = climb(fake, repair_rounds=repairs, simplify_rounds=simplifies)

    ceiling = (1 + simplifies) * (1 + repairs)
    assert outcome.calls <= ceiling
    assert outcome.repairs <= repairs * (1 + simplifies)
    assert outcome.simplifications <= simplifies


def test_the_default_ceiling_is_twelve_calls_a_scene():
    # The number the agent's budget is set against. If the defaults move, this
    # is where the new worst case shows up.
    outcome = climb(Fake(never_passes=True))

    assert outcome.calls <= 12
    assert (1 + repair.SIMPLIFY_ROUNDS) * (1 + repair.REPAIR_ROUNDS) == 12


def test_zero_rounds_means_one_attempt_and_no_more():
    fake = Fake(never_passes=True)
    outcome = climb(fake, repair_rounds=0, simplify_rounds=0)

    assert kinds(outcome) == ["codegen"]
    assert outcome.calls == 1


# ---------------------------------------------------------------------------
# Failures that are not a failing check
# ---------------------------------------------------------------------------


def test_a_render_that_fails_simplifies_rather_than_repairing_again():
    # The code already runs, so repairing it is the wrong rung: a scene that
    # executes but will not rasterise is asking for too much.
    fake = Fake(failures=0, render_fails=1)
    outcome = climb(fake)

    assert kinds(outcome) == ["codegen", "render", "simplify", "codegen"]
    assert outcome.rendered
    assert outcome.repairs == 0


def test_a_truncated_generation_simplifies_rather_than_asking_again():
    # A scene that asks for less produces a shorter file, so simplification is
    # the useful response to hitting max_tokens.
    fake = Fake(raises=llm.Truncated("output hit max_tokens"))
    outcome = climb(fake, repair_rounds=3, simplify_rounds=1)

    # Note what is absent: no repair round, and no second simplification --
    # the ladder does not attempt a rung it is not allowed to take.
    assert kinds(outcome) == ["codegen", "simplify", "codegen"]
    assert outcome.status == repair.DROPPED
    assert outcome.repairs == 0


def test_a_refusal_is_recorded_rather_than_retried():
    fake = Fake(raises=llm.Refused("declined", "cyber"))
    outcome = climb(fake, repair_rounds=3, simplify_rounds=0)

    assert outcome.status == repair.DROPPED
    assert "Refused" in outcome.attempts[0].error


# ---------------------------------------------------------------------------
# The attempt log
# ---------------------------------------------------------------------------


def test_the_log_survives_success_as_well_as_failure():
    # A scene that took three repairs is as interesting to the metrics as one
    # that was dropped.
    outcome = climb(Fake(failures=3))

    assert outcome.rendered
    assert len(outcome.attempts) == 4
    assert [a.ok for a in outcome.attempts] == [False, False, False, True]


def test_every_attempt_says_which_scene_version_it_was_working_on():
    outcome = climb(Fake(never_passes=True), repair_rounds=1, simplify_rounds=2)
    stages = [(a.kind, a.stage) for a in outcome.attempts]

    # Stage 0 is what the layout asked for; each simplification moves it on.
    assert stages[0] == ("codegen", 0)
    assert stages[-1][1] > 0
    assert [s for _, s in stages] == sorted(s for _, s in stages)


def test_the_trail_reads_as_what_happened():
    outcome = climb(Fake(failures=1))

    assert outcome.trail() == "codegen x -> repair"


def test_the_log_carries_the_tier_that_rejected_each_attempt():
    outcome = climb(Fake(failures=1))

    assert outcome.attempts[0].level == "DRYRUN"
