"""The state machine, the budget, and the numbers read off a finished run.

Every stage is injected, so none of this spends anything or starts a
subprocess. What is being tested is not whether the stages work -- each has
its own file for that -- but what the agent does with them: the order it calls
them in, when it refuses to call them at all, and what it keeps when it stops
early.

The budget tests are the ones that matter most. The ladder bounds a scene at
twelve calls and knows nothing about how many scenes a lesson has, so the
agent is the only thing standing between a bad night and a bill.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from explainer import agent, ir, llm, metrics, plan as planning, repair, speech


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def reply(dollars=0.01) -> llm.Reply:
    # input tokens chosen so one call costs about *dollars*.
    return llm.Reply(
        text="",
        usage=llm.Usage(input_tokens=int(dollars * 200_000), calls=1),
        seconds=0.0,
        model=llm.MODEL,
        stop_reason="end_turn",
    )


def lesson(*ids: str) -> planning.LessonPlan:
    return planning.LessonPlan(
        title="A Lesson",
        audience="beginners",
        scenes=tuple(
            planning.Beat(id=i, beat=f"Teach {i}.", narration="word " * 40)
            for i in (ids or ("one", "two"))
        ),
    )


def document(*ids: str) -> ir.Document:
    return ir.Document(
        title="A Lesson",
        scenes=tuple(
            ir.Scene(
                id=i,
                narration="word " * 40,
                objects=(ir.Text(id="t", content="x", position=(0.0, 0.0)),),
                steps=(ir.Step(action=ir.Action.WRITE, target="t", duration=16.0),),
            )
            for i in (ids or ("one", "two"))
        ),
    )


@dataclass
class Joined:
    ok: bool = True
    output: Path | None = Path("lesson.mp4")
    error: str = ""


class Stub:
    """Records every dispatch and hands back whatever it was told to."""

    def __init__(
        self,
        *,
        plan=None,
        plan_issues=(),
        doc=None,
        layout_issues=(),
        retry_issues=None,
        scene_cost=0.01,
        fails: set[str] = frozenset(),
        join_ok=True,
        burn_ok=True,
        mux_ok=True,
        mute: set[str] = frozenset(),
        speech_seconds=16.0,
        clip_seconds=16.0,
    ):
        self.plan_value = plan if plan is not None else lesson()
        self.plan_issues = plan_issues
        self.doc = doc if doc is not None else document()
        self.layout_issues = layout_issues
        self.retry_issues = retry_issues
        self.scene_cost = scene_cost
        self.fails = set(fails)
        self.join_ok = join_ok
        self.burn_ok = burn_ok
        self.mux_ok = mux_ok
        self.mute = set(mute)
        self.speech_seconds = speech_seconds
        self.clip_seconds = clip_seconds
        self.subtitled: list[str] = []
        self.calls: list[str] = []
        self.climbed: list[str] = []
        self.spoke: list[str] = []
        self.cues: list = []
        self.placements: list = []
        self.total = 0.0
        self.voice = ""

    def plan(self, client, prompt, **kwargs):
        self.calls.append("plan")
        return self.plan_value, self.plan_issues, reply()

    def layout(self, client, lesson_, **kwargs):
        self.calls.append("layout")
        return self.doc, self.layout_issues, reply()

    def retry(self, client, lesson_, doc, issues, **kwargs):
        self.calls.append("retry")
        return self.doc, (
            self.layout_issues if self.retry_issues is None else self.retry_issues
        ), reply()

    def climb(self, scene, workdir, client, **kwargs):
        self.calls.append(f"climb:{scene.id}")
        self.climbed.append(scene.id)
        failed = scene.id in self.fails
        return repair.Outcome(
            scene_id=scene.id,
            status=repair.DROPPED if failed else repair.RENDERED,
            scene=scene,
            attempts=(
                repair.Attempt(
                    repair.CODEGEN, 0, 0, not failed, usage=reply(self.scene_cost).usage
                ),
            ),
            video=None if failed else Path(f"{scene.id}.mp4"),
            usage=reply(self.scene_cost).usage,
            error="it would not compile" if failed else "",
        )

    def speak(self, scenes, directory, **kwargs):
        self.calls.append("speak")
        self.spoke = [scene_id for scene_id, _ in scenes]
        self.voice = kwargs.get("voice")
        out = {}
        for scene_id, narration in scenes:
            if scene_id in self.mute:
                out[scene_id] = speech.Spoken(False, error="the endpoint refused")
                continue
            words = narration.split()
            seconds = self.speech_seconds
            step = seconds / max(len(words), 1)
            out[scene_id] = speech.Spoken(
                True,
                path=directory / f"{scene_id}.mp3",
                seconds=seconds,
                words=tuple(
                    speech.Word(i * step, (i + 1) * step, w)
                    for i, w in enumerate(words)
                ),
            )
        return out

    def concat(self, clips, out, **kwargs):
        self.calls.append("concat")
        return Joined(ok=self.join_ok, output=out, error="" if self.join_ok else "mismatch")

    def measure(self, clip, **kwargs):
        return self.clip_seconds

    def subtitle(self, video_path, cues, **kwargs):
        self.calls.append("subtitle")
        self.cues = list(cues)
        self.subtitled = [cue.text for cue in cues]
        return Joined(ok=self.burn_ok, output=video_path,
                      error="" if self.burn_ok else "libass said no")

    def sound(self, video_path, placements, total, **kwargs):
        self.calls.append("sound")
        self.placements = list(placements)
        self.total = total
        return Joined(ok=self.mux_ok, output=video_path,
                      error="" if self.mux_ok else "no such encoder")

    def stages(self) -> agent.Stages:
        return agent.Stages(
            self.plan,
            self.speak,
            self.layout,
            self.retry,
            self.climb,
            self.concat,
            self.measure,
            self.subtitle,
            self.sound,
        )


def run(stub: Stub, tmp_path, **kwargs) -> agent.Run:
    kwargs.setdefault("audio", False)
    return agent.run("explain something", tmp_path, object(), stages=stub.stages(), **kwargs)


# ---------------------------------------------------------------------------
# The happy path, and the order
# ---------------------------------------------------------------------------


def test_the_stages_run_in_order_and_once_each(tmp_path):
    stub = Stub()
    state = run(stub, tmp_path)

    assert stub.calls == [
        "plan", "layout", "climb:one", "climb:two", "concat", "subtitle"
    ]
    assert state.ok
    assert state.video == tmp_path / "lesson.mp4"
    assert state.subtitled


def test_each_scene_gets_its_own_directory(tmp_path):
    captured = []
    stub = Stub()
    inner = stub.climb
    stub.climb = lambda scene, workdir, client, **kw: (
        captured.append(workdir) or inner(scene, workdir, client, **kw)
    )
    run(stub, tmp_path)

    assert captured == [tmp_path / "one", tmp_path / "two"]


def test_the_cost_of_every_stage_lands_in_one_total(tmp_path):
    state = run(Stub(scene_cost=0.05), tmp_path)

    # plan + layout + two scenes.
    assert state.usage.calls == 4
    assert state.dollars() == pytest.approx(0.01 + 0.01 + 0.05 + 0.05)


# ---------------------------------------------------------------------------
# Refusing to spend
# ---------------------------------------------------------------------------


def test_a_rejected_plan_never_reaches_the_layout(tmp_path):
    stub = Stub(plan_issues=(ir.Issue("scene-count", "one scene", "$"),))
    state = run(stub, tmp_path)

    assert stub.calls == ["plan"]
    assert not state.ok
    assert "the plan was rejected" in state.error


def test_a_plan_with_only_warnings_carries_on(tmp_path):
    warning = ir.Issue("narration-length", "a bit short", "$", ir.Severity.WARNING)
    stub = Stub(plan_issues=(warning,))
    state = run(stub, tmp_path)

    assert state.ok
    assert state.plan_issues == (warning,)


def test_a_layout_with_errors_is_sent_back_once_and_then_given_up_on(tmp_path):
    broken = (ir.Issue("non-empty", "a scene needs a step", "scenes[0]"),)
    stub = Stub(layout_issues=broken)
    state = run(stub, tmp_path)

    # Two attempts, not three: a third attempt at the same failure is a third
    # bill for the same answer.
    assert stub.calls == ["plan", "layout", "retry"]
    assert state.layout_attempts == agent.LAYOUT_ROUNDS
    assert not state.ok
    assert "after 2 attempts" in state.error


def test_a_layout_fixed_on_the_retry_carries_on(tmp_path):
    stub = Stub(
        layout_issues=(ir.Issue("non-empty", "x", "scenes[0]"),), retry_issues=()
    )
    state = run(stub, tmp_path)

    assert stub.calls == [
        "plan", "layout", "retry", "climb:one", "climb:two", "concat", "subtitle"
    ]
    assert state.ok
    assert state.layout_attempts == 2


# ---------------------------------------------------------------------------
# The budget
# ---------------------------------------------------------------------------


def test_the_budget_stops_the_run_and_keeps_what_rendered(tmp_path):
    # Two cents of planning, then four scenes at ten cents, against a quarter.
    # The check runs before each scene, so a, b and c are attempted and the
    # run stops with d unattempted -- the lesson is still made from what got
    # through.
    stub = Stub(plan=lesson("a", "b", "c", "d"), doc=document("a", "b", "c", "d"),
                scene_cost=0.10)
    state = run(stub, tmp_path, budget=agent.Budget(dollars=0.25))

    assert stub.climbed == ["a", "b", "c"]
    assert "concat" in stub.calls
    assert state.ok
    assert state.video is not None
    assert "budget" in state.stopped
    assert "1 scene(s) not attempted" in state.stopped


def test_the_budget_is_handed_down_so_a_scene_cannot_overshoot_it(tmp_path):
    # Checking between scenes is not enough on its own: one scene is twelve
    # calls, so a ceiling noticed only at the boundary can be passed by a whole
    # scene's worth. The ladder gets a predicate and checks it between its own
    # calls.
    seen = []
    stub = Stub()
    inner = stub.climb
    stub.climb = lambda scene, workdir, client, **kw: (
        seen.append(kw.get("stop")) or inner(scene, workdir, client, **kw)
    )
    run(stub, tmp_path)

    assert all(callable(s) for s in seen)
    assert seen and seen[0]() == ""  # nothing breached yet


def test_running_out_of_budget_is_a_stop_not_a_crash(tmp_path):
    stub = Stub(plan=lesson("a", "b"), doc=document("a", "b"), scene_cost=1.0)
    state = run(stub, tmp_path, budget=agent.Budget(dollars=0.5))

    assert state.ok  # one scene rendered, so there is a lesson
    assert len(state.rendered) == 1
    assert state.stopped


def test_a_call_ceiling_stops_a_run_that_is_still_cheap(tmp_path):
    stub = Stub(plan=lesson("a", "b", "c"), doc=document("a", "b", "c"), scene_cost=0.0)
    state = run(stub, tmp_path, budget=agent.Budget(calls=3))

    # plan and layout are two of the three.
    assert stub.climbed == ["a"]
    assert "model calls" in state.stopped


def test_a_time_ceiling_stops_a_run_that_is_cheap_and_quiet(tmp_path):
    stub = Stub(scene_cost=0.0)
    state = run(stub, tmp_path, budget=agent.Budget(seconds=0.0))

    assert stub.calls == ["plan"]  # the layout is never attempted
    assert "elapsed" in state.stopped


def test_the_three_ceilings_are_independent():
    budget = agent.Budget(dollars=1.0, calls=10, seconds=100.0)

    assert not budget.breach(spent=0.5, calls=5, elapsed=50.0)
    assert "spent" in budget.breach(spent=1.0, calls=0, elapsed=0.0)
    assert "model calls" in budget.breach(spent=0.0, calls=10, elapsed=0.0)
    assert "elapsed" in budget.breach(spent=0.0, calls=0, elapsed=100.0)


def test_the_default_budget_covers_the_ladders_worst_case_for_a_small_lesson():
    # Twelve calls a scene at roughly two cents is about a dollar and a half
    # for six scenes; the default has to be at least that or a normal lesson
    # stops halfway.
    worst = (1 + repair.SIMPLIFY_ROUNDS) * (1 + repair.REPAIR_ROUNDS)
    assert worst == 12
    assert agent.DEFAULT_DOLLARS >= 6 * worst * 0.02


# ---------------------------------------------------------------------------
# What survives a partial failure
# ---------------------------------------------------------------------------


def test_one_dropped_scene_costs_one_scene(tmp_path):
    stub = Stub(plan=lesson("a", "b", "c"), doc=document("a", "b", "c"), fails={"b"})
    state = run(stub, tmp_path)

    assert state.ok
    assert [o.scene_id for o in state.rendered] == ["a", "c"]
    assert [o.scene_id for o in state.dropped] == ["b"]


def test_every_scene_failing_is_reported_rather_than_joined(tmp_path):
    stub = Stub(fails={"one", "two"})
    state = run(stub, tmp_path)

    assert "concat" not in stub.calls
    assert not state.ok
    assert state.error == "no scene rendered"


def test_a_join_that_fails_says_so(tmp_path):
    state = run(Stub(join_ok=False), tmp_path)

    assert not state.ok
    assert "would not join" in state.error


def test_the_summary_names_what_was_dropped_and_what_it_cost(tmp_path):
    stub = Stub(plan=lesson("a", "b"), doc=document("a", "b"), fails={"b"})
    text = run(stub, tmp_path).summary()

    assert "1/2 scenes" in text
    assert "DROP b" in text
    assert "$" in text


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_the_summary_reads_the_run_rather_than_re_deriving_it(tmp_path):
    stub = Stub(plan=lesson("a", "b", "c", "d"), doc=document("a", "b", "c", "d"),
                fails={"d"})
    report = metrics.summarise(run(stub, tmp_path))

    assert report.scenes == 4 and report.rendered == 3 and report.dropped == 1
    assert report.drop_rate == pytest.approx(0.25)
    assert report.code_first_try == pytest.approx(0.75)
    assert report.ir_clean_first_try is True


def test_a_rate_over_nothing_is_unknown_rather_than_zero(tmp_path):
    # A run with no scenes has no first-attempt success rate, and 0% would
    # read as a catastrophic result rather than an absent one.
    empty = agent.Run(prompt="x")
    report = metrics.summarise(empty)

    assert report.code_first_try is None
    assert report.drop_rate is None
    assert "n/a" in report.render()


def test_first_attempt_means_before_any_simplification():
    # A scene that only compiled once reduced did not pass on the first
    # attempt, and counting it as though it had is how a ladder makes itself
    # look unnecessary.
    scene = document("a").scenes[0]
    outcome = repair.Outcome(
        scene_id="a",
        status=repair.RENDERED,
        scene=scene,
        attempts=(
            repair.Attempt(repair.CODEGEN, 0, 0, False),
            repair.Attempt(repair.SIMPLIFIED, 0, 0, True, detail="flatten"),
            repair.Attempt(repair.CODEGEN, 1, 0, True),
        ),
        video=Path("a.mp4"),
    )
    report = metrics.summarise(agent.Run(prompt="x", scenes=(outcome,)))

    assert report.code_first_try == 0.0
    assert report.escalated == 1.0


def test_the_repair_distribution_counts_rounds_to_success():
    def outcome(scene_id, repairs):
        return repair.Outcome(
            scene_id=scene_id,
            status=repair.RENDERED,
            scene=document(scene_id).scenes[0],
            attempts=(repair.Attempt(repair.CODEGEN, 0, 0, repairs == 0),)
            + tuple(
                repair.Attempt(repair.REPAIRED, 0, n + 1, n == repairs - 1)
                for n in range(repairs)
            ),
            video=Path("x.mp4"),
        )

    report = metrics.summarise(
        agent.Run(prompt="x", scenes=(outcome("a", 0), outcome("b", 2), outcome("c", 2)))
    )
    assert report.repair_rounds == {0: 1, 2: 2}


def test_the_record_keeps_the_log_the_numbers_came_from(tmp_path):
    stub = Stub(fails={"two"})
    state = run(stub, tmp_path)
    path = metrics.record(state, tmp_path / "run.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["prompt"] == "explain something"
    assert payload["summary"]["scenes"] == 2
    # A summary you cannot recompute is a claim rather than a measurement.
    assert [row["scene"] for row in payload["attempts"]] == ["one", "two"]
    assert payload["scenes"][1]["status"] == repair.DROPPED


def test_the_rendered_summary_is_readable(tmp_path):
    text = metrics.summarise(run(Stub(), tmp_path)).render()

    assert "scenes" in text and "cost" in text and "code passed L2 first" in text


def test_a_stage_that_raises_is_a_failed_run_not_a_traceback(tmp_path):
    # The ladder turns a refusal or a truncation into an outcome; plan and
    # layout have no ladder around them, so the agent is where that happens.
    # A real five-scene layout exhausted max_tokens on the first full run.
    class Boom(Stub):
        def layout(self, client, lesson_, **kwargs):
            self.calls.append("layout")
            raise llm.Truncated("output hit max_tokens after 16000 tokens")

    stub = Boom()
    state = run(stub, tmp_path)

    assert not state.ok
    assert "the layout could not be generated" in state.error
    assert "max_tokens" in state.error
    assert stub.calls == ["plan", "layout"]


def test_a_plan_that_raises_stops_before_the_layout(tmp_path):
    class Boom(Stub):
        def plan(self, client, prompt, **kwargs):
            self.calls.append("plan")
            raise llm.Refused("declined", "cyber")

    stub = Boom()
    state = run(stub, tmp_path)

    assert stub.calls == ["plan"]
    assert "the plan could not be generated" in state.error


def test_a_large_ceiling_streams_rather_than_risking_a_timeout():
    from explainer import layout as layout_stage

    # Thinking is billed as output, so a layout can exhaust the default before
    # writing anything; the larger ceiling is only safe because it streams.
    assert layout_stage.MAX_TOKENS > llm.STREAM_ABOVE


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def test_the_narration_reaches_the_viewer(tmp_path):
    # The words decided how long every scene runs and then, until now, never
    # reached anyone. A silent animation is not an explainer.
    stub = Stub()
    state = run(stub, tmp_path)

    assert state.subtitled
    # Both scenes' words, all of them, and none invented.
    assert " ".join(cue.text for cue in stub.cues).split() == ["word"] * 80


def test_only_the_scenes_that_rendered_are_subtitled(tmp_path):
    # A dropped scene has no clip to time its lines against, and captioning it
    # would slide every later line out from under its animation.
    stub = Stub(plan=lesson("a", "b", "c"), doc=document("a", "b", "c"), fails={"b"})
    run(stub, tmp_path)

    # Two clips of sixteen seconds. Captioning the dropped scene as well would
    # run the last line out to forty-eight.
    assert len(" ".join(cue.text for cue in stub.cues).split()) == 80
    assert stub.cues[-1].end == pytest.approx(32.0)


def test_losing_the_subtitles_does_not_lose_the_lesson(tmp_path):
    state = run(Stub(burn_ok=False), tmp_path)

    assert state.ok
    assert state.video is not None
    assert not state.subtitled
    assert "subtitles were not burned in" in state.error
    assert "no subtitles" in state.summary()


def test_subtitles_can_be_turned_off(tmp_path):
    stub = Stub()
    state = run(stub, tmp_path, subtitles=False)

    assert "subtitle" not in stub.calls
    assert state.ok and not state.subtitled


# ---------------------------------------------------------------------------
# Layout history
# ---------------------------------------------------------------------------


def test_a_layout_fixed_on_the_retry_still_records_what_was_wrong(tmp_path):
    # Before this, only the final attempt's issues survived, so a retry that
    # worked erased the evidence of what the rules had caught.
    first = (ir.Issue("non-empty", "a scene needs a step", "scenes[0]"),)
    stub = Stub(layout_issues=first, retry_issues=())
    state = run(stub, tmp_path)

    assert state.ok
    assert state.layout_history == (first, ())
    assert "non-empty" in {i.rule for i in agent.issues_of(state)}
    assert metrics.summarise(state).rules_fired == {"non-empty": 1}


def test_the_record_carries_every_layout_attempt(tmp_path):
    first = (ir.Issue("non-empty", "x", "scenes[0]"),)
    state = run(Stub(layout_issues=first, retry_issues=()), tmp_path)
    payload = json.loads(metrics.record(state, tmp_path / "run.json").read_text(encoding="utf-8"))

    assert [len(a) for a in payload["layout_attempts"]] == [1, 0]
    assert payload["layout_attempts"][0][0]["rule"] == "non-empty"


def test_the_record_keeps_the_layout_the_rules_judged(tmp_path):
    # A warning that cannot be re-examined is a claim. Re-checking the extent
    # estimator against past runs needed the documents, and they were not
    # there -- the layouts had to be reconstructed from generated Python.
    state = run(Stub(), tmp_path)
    payload = json.loads(
        metrics.record(state, tmp_path / "run.json").read_text(encoding="utf-8")
    )

    assert payload["document"]["scenes"][0]["id"] == "one"
    assert payload["document"]["scenes"][0]["objects"][0]["type"] == "text"
    # Round-trips, so the rules can be re-run over it exactly as they were.
    assert ir.parse_document(payload["document"]).ok


# ---------------------------------------------------------------------------
# Speech, and the order that makes it line up
# ---------------------------------------------------------------------------


def test_the_narration_is_spoken_before_the_layout_is_asked_for(tmp_path):
    # The whole reason the audio lines up. Without speech the word count
    # estimates how long a scene should run; with it the synthesiser decides,
    # and the layout has to be told before it spends its budget rather than
    # after. Speaking last would leave the animation and the voice two
    # independent lengths, reconcilable only by stretching one of them.
    stub = Stub()
    run(stub, tmp_path, audio=True)

    assert stub.calls.index("speak") < stub.calls.index("layout")
    assert stub.calls[:3] == ["plan", "speak", "layout"]


def test_the_layout_is_given_the_measured_length_not_the_estimate(tmp_path):
    captured = []
    stub = Stub(speech_seconds=31.7)
    inner = stub.layout

    def watch(client, lesson_, **kwargs):
        captured.append(lesson_)
        return inner(client, lesson_, **kwargs)

    stub.layout = watch
    run(stub, tmp_path, audio=True)

    assert [s.seconds for s in captured[0].scenes] == [31.7, 31.7]
    assert captured[0].measured
    # 40 words at the estimated rate is nowhere near 31.7s, so this is the
    # measurement rather than the guess.
    assert captured[0].scenes[0].spoken_seconds == 31.7


def test_the_narration_is_spoken_once_however_many_times_a_scene_is_repaired(tmp_path):
    # The words are final once the plan is. Re-synthesising per layout attempt
    # would re-time every scene against audio the last attempt was fitted to.
    stub = Stub(layout_issues=(ir.Issue("non-empty", "x", "scenes[0]"),), retry_issues=())
    run(stub, tmp_path, audio=True)

    assert stub.calls.count("speak") == 1
    assert stub.calls.count("layout") == 1 and stub.calls.count("retry") == 1


def test_speech_can_be_turned_off_and_then_nothing_is_synthesised(tmp_path):
    stub = Stub()
    state = run(stub, tmp_path, audio=False)

    assert "speak" not in stub.calls
    assert "sound" not in stub.calls
    assert state.ok and not state.voiced
    assert state.spoken == {}


def test_the_voice_is_passed_through(tmp_path):
    stub = Stub()
    run(stub, tmp_path, audio=True, voice="en-GB-SoniaNeural")

    assert stub.voice == "en-GB-SoniaNeural"


# ---------------------------------------------------------------------------
# Sound on the finished lesson
# ---------------------------------------------------------------------------


def test_each_scenes_audio_is_placed_where_that_scene_starts(tmp_path):
    # Measured from the clips, not assumed: what a scene asked for and what
    # manim produced differ every time, and the gap puts the voice over the
    # wrong animation.
    stub = Stub(
        plan=lesson("a", "b", "c"),
        doc=document("a", "b", "c"),
        clip_seconds=12.0,
        # What `layout.fit_to_speech` is for: the clip and the voice agree.
        speech_seconds=12.0,
    )
    state = run(stub, tmp_path, audio=True)

    assert state.voiced
    assert [start for _, start in stub.placements] == pytest.approx([0.0, 12.0, 24.0])
    assert stub.total == pytest.approx(36.0)
    assert state.drift == pytest.approx(0.0)


def test_a_dropped_scene_leaves_no_gap_in_the_narration(tmp_path):
    # The dropped scene has no clip, so its words are never said over one. If
    # its audio were placed anyway every later scene would be a scene late.
    stub = Stub(plan=lesson("a", "b", "c"), doc=document("a", "b", "c"), fails={"b"},
                clip_seconds=10.0)
    state = run(stub, tmp_path, audio=True)

    assert [p.name for p, _ in stub.placements] == ["a.mp3", "c.mp3"]
    assert [start for _, start in stub.placements] == pytest.approx([0.0, 10.0])
    assert state.voiced


def test_the_sound_goes_on_after_the_subtitles_are_burned(tmp_path):
    # Burning re-encodes the picture; muxing copies it. The other order would
    # either lose the audio or pay for a second encode of the video.
    stub = Stub()
    run(stub, tmp_path, audio=True)

    assert stub.calls.index("subtitle") < stub.calls.index("sound")


def test_the_subtitles_are_timed_by_the_synthesiser_when_there_is_one(tmp_path):
    # Word timings, so a line lands on the syllable rather than on a share of
    # the scene worked out from how many words it has.
    stub = Stub(speech_seconds=10.0, clip_seconds=10.0)
    run(stub, tmp_path, audio=True)

    # Forty words spoken evenly over ten seconds; the first cue is the first
    # twelve of them, so it ends at 12/40 of the scene.
    assert stub.cues[0].end == pytest.approx(3.0)


def test_losing_the_voice_does_not_lose_the_lesson(tmp_path):
    stub = Stub(mux_ok=False)
    state = run(stub, tmp_path, audio=True)

    assert state.ok
    assert state.video is not None
    assert state.subtitled
    assert not state.voiced
    assert "the narration was not added" in state.error
    assert "no sound" in state.summary()


def test_a_synthesiser_that_will_not_speak_leaves_a_subtitled_silent_lesson(tmp_path):
    # edge-tts talks to an endpoint Microsoft does not promise to keep. When
    # it is gone the lesson is the one this pipeline made before it had audio,
    # which is a working lesson.
    stub = Stub(mute={"one"})
    state = run(stub, tmp_path, audio=True)

    assert state.ok
    assert state.subtitled and not state.voiced
    assert "sound" not in stub.calls
    assert "the lesson is silent" in state.error
    assert "the endpoint refused" in state.error


def test_one_scene_failing_to_speak_does_not_half_narrate_the_lesson(tmp_path):
    # Placing the scenes that did speak would leave the others silent, and a
    # viewer cannot tell a missing scene from a pause. All or none.
    stub = Stub(plan=lesson("a", "b", "c"), doc=document("a", "b", "c"), mute={"b"})
    state = run(stub, tmp_path, audio=True)

    assert not state.voiced
    assert stub.placements == []


def test_a_scene_that_could_not_be_spoken_is_timed_by_word_count_instead(tmp_path):
    # The estimate is what the pipeline had before it had a synthesiser, and
    # it still times the subtitles of a lesson that ends up silent.
    stub = Stub(mute={"one", "two"}, clip_seconds=10.0)
    run(stub, tmp_path, audio=True)

    assert stub.cues
    assert stub.cues[-1].end == pytest.approx(20.0)


def test_what_was_spoken_is_kept_whether_it_was_used_or_not(tmp_path):
    stub = Stub(mute={"one"})
    state = run(stub, tmp_path, audio=True)

    assert set(state.spoken) == {"one", "two"}
    assert not state.spoken["one"].ok
    assert state.spoken["two"].ok
    assert state.spoken_seconds == pytest.approx(16.0)


def test_the_gap_between_what_was_said_and_what_was_shown_is_recorded(tmp_path):
    # The number that says whether the layout took the measured duration
    # seriously. Positive means the voice outlasted the animation under it.
    stub = Stub(speech_seconds=18.0, clip_seconds=16.0)
    state = run(stub, tmp_path, audio=True)

    assert state.drift == pytest.approx(2.0)
    assert "+2.00s" in state.summary()


def test_an_animation_that_outlasts_its_narration_is_a_negative_gap(tmp_path):
    stub = Stub(speech_seconds=14.0, clip_seconds=20.0)
    state = run(stub, tmp_path, audio=True)

    assert state.drift == pytest.approx(-6.0)


def test_a_voiced_lesson_says_nothing_is_missing(tmp_path):
    state = run(Stub(), tmp_path, audio=True)

    assert state.voiced and state.subtitled
    assert "(no " not in state.summary()


def test_the_last_word_of_the_lesson_is_never_cut_off(tmp_path):
    # The track is capped so it cannot run arbitrarily past the picture. If
    # that cap were the video's length, a scene whose voice overran would lose
    # its closing syllable -- and the end of a lesson is the worst place to
    # lose one.
    stub = Stub(speech_seconds=20.0, clip_seconds=16.0)
    run(stub, tmp_path, audio=True)

    # Two scenes of sixteen seconds; the second's voice ends at 16 + 20 = 36.
    assert stub.total == pytest.approx(36.0)


def test_a_track_no_longer_than_the_picture_is_not_stretched_to_fit(tmp_path):
    stub = Stub(speech_seconds=10.0, clip_seconds=16.0)
    run(stub, tmp_path, audio=True)

    assert stub.total == pytest.approx(32.0)
