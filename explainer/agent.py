"""State, dispatch, and a budget that is enforced rather than hoped for.

The whole pipeline, driven: a prompt becomes a plan, the plan becomes a
layout, each scene climbs the escalation ladder, and what rendered is joined
into a lesson. `spine.py` does the middle of that without an agent around it
and stays that way -- it is the no-repair baseline the ablation needs.

## The budget is not optional and it is not advisory

The ladder bounds each scene at (1 + SIMPLIFY_ROUNDS) x (1 + REPAIR_ROUNDS)
model calls -- twelve at the defaults. Six scenes is seventy-two, and nothing
in the ladder knows that a lesson has six scenes. HANDOFF.md is blunt about
the consequence: an agent that can loop on a scene the model cannot get right
is a way to spend money overnight. So the budget is a constructor argument
with a default, checked *before* each stage that spends, and it carries three
independent ceilings -- dollars, calls and wall-clock -- because a run can
blow through any one of them without touching the others.

**Running out of budget is a stop, not a crash.** The scenes that rendered
before the ceiling was reached are still joined into a lesson, and the run
says where it stopped and why. That is the same graceful degradation a dropped
scene gets, applied one level up: four scenes and a note beats an exception
and nothing.

## Order follows cost

The rules run before the model does. A plan whose scenes are unusable never
reaches the layout stage, and a layout with errors is sent back with the
issues rather than rendered -- twice at most, because a third attempt at the
same failure is a third bill for the same answer.

Every stage is injected, so the state machine and the budget can be tested
without a model, a subprocess or a renderer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

from . import ir, layout, llm, plan as planning, repair, speech, video
from .ir import Issue

#: What one lesson may cost before the run stops and keeps what it has. The
#: ladder's worst case is twelve calls a scene, so a six-scene lesson can reach
#: seventy-two; at roughly two cents a call that is about a dollar and a half.
DEFAULT_DOLLARS = 2.00
DEFAULT_CALLS = 100
DEFAULT_SECONDS = 1800.0

#: Attempts at the layout before giving up on it. A third attempt at the same
#: failure is a third bill for the same answer.
LAYOUT_ROUNDS = 2


@dataclass(frozen=True)
class Budget:
    """Three independent ceilings. A run can breach any one alone."""

    dollars: float = DEFAULT_DOLLARS
    calls: int = DEFAULT_CALLS
    seconds: float = DEFAULT_SECONDS

    def breach(self, *, spent: float, calls: int, elapsed: float) -> str:
        if spent >= self.dollars:
            return f"budget: ${spent:.2f} of ${self.dollars:.2f} spent"
        if calls >= self.calls:
            return f"budget: {calls} of {self.calls} model calls used"
        if elapsed >= self.seconds:
            return f"budget: {elapsed:.0f}s of {self.seconds:.0f}s elapsed"
        return ""


@dataclass(frozen=True)
class Run:
    """Everything that happened, kept whether it worked or not."""

    prompt: str
    ok: bool = False
    video: Path | None = None
    plan: planning.LessonPlan | None = None
    document: ir.Document | None = None
    scenes: tuple[repair.Outcome, ...] = ()
    plan_issues: tuple[Issue, ...] = ()
    layout_issues: tuple[Issue, ...] = ()
    layout_attempts: int = 0
    # Every attempt's issues, not only the last. A layout fixed on its retry
    # otherwise records nothing about what was wrong the first time -- which
    # is exactly the "which rule caught which defect" the metrics exist for.
    layout_history: tuple[tuple[Issue, ...], ...] = ()
    usage: llm.Usage = field(default_factory=llm.Usage)
    seconds: float = 0.0
    stopped: str = ""
    error: str = ""
    subtitled: bool = False
    #: Each scene's narration as audio, keyed by scene id. Empty when speech
    #: was switched off; present but not `ok` when it was tried and failed.
    spoken: dict[str, speech.Spoken] = field(default_factory=dict)
    voiced: bool = False
    #: The worst gap between a scene's speech and the clip under it, in
    #: seconds. Positive means the voice outlasted the animation. This is the
    #: number that says whether the layout took the measured duration
    #: seriously, so it is recorded rather than checked and forgotten.
    drift: float = 0.0

    @property
    def spoken_seconds(self) -> float:
        return sum(s.seconds for s in self.spoken.values() if s.ok)

    @property
    def rendered(self) -> tuple[repair.Outcome, ...]:
        return tuple(s for s in self.scenes if s.rendered)

    @property
    def dropped(self) -> tuple[repair.Outcome, ...]:
        return tuple(s for s in self.scenes if not s.rendered)

    def dollars(self) -> float:
        return self.usage.dollars()

    def summary(self) -> str:
        head = (
            f"{len(self.rendered)}/{len(self.scenes)} scenes in {self.seconds:.0f}s "
            f"for ${self.dollars():.4f} over {self.usage.calls} calls"
        )
        lines = [head]
        for outcome in self.scenes:
            mark = "ok  " if outcome.rendered else "DROP"
            line = f"  {mark} {outcome.scene_id:16} {outcome.trail()}"
            if outcome.error:
                line += f"  -- {outcome.error.strip().splitlines()[-1][:90]}"
            lines.append(line)
        if self.stopped:
            lines.append(f"  !! stopped early -- {self.stopped}")
        if self.video:
            missing = [
                name
                for name, present in (("subtitles", self.subtitled), ("sound", self.voiced))
                if not present
            ]
            lines.append(
                f"  -> {self.video}"
                + (f"  (no {' or '.join(missing)})" if missing else "")
            )
            if self.voiced and self.drift:
                lines.append(f"     worst audio/video gap {self.drift:+.2f}s")
        elif self.error:
            lines.append(f"  !! {self.error}")
        return "\n".join(lines)


@dataclass(frozen=True)
class Stages:
    """Everything the agent dispatches to, injectable for testing.

    Speech sits between `plan` and `layout` rather than at the end, which is
    the whole reason the audio lines up. See `run`.
    """

    plan: Callable[..., tuple[planning.LessonPlan | None, tuple[Issue, ...], llm.Reply]]
    speak: Callable[..., dict[str, speech.Spoken]]
    layout: Callable[..., tuple[ir.Document | None, tuple[Issue, ...], llm.Reply]]
    retry: Callable[..., tuple[ir.Document | None, tuple[Issue, ...], llm.Reply]]
    climb: Callable[..., repair.Outcome]
    concat: Callable[..., object]
    measure: Callable[..., float | None]
    subtitle: Callable[..., object]
    sound: Callable[..., object]


def default_stages() -> Stages:
    return Stages(
        plan=planning.make,
        speak=speech.narrate,
        layout=layout.make,
        retry=layout.retry,
        climb=repair.climb,
        concat=video.concat,
        measure=video.duration,
        subtitle=video.subtitle,
        sound=video.soundtrack,
    )


def run(
    prompt: str,
    workdir: Path,
    client: llm.Client | None = None,
    *,
    budget: Budget | None = None,
    quality: str = "l",
    subtitles: bool = True,
    audio: bool = True,
    voice: str = speech.VOICE,
    repair_rounds: int = repair.REPAIR_ROUNDS,
    simplify_rounds: int = repair.SIMPLIFY_ROUNDS,
    stages: Stages | None = None,
) -> Run:
    """Take *prompt* as far as the budget allows, and report all of it."""
    stages = stages or default_stages()
    budget = budget or Budget()
    client = client or llm.Client()
    workdir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    state = Run(prompt=prompt)

    def spent() -> str:
        return budget.breach(
            spent=state.usage.dollars(),
            calls=state.usage.calls,
            elapsed=time.perf_counter() - started,
        )

    def close(**changes) -> Run:
        return replace(state, seconds=time.perf_counter() - started, **changes)

    # -- plan ---------------------------------------------------------------
    # A stage that raises is a run that failed, not a traceback. The ladder
    # already treats a refusal or a truncation as an outcome; these two stages
    # have no ladder around them, so the agent is where that happens.
    try:
        lesson, issues, reply = stages.plan(client, prompt)
    except llm.LLMError as exc:
        return close(error=f"the plan could not be generated: {exc}")
    state = replace(state, usage=state.usage + reply.usage, plan=lesson, plan_issues=issues)
    if lesson is None or planning.errors(issues):
        return close(error=f"the plan was rejected:\n{planning.report(issues)}")

    # -- speech -------------------------------------------------------------
    # Here, and not at the end, because speech inverts the timing. Without it
    # the narration's word count *estimates* how long a scene should run; with
    # it the synthesiser *decides*, and the layout has to be told before it
    # spends its budget rather than after. The words are final once the plan
    # is, so nothing is re-synthesised when a layout is retried or a scene
    # repaired -- this runs once, before either can happen.
    #
    # The cost of getting this wrong is not an error, which is why it is worth
    # stating: stretching video to meet audio afterwards is a quality loss, and
    # letting them disagree is a lesson where the voice says "now square both
    # sides" over an animation that did it eight seconds ago.
    spoken: dict[str, speech.Spoken] = {}
    if audio:
        spoken = stages.speak(
            [(scene.id, scene.narration) for scene in lesson.scenes],
            workdir / "audio",
            voice=voice,
        )
        state = replace(state, spoken=spoken)
        if all(spoken.get(scene.id, speech.Spoken(False)).ok for scene in lesson.scenes):
            lesson = replace(
                lesson,
                scenes=tuple(
                    replace(scene, seconds=spoken[scene.id].seconds)
                    for scene in lesson.scenes
                ),
            )
            state = replace(state, plan=lesson)

    # -- layout -------------------------------------------------------------
    document: ir.Document | None = None
    layout_issues: tuple[Issue, ...] = ()
    history: tuple[tuple[Issue, ...], ...] = ()
    for attempt in range(1, LAYOUT_ROUNDS + 1):
        if (breach := spent()):
            return close(stopped=breach, error="stopped before the layout was usable")
        try:
            if attempt == 1:
                document, layout_issues, reply = stages.layout(client, lesson)
            else:
                document, layout_issues, reply = stages.retry(
                    client, lesson, document, layout_issues
                )
        except llm.LLMError as exc:
            return close(error=f"the layout could not be generated: {exc}")
        state = replace(
            state,
            usage=state.usage + reply.usage,
            document=document,
            layout_issues=layout_issues,
            layout_attempts=attempt,
            layout_history=(history := history + (layout_issues,)),
        )
        if document is not None and not layout.errors(layout_issues):
            break
    else:
        return close(
            error=(
                f"the layout still had errors after {LAYOUT_ROUNDS} attempts:\n"
                f"{layout.report(layout.errors(layout_issues))}"
            )
        )

    # -- scenes -------------------------------------------------------------
    outcomes: list[repair.Outcome] = []
    stopped = ""
    for scene in document.scenes:
        if (breach := spent()):
            stopped = f"{breach}; {len(document.scenes) - len(outcomes)} scene(s) not attempted"
            break
        outcome = stages.climb(
            scene,
            workdir / scene.id,
            client,
            title=document.title,
            quality=quality,
            repair_rounds=repair_rounds,
            simplify_rounds=simplify_rounds,
            # Checked between the ladder's own calls, so the ceiling is a cap
            # rather than a thing noticed a scene late.
            stop=spent,
        )
        outcomes.append(outcome)
        state = replace(
            state, scenes=tuple(outcomes), usage=state.usage + outcome.usage
        )

    state = replace(state, scenes=tuple(outcomes), stopped=stopped)

    # -- join ---------------------------------------------------------------
    clips = [o.video for o in outcomes if o.video is not None]
    if not clips:
        return close(error="no scene rendered")

    joined = stages.concat(clips, workdir / "lesson.mp4")
    if not getattr(joined, "ok", False):
        return close(error=f"scenes rendered but would not join: {joined.error}")
    lesson = getattr(joined, "output", None)

    if lesson is None:
        return close(error="the scenes joined but produced no file")

    # -- where each scene starts --------------------------------------------
    # Measured from the clips rather than taken from the IR: what a scene asked
    # for and what manim produced differ slightly every time, and over six
    # scenes that drift is enough to put a subtitle under the wrong animation
    # and the voice over the wrong one. Probed once, and used for both.
    rendered = [o for o in outcomes if o.video]
    starts: list[float] = []
    at = 0.0
    for outcome in rendered:
        starts.append(at)
        at += max(stages.measure(outcome.video) or 0.0, 0.0)
    total = at

    # -- subtitles ----------------------------------------------------------
    # The words reach a viewer who has the sound off, or who is reading a
    # formula rather than parsing it by ear. Burned in either way.
    #
    # When the narration was spoken, the synthesiser said when each word was
    # said, so the cues land on the syllable. Otherwise they are shared out
    # across the scene by word count, which is the best an estimate can do: a
    # long word and a short one do not take the same time to say.
    voiced_all = bool(spoken) and all(
        spoken.get(o.scene.id, speech.Spoken(False)).ok for o in rendered
    )
    burned = False
    notes: list[str] = []
    if subtitles:
        if voiced_all:
            cues = video.spoken_cues(
                [(spoken[o.scene.id].words, start) for o, start in zip(rendered, starts)]
            )
        else:
            cues = video.cues_from(
                [
                    (o.scene.narration, (end - start))
                    for o, start, end in zip(rendered, starts, starts[1:] + [total])
                ]
            )
        result = stages.subtitle(lesson, cues)
        burned = bool(getattr(result, "ok", False))
        if not burned:
            # Losing the subtitles is not losing the lesson.
            notes.append(f"subtitles were not burned in: {getattr(result, 'error', '')}")

    # -- sound ---------------------------------------------------------------
    # After the burn, because burning re-encodes the video and this does not.
    voiced = False
    drift = 0.0
    if voiced_all:
        drift = max(
            (
                spoken[o.scene.id].seconds - (end - start)
                for o, start, end in zip(rendered, starts, starts[1:] + [total])
            ),
            default=0.0,
        )
        # To the end of the picture or the end of the sentence, whichever is
        # later. `fit_to_speech` should make these the same to within a frame,
        # but if it has not, a track cut at the video's length would take the
        # last syllable of the lesson with it -- and a second of audio past
        # the final frame is much the smaller fault.
        result = stages.sound(
            lesson,
            [(spoken[o.scene.id].path, start) for o, start in zip(rendered, starts)],
            total=max(
                total,
                max(
                    (start + spoken[o.scene.id].seconds
                     for o, start in zip(rendered, starts)),
                    default=0.0,
                ),
            ),
        )
        voiced = bool(getattr(result, "ok", False))
        if not voiced:
            notes.append(f"the narration was not added: {getattr(result, 'error', '')}")
    elif audio:
        failed = [
            s.error for s in spoken.values() if not s.ok
        ] or ["no narration was synthesised"]
        notes.append(f"the lesson is silent: {failed[0]}")

    return close(
        ok=True,
        video=lesson,
        subtitled=burned,
        voiced=voiced,
        drift=drift,
        error="; ".join(notes),
    )


def issues_of(state: Run) -> tuple[Issue, ...]:
    """Every issue the run recorded, in the order the stages produced them.

    Across every layout attempt, so a defect the retry fixed still counts as
    one the rules caught. Falls back to the final attempt for a Run built
    without a history.
    """
    layouts = (
        tuple(i for attempt in state.layout_history for i in attempt)
        if state.layout_history
        else tuple(state.layout_issues)
    )
    return tuple(state.plan_issues) + layouts
