"""One IR document to one MP4, with no agent in the way.

The milestone the handoff notes recommend: a document goes in, codegen runs
per scene, each result is validated and rendered, and the clips are joined.
There is no planner, no repair, no escalation and no budget -- those arrive in
their own commits, and each of them wraps this rather than replacing it.

**Graceful degradation is here from the first version, not bolted on later.**
A scene whose code will not run is recorded and dropped, and the lesson is
still assembled from the rest. That is the promise in ARCHITECTURE.md, and it
is worth having before the repair loop exists rather than after: it means the
loop is an improvement on a working pipeline instead of the thing holding one
together. A run reports what it dropped and why.

Order of operations follows the cost. The IR rules run first, in microseconds,
and an document with errors never reaches the model at all. Code is checked to
L2 and rendered once, after it is known to execute -- the tier measurements in
HANDOFF.md are what that ordering is drawn from.

Each scene gets its own working directory, which is what makes the failures
independent and what will let the renders run in parallel later.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import codegen, ir, ir_rules, llm, video
from .validate import Check, Level, render, validate

RENDERED = "rendered"
DROPPED = "dropped"


@dataclass(frozen=True)
class SceneOutcome:
    scene_id: str
    status: str
    seconds: float = 0.0
    usage: llm.Usage = field(default_factory=llm.Usage)
    code: str = ""
    check: Check | None = None
    video: Path | None = None
    error: str = ""


@dataclass(frozen=True)
class Run:
    ok: bool
    scenes: tuple[SceneOutcome, ...] = ()
    video: Path | None = None
    issues: tuple[ir.Issue, ...] = ()
    usage: llm.Usage = field(default_factory=llm.Usage)
    seconds: float = 0.0
    error: str = ""

    @property
    def rendered(self) -> tuple[SceneOutcome, ...]:
        return tuple(s for s in self.scenes if s.status == RENDERED)

    @property
    def dropped(self) -> tuple[SceneOutcome, ...]:
        return tuple(s for s in self.scenes if s.status == DROPPED)

    def dollars(self, model: str = llm.MODEL) -> float:
        return self.usage.dollars(model)

    def summary(self) -> str:
        lines = [
            f"{len(self.rendered)}/{len(self.scenes)} scenes rendered "
            f"in {self.seconds:.1f}s for ${self.dollars():.4f}"
        ]
        for outcome in self.scenes:
            mark = "ok  " if outcome.status == RENDERED else "DROP"
            line = f"  {mark} {outcome.scene_id} ({outcome.seconds:.1f}s)"
            if outcome.error:
                line += f" -- {outcome.error.strip().splitlines()[-1][:120]}"
            lines.append(line)
        if self.video:
            lines.append(f"  -> {self.video}")
        elif self.error:
            lines.append(f"  !! {self.error}")
        return "\n".join(lines)


def render_document(
    document: ir.Document,
    workdir: Path,
    client: llm.Client | None = None,
    *,
    quality: str = "l",
) -> Run:
    """Take *document* as far as it goes, and report what did not make it."""
    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    client = client or llm.Client()

    # Microseconds, and no model call: a document with errors should never
    # cost a token.
    issues = ir_rules.check_document(document)
    if ir_rules.errors(issues):
        return Run(
            ok=False,
            issues=issues,
            seconds=time.perf_counter() - started,
            error=f"the IR has {len(ir_rules.errors(issues))} error(s); nothing was generated",
        )

    outcomes = []
    total = llm.Usage()
    for scene in document.scenes:
        outcome = _one_scene(scene, workdir / scene.id, client, document.title, quality)
        outcomes.append(outcome)
        total = total + outcome.usage

    clips = [o.video for o in outcomes if o.video is not None]
    lesson: Path | None = None
    error = ""
    if clips:
        joined = video.concat(clips, workdir / "lesson.mp4")
        if joined.ok:
            lesson = joined.output
        else:
            error = f"scenes rendered but would not join: {joined.error}"
    else:
        error = "no scene rendered"

    return Run(
        ok=lesson is not None,
        scenes=tuple(outcomes),
        video=lesson,
        issues=issues,
        usage=total,
        seconds=time.perf_counter() - started,
        error=error,
    )


def _one_scene(
    scene: ir.Scene,
    workdir: Path,
    client: llm.Client,
    title: str,
    quality: str,
) -> SceneOutcome:
    started = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        code, reply = codegen.generate(client, scene, title=title)
    except llm.LLMError as exc:
        return SceneOutcome(
            scene.id,
            DROPPED,
            seconds=time.perf_counter() - started,
            error=f"generation failed: {exc}",
        )

    checked = validate(workdir, code, up_to=Level.DRYRUN)
    if checked.failed:
        return SceneOutcome(
            scene.id,
            DROPPED,
            seconds=time.perf_counter() - started,
            usage=reply.usage,
            code=code,
            check=checked,
            error=f"L{int(checked.level)} {checked.level.name}: {checked.error}",
        )

    produced = render(workdir, checked.scene_name or "", quality=quality)
    if produced.failed:
        return SceneOutcome(
            scene.id,
            DROPPED,
            seconds=time.perf_counter() - started,
            usage=reply.usage,
            code=code,
            check=produced,
            error=f"render: {produced.error}",
        )

    return SceneOutcome(
        scene.id,
        RENDERED,
        seconds=time.perf_counter() - started,
        usage=reply.usage,
        code=code,
        check=produced,
        video=produced.output,
    )


def render_file(path: Path, workdir: Path, **kwargs) -> Run:
    """Read an IR document from *path* and render it."""
    result = ir.parse_json(path.read_text(encoding="utf-8"))
    if not result.ok or result.document is None:
        return Run(ok=False, issues=result.issues, error=f"unreadable IR:\n{result.report()}")
    return render_document(result.document, workdir, **kwargs)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: python -m explainer.spine <document.json> [workdir]\n"
            "  e.g. python -m explainer.spine docs/examples/pythagoras.json runs/spine",
            file=sys.stderr,
        )
        return 2
    document = Path(argv[0])
    workdir = Path(argv[1]) if len(argv) > 1 else Path("runs") / document.stem
    run = render_file(document, workdir)
    print(run.summary())
    if run.issues:
        print(ir_rules.report(run.issues))
    (workdir / "run.json").write_text(
        json.dumps(
            {
                "ok": run.ok,
                "video": str(run.video) if run.video else None,
                "seconds": round(run.seconds, 2),
                "dollars": round(run.dollars(), 4),
                "scenes": [
                    {
                        "id": s.scene_id,
                        "status": s.status,
                        "seconds": round(s.seconds, 2),
                        "error": s.error,
                    }
                    for s in run.scenes
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if run.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
