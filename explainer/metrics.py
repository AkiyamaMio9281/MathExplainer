"""The numbers, and the log they are computed from.

ARCHITECTURE.md lists what to record and says why: the metrics are the
research output, and without them this is a demo. This module turns a run into
those numbers and writes the raw attempt log beside them, because a summary
you cannot recompute is a claim rather than a measurement -- and the questions
worth asking of a run are rarely the ones anticipated when it was written.

Two conventions keep the numbers honest.

**A rate over nothing is not zero, it is unknown.** A run with no scenes has no
first-attempt success rate, and reporting 0% would read as a catastrophic
result rather than an absent one. Every rate here is `None` when its
denominator is empty, and renders as `n/a`.

**First attempt means the first attempt at the scene the layout asked for.**
Not the first attempt after a simplification: a scene that only compiled once
it had been reduced did not pass on the first attempt, and counting it as
though it had is how an escalation ladder makes itself look unnecessary.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import agent, ir_rules, llm, repair


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


@dataclass(frozen=True)
class Summary:
    """One run, as the table in ARCHITECTURE.md."""

    scenes: int = 0
    rendered: int = 0
    dropped: int = 0

    ir_clean_first_try: bool | None = None
    layout_attempts: int = 0

    code_first_try: float | None = None
    repair_rounds: dict[int, int] = field(default_factory=dict)
    escalated: float | None = None
    drop_rate: float | None = None

    rules_fired: dict[str, int] = field(default_factory=dict)

    render_seconds: float = 0.0
    seconds: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_rate: float = 0.0
    dollars: float = 0.0
    dollars_per_scene: float | None = None
    stopped: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        lines = [
            f"scenes                  {self.rendered}/{self.scenes} rendered, "
            f"{self.dropped} dropped ({_percent(self.drop_rate)})",
            f"IR clean first try      "
            f"{'yes' if self.ir_clean_first_try else 'no' if self.ir_clean_first_try is not None else 'n/a'}"
            f"  ({self.layout_attempts} layout attempt(s))",
            f"code passed L2 first    {_percent(self.code_first_try)}",
            f"escalated to simplify   {_percent(self.escalated)}",
            f"repair rounds           {self._distribution()}",
            f"rules fired             {self._rules()}",
            f"time                    {self.seconds:.0f}s total, "
            f"{self.render_seconds:.0f}s rendering",
            f"cost                    ${self.dollars:.4f} over {self.calls} calls"
            + (
                f", ${self.dollars_per_scene:.4f} a rendered scene"
                if self.dollars_per_scene is not None
                else ""
            ),
            f"tokens                  {self.input_tokens} in, {self.output_tokens} out, "
            f"{self.cache_hit_rate:.0%} of the prompt from cache",
        ]
        if self.stopped:
            lines.append(f"stopped early           {self.stopped}")
        return "\n".join(lines)

    def _distribution(self) -> str:
        if not self.repair_rounds:
            return "n/a"
        return ", ".join(
            f"{rounds}: {count}" for rounds, count in sorted(self.repair_rounds.items())
        )

    def _rules(self) -> str:
        if not self.rules_fired:
            return "none"
        return ", ".join(
            f"{name} x{count}" for name, count in sorted(self.rules_fired.items())
        )


def summarise(state: agent.Run) -> Summary:
    """Read a run. Nothing here re-derives anything the run did not record."""
    scenes = state.scenes
    total = len(scenes)
    rendered = [s for s in scenes if s.rendered]

    # First attempt at the scene the layout asked for -- stage 0, round 0.
    def first_try(outcome: repair.Outcome) -> bool:
        return any(
            a.kind == repair.CODEGEN and a.stage == 0 and a.round == 0 and a.ok
            for a in outcome.attempts
        )

    rounds_to_success: Counter[int] = Counter()
    for outcome in rendered:
        rounds_to_success[outcome.repairs] += 1

    rules: Counter[str] = Counter()
    for issue in agent.issues_of(state):
        rules[issue.rule] += 1

    render_seconds = sum(
        getattr(o.check, "seconds", 0.0) or 0.0 for o in rendered
    )

    return Summary(
        scenes=total,
        rendered=len(rendered),
        dropped=total - len(rendered),
        ir_clean_first_try=(state.layout_attempts == 1) if state.layout_attempts else None,
        layout_attempts=state.layout_attempts,
        code_first_try=_rate(sum(1 for o in scenes if first_try(o)), total),
        repair_rounds=dict(rounds_to_success),
        escalated=_rate(sum(1 for o in scenes if o.simplifications), total),
        drop_rate=_rate(total - len(rendered), total),
        rules_fired=dict(rules),
        render_seconds=render_seconds,
        seconds=state.seconds,
        calls=state.usage.calls,
        input_tokens=state.usage.input_tokens
        + state.usage.cache_write_tokens
        + state.usage.cache_read_tokens,
        output_tokens=state.usage.output_tokens,
        cache_hit_rate=state.usage.cache_hit_rate,
        dollars=state.dollars(),
        dollars_per_scene=(state.dollars() / len(rendered)) if rendered else None,
        stopped=state.stopped,
    )


def attempt_log(state: agent.Run) -> list[dict[str, Any]]:
    """Every action, flat, so the numbers above can be recomputed."""
    rows = []
    for outcome in state.scenes:
        for index, attempt in enumerate(outcome.attempts):
            rows.append(
                {
                    "scene": outcome.scene_id,
                    "n": index,
                    "kind": attempt.kind,
                    "stage": attempt.stage,
                    "round": attempt.round,
                    "ok": attempt.ok,
                    "level": attempt.level,
                    "detail": attempt.detail,
                    "seconds": round(attempt.seconds, 3),
                    "calls": attempt.usage.calls,
                    "input_tokens": attempt.usage.input_tokens,
                    "output_tokens": attempt.usage.output_tokens,
                    "error": attempt.error[-300:],
                }
            )
    return rows


def record(state: agent.Run, path: Path) -> Path:
    """Write the summary and the log it came from, side by side."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": state.prompt,
        "ok": state.ok,
        "video": str(state.video) if state.video else None,
        "stopped": state.stopped,
        "error": state.error,
        "summary": summarise(state).as_dict(),
        "issues": [
            {"rule": i.rule, "where": i.where, "severity": i.severity.value,
             "message": i.message}
            for i in agent.issues_of(state)
        ],
        "scenes": [
            {
                "id": o.scene_id,
                "status": o.status,
                "repairs": o.repairs,
                "simplifications": o.simplifications,
                "calls": o.calls,
                "seconds": round(o.seconds, 2),
                "error": o.error,
            }
            for o in state.scenes
        ],
        "attempts": attempt_log(state),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
