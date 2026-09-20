"""Prompt in, MP4 out.

    python cli.py "explain why the derivative of sin is cos"

Everything else has a default. The one flag worth knowing about is --budget:
the run stops when it reaches the ceiling and keeps the scenes that rendered
before it, so lowering it is how you cap an experiment rather than how you
break one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from explainer import agent, llm, metrics, video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Turn a one-line prompt into a math explainer video.",
    )
    parser.add_argument("prompt", help="what the lesson should explain")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="where to work and write lesson.mp4 (default: runs/<slug>)",
    )
    parser.add_argument(
        "-q",
        "--quality",
        default="l",
        choices=list("lmhpk"),
        help="manim render quality (default: l)",
    )
    parser.add_argument(
        "-b",
        "--budget",
        type=float,
        default=agent.DEFAULT_DOLLARS,
        help=f"dollars this run may spend (default: {agent.DEFAULT_DOLLARS:.2f})",
    )
    parser.add_argument(
        "--repair-rounds",
        type=int,
        default=None,
        help="repairs per version of a scene (default: 3)",
    )
    parser.add_argument(
        "--simplify-rounds",
        type=int,
        default=None,
        help="simplifications per scene (default: 2)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the run record to stdout as well as writing it",
    )
    return parser


def slug(prompt: str, limit: int = 40) -> str:
    kept = [c.lower() if c.isalnum() else "-" for c in prompt]
    out = "".join(kept).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    return out[:limit].strip("-") or "lesson"


def main(argv: list[str] | None = None) -> int:
    from explainer import repair

    args = build_parser().parse_args(argv)

    if not llm.have_key():
        print(
            "ANTHROPIC_API_KEY is not set, and every stage needs it.",
            file=sys.stderr,
        )
        return 2
    if not video.available():
        print("ffmpeg and ffprobe must both be on PATH.", file=sys.stderr)
        return 2

    workdir = args.out or Path("runs") / slug(args.prompt)
    state = agent.run(
        args.prompt,
        workdir,
        budget=agent.Budget(dollars=args.budget),
        quality=args.quality,
        repair_rounds=(
            repair.REPAIR_ROUNDS if args.repair_rounds is None else args.repair_rounds
        ),
        simplify_rounds=(
            repair.SIMPLIFY_ROUNDS
            if args.simplify_rounds is None
            else args.simplify_rounds
        ),
    )

    print(state.summary())
    print()
    print(metrics.summarise(state).render())

    record = metrics.record(state, workdir / "run.json")
    print(f"\nrecord: {record}")
    if args.json:
        print(record.read_text(encoding="utf-8"))

    return 0 if state.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
