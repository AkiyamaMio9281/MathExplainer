# Roadmap

A commit-by-commit plan from where the repository is now to a finished,
measured pipeline. `HANDOFF.md` section 6 gives the build order; this document
turns that order into individual commits and attaches an estimate to each.

Written 2026-09-08. Phases A and A′ are done and on `main`; everything
from commit 12 onward is still a plan.

---

## How to read the estimates

**Hours of focused work by one developer working with an AI assistant, not
wall-clock.** They cover writing the code, getting it to run, and writing the
tests named in the same row — not the reading, the false starts, or the time
spent watching a render.

Three things the numbers deliberately exclude:

- **Prompt iteration.** Commits 19 and 22 produce a *working* generator. Making
  what it generates *good* is open-ended and does not belong in an estimate.
- **Experiment runs.** Commit 28's ablations cost wall-clock time and dollars,
  both of which scale with how many prompts get run, not with how long the
  flag took to write.
- **Cold-start surprises.** Commit 20 is the first time every stage runs
  together against a real render. Unknown-unknowns land there; its range is
  wide on purpose.

A range means the low end is "this goes the way it should" and the high end is
"one thing in it does not".

---

## Phase A — backfill what already exists

Seven commits over files that are already written. The work is choosing the
boundaries and writing the messages, not producing code.

| # | Commit | Est. |
|---|---|---|
| 2 | `docs: the five stages, the agent loop, the escalation ladder` | 5 min |
| 3 | `test: toolchain smoke test for Cairo, LaTeX and ffmpeg` | 5 min |
| 4 | `feat(sandbox): run generated code with an allowlisted environment` | 5 min |
| 5 | `feat(validate): L0-L3 tiered checks on generated code` | 5 min |
| 6 | `docs: Scene IR schema and its validation rules` | 5 min |
| 7 | `docs: handoff notes, verified environment and tier measurements` | 5 min |
| 8 | `docs: README` | 5 min |
| | **Phase A** | **~30 min** |

All seven landed in that order, one file per commit.

**The order is not arbitrary, and two constraints fix it.** `validate.py`
imports `sandbox.py`, so 5 follows 4. And `HANDOFF.md` contains tier timings
that only exist once `validate.py` has been run, so 7 follows 5 — a history
that adds the measurements before the code that produced them reads as
fabricated, because it would be. `ARCHITECTURE.md` goes first for the opposite
reason: it is the design that justifies everything after it.

## Phase A′ — housekeeping before building

| # | Commit | Est. |
|---|---|---|
| 9 | `chore: normalise line endings` | 10 min |
| 10 | `fix(validate): reject multiple Scene classes; ignore stale renders` | 30 min |
| 11 | `docs: commit roadmap with estimates` (this file) | 15 min |
| | **Phase A′** | **~1 h** |

Commit 9 adds a `.gitattributes` (`* text=auto eol=lf`). Every `git add` up to
that point warned that LF would be replaced by CRLF, because `core.autocrlf`
is on globally; the attributes file overrides it and pins LF on both sides, so
the rule travels with the repository instead of depending on each
contributor's config. Verified with `git add --renormalize`: no file content
changed, because the blobs were already LF.

Commit 10 carries three defects found reading the code, none of which fails
loudly on its own:

- `check_syntax` silently takes `names[0]` when a file declares more than one
  `Scene` subclass, while `ARCHITECTURE.md` requires exactly one. Failing at
  L0 costs nothing; discovering it at render time costs a scene.
- `_find_video` takes the newest match under `media/videos`, so a stale MP4
  from an earlier render of the same workdir can be returned as this run's
  output. Clear the tree before rendering rather than guessing by mtime.
- `run_python` calls `cwd.mkdir(...)`, a side effect a process runner should
  not have. The caller that needs the directory should create it.

All three were verified by running them: a two-class file fails at L0 naming
both classes, a missing cwd raises without creating anything, and a planted
stale MP4 is cleared rather than returned. `HANDOFF.md` gained a row for the
new L0 failure and a note on the cwd contract, so the docs describe the code
as it now behaves.

Kept out of Phase A on purpose: Phase A is "the current state, in history".
Mixing fixes into it makes both harder to read.

---

## Phase B1 — tests for what already exists

No API key, no new dependencies. Cheapest confidence available.

| # | Commit | Est. |
|---|---|---|
| 12 | `test(sandbox): no credential-shaped name reaches the child` | 45 min – 1 h |
| 13 | `test(validate): each tier catches what it claims` | 1 – 1.5 h |
| | **Phase B1** | **~2 – 2.5 h** |

Commit 12 is a **security property**, not a unit test: it asserts that a child
process cannot read `ANTHROPIC_API_KEY`, that nothing matching
`_SECRET_MARKERS` survives, and that `child_env` refuses a secret passed
explicitly. It also introduces `tests/` and the pytest marker configuration
(`slow` for anything spawning a subprocess, `api` for anything needing a key)
so the suite runs green on a machine with neither.

Commit 13's test matrix is the table in `HANDOFF.md` section 4, including the
case that matters most: an undefined name inside `construct()` must stop at
**L2, not L1**. That split is the entire justification for having both tiers.

## Phase B2 — the IR layer

Pure functions over data. Still no API key.

| # | Commit | Est. |
|---|---|---|
| 14 | `feat(ir): dataclasses for the Scene IR document` | 2 – 3 h |
| 15 | `feat(ir): structural validation rules` | 3 – 4 h |
| 16 | `feat(ir): geometry and pacing warnings` | 2 – 3 h |
| | **Phase B2** | **~7 – 10 h** |

**Rules before the generator.** This is the one ordering in `HANDOFF.md` worth
restating: the rules are the only instrument that says whether generation is
working. Written after the generator, they get shaped to accept whatever it
already produces.

Commit 15's nine error rules are mostly five to fifteen lines each. Two are
not: `introduced-before-use` and `not-after-removal` both need a simulation of
the timeline — walking `steps` while tracking which objects are visible. Build
that walk once; commit 16 needs it too.

Commit 16 splits off because the bounding-box estimator is used only by
`in-frame` and `no-overlap`, and `no-overlap` needs pairwise intersection over
the *simultaneously visible* set, which is the timeline walk again. The
estimates are deliberately rough — exact text extents need a font engine the
validator does not load, which is why these are warnings rather than errors.

## Phase B3 — get the spine running

| # | Commit | Est. |
|---|---|---|
| 17 | `feat(video): concatenate per-scene clips with ffmpeg` | 1.5 – 2 h |
| 18 | `feat(llm): Anthropic client with token and cost accounting` | 3 – 4 h |
| 19 | `feat(codegen): compile one IR scene to Manim source` | 3 – 4 h |
| 20 | `feat: drive a hand-written IR end to end` | 2 – 4 h |
| | **Phase B3** | **~10 – 14 h** |

Commit 17 goes first because it needs no model: two throwaway clips are enough
to test it. Budget for the trap rather than the happy path — the concat
demuxer requires matching resolution, frame rate and codec parameters across
inputs, and a scene that rendered at a different size fails at the join, not
at the render. Decide there whether to normalise inputs or re-encode.

Commit 18 is where stale API knowledge burns an afternoon. **Read the
`claude-api` skill before writing it**; several shapes changed across 2025–26,
and structured outputs, prompt caching and thinking configuration are exactly
the parts that moved.

Commit 20 is the milestone `HANDOFF.md` recommends: one IR document, written
by hand, through codegen → validate → render → concat, with no planner and no
agent. It proves the spine and gives the repair loop something real to be
measured against. Its range is the widest in this document because it is the
first time the stages meet.

## Phase B4 — the agent

| # | Commit | Est. |
|---|---|---|
| 21 | `feat(plan): prompt to LessonPlan` | 1.5 – 2 h |
| 22 | `feat(ir): LessonPlan to SceneIR` | 2 – 3 h |
| 23 | `feat(repair): repair_code and simplify_ir` | 3 – 4 h |
| 24 | `test(repair): the ladder escalates and its bounds hold` | 1.5 – 2 h |
| 25 | `feat(agent): state, tool dispatch, and a dollar budget` | 3 – 4 h |
| 26 | `feat(metrics): attempt log and run summary` | 2 – 3 h |
| 27 | `feat(cli): prompt in, mp4 out` | 1 – 2 h |
| | **Phase B4** | **~15 – 20 h** |

Commit 23 is two unequal halves. `repair_code` is small — feed the traceback
back and ask again. `simplify_ir` is four ordered transformations over the
document, and the fourth (flatten to introductions only) is what guarantees
the ladder terminates. Get that one right first; the rest are optimisations
above it.

Commit 24 needs no API key: a fake codegen that fails a set number of times is
enough to assert that the ladder repairs, then simplifies, then drops, and
that both bounds hold.

Commit 25 carries the dollar budget **in its first version**. `HANDOFF.md`
section 8 is blunt about why: an agent that can loop on a scene the model
cannot get right is a way to spend money overnight.

Commit 26 is not optional polish. Without the attempt log the project is a
demo; with it, the numbers in `ARCHITECTURE.md`'s metrics table become the
research output.

## Phase B5 — what the numbers are for

| # | Commit | Est. |
|---|---|---|
| 28 | `feat: --no-repair and --no-validate-ir ablation flags` | 1 – 2 h |
| 29 | `perf: per-scene and per-animation render timings` | 2 – 3 h |
| 30 | `perf: persistent worker keeps manim imported` | 4 – 6 h |
| 31 | `feat(web): self-contained HTML player page` | 2 – 3 h |
| | **Phase B5** | **~9 – 14 h** |
| 32+ | `feat(web): canvas renderer consuming the same IR` | 15 – 30 h |

**Commit 28 has the best ratio in this document.** Two flags, an afternoon,
and the metrics stop being a pile of statistics and become an argument: this
is what the repair loop bought, this is what IR validation bought. The runs
themselves cost wall-clock and dollars on top.

Commit 30 attacks the fixed ~1 s of Python start-up plus `import manim` that
every check tier pays — see the measurements in `HANDOFF.md` section 5. A
worker process that keeps manim imported removes it entirely, and the
before/after is clean and quotable. It is real engineering, though: process
lifecycle, IPC, timeouts, crash recovery.

Commit 31 is the cheapest honest answer to the web requirement, and
`HANDOFF.md` is right that it should be described as partial coverage out
loud. Commit 32 is the strong answer and the largest single piece of work in
the project.

---

## Totals

| Through | Reached state | Est. |
|---|---|---|
| Commit 11 | History is clean, known defects fixed | ~1.5 h |
| Commit 16 | IR layer validated and tested, no API key needed | ~10 – 13 h |
| Commit 20 | Spine runs end to end on a hand-written IR | ~20 – 27 h |
| Commit 27 | Prompt in, MP4 out, with metrics and a budget | ~35 – 47 h |
| Commit 31 | Ablations, profiling, and a web page | ~44 – 61 h |
| Commit 32 | Second renderer on the same IR | ~59 – 91 h |

Against the autumn 2026 application window, commit 31 is comfortably **weeks,
not months**. Commit 32 is the item that changes that answer, which is why
`HANDOFF.md` asks for the scope decision early rather than late.

## What is soft

- **Prompt quality has no upper bound.** Commits 19 and 22 are estimated to
  first working output. How good the lessons are afterwards is a different
  activity, and the metrics from commit 26 are what should decide when to stop.
- **Commit 20 is the risk concentrator.** Every estimate before it is over code
  that can be reasoned about; it is the first that depends on how manim behaves
  on generated input.
- **The tier measurements are from a trivial scene.** `HANDOFF.md` section 5
  says to re-measure on real generated scenes before assuming L2 saves much
  over L3. If it does not, the loop's shape is worth revisiting — and that is a
  finding, not a setback.
