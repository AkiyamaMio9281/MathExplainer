# Handoff

Everything a fresh session needs. Read this before writing code.

Written 2026-09-08. Anything dated here was true on that machine on that day —
re-run the checks in "Verify the environment" before trusting them.

---

## 1. What this project is for

A prompt goes in, a finished math explainer video comes out. Read
`ARCHITECTURE.md` for the five stages and `SCENE_IR.md` for the layout format.

It is also portfolio work for a specific job posting, and that shapes the
priorities. The role is a Huawei Canada co-op, "Researcher — Web & AI", on the
Web/Windowing/Graphics team; the application window is autumn 2026, so the
timeline is **weeks, not months**. Its required skills include *applying AI in
content generation* and *fine tuning AI models and training data processing*;
*web profiling, CPU/GPU performance analysis, rendering pipeline, compute
pipeline, and GPU architecture* are listed as assets.

What that means for decisions here:

- **Content generation is the core** and it is covered by construction.
- **The rendering pipeline is real** — Manim rasterises through Cairo and has
  an OpenGL path. Profiling it is genuine work against a listed asset, not a
  stretch.
- **Web is the weak spot.** Video is not web content. The cheapest honest
  coverage is emitting a self-contained HTML player page alongside the MP4.
  The strong version is a second renderer consuming the same Scene IR, so two
  pipelines can be compared — that is why the IR does not mention Manim. Do
  not fake this; a wrapper page is worth saying out loud as partial coverage.
- **Metrics beat features.** A repair loop with measured success rates is
  worth more here than three more object types. `ARCHITECTURE.md` lists what
  to record.

A companion repo (an AlphaZero chess engine, `AkiyamaMio9281/Alpha-Zero-Style-Chess-Engine`)
covers PyTorch, CUDA, and profiling. This project does not need to.

---

## 2. Environment (verified 2026-09-08)

| Thing | Version | Note |
|---|---|---|
| OS | Windows 11 (10.0.26200) | |
| Python | 3.14.7 at `C:\Python314` | |
| manim | 0.21.0 | ManimCE, installed this session |
| anthropic | 1.4.0 | the 1.x line; built on `httpx2`, not `httpx` |
| ffmpeg | N-125781 | already present |
| LaTeX | TeX Live 2026 | already present — the painful dependency, and it works |
| `ANTHROPIC_API_KEY` | set | |
| GPU | RTX 5090 Laptop, sm_120 | irrelevant so far; matters if the OpenGL renderer gets profiled |

### Verify the environment

```bash
ffmpeg -version                       # must exist
latex --version                       # must exist
python -c "import manim, anthropic"   # must import
python -m manim -ql --disable_caching docs/examples/smoke.py Smoke
```

The last one is the real test: it exercises Cairo rasterisation, the LaTeX path
(`MathTex`), and ffmpeg encoding in one go. If it passes, the toolchain is
sound.

---

## 3. Traps that cost time

Four things bit during setup. Each looks like something else.

**`manim` is not on PATH.** pip installed `manim.exe` into
`C:\Users\cccct\AppData\Roaming\Python\Python314\Scripts`, which is not on
PATH. Always invoke `python -m manim`. The code already does.

**`os.environ.copy()` leaks the API key into generated code.** The pipeline
runs Python an LLM wrote, in a process that holds `ANTHROPIC_API_KEY`.
`sandbox.py` builds the child environment from an **allowlist**, not by copying
and deleting a few names — the copy-and-delete version leaks every credential
nobody thought to name. Verified: a child process reading
`os.environ["ANTHROPIC_API_KEY"]` gets nothing.

**Dropping `APPDATA` breaks `import manim` inside the sandbox.** Windows
resolves per-user site-packages from `%APPDATA%`, and manim is installed there.
Omit it from the allowlist and the sandbox fails with
`ModuleNotFoundError: No module named 'manim'`, which reads like a broken
install rather than a missing environment variable. It is in the allowlist now,
with a comment; do not "tidy it away".

**The console is cp1252.** Any subprocess printing non-ASCII dies with
`UnicodeEncodeError`. `sandbox.py` sets `PYTHONIOENCODING=utf-8` for children.
Set it yourself when running ad-hoc scripts that print anything but ASCII.

---

## 4. What is built

```
explainer/
  sandbox.py     isolated subprocess execution   done, verified
  validate.py    tiered code validation          done, verified
```

Nothing else exists yet. No LLM client, no plan, no IR, no codegen, no agent.

### `sandbox.py`

`run_python(args, cwd=, timeout=)` runs `sys.executable` as a child and returns
a `Completed` with `returncode`, `stdout`, `stderr`, `seconds`, `timed_out`,
plus `failure_text()` which trims to the tail (where the traceback is) for
feeding back to the model. *cwd* must already exist -- it raises rather than
creating it, so a workdir path typed wrong fails loudly instead of quietly
rendering into a new directory nobody looks at.

`child_env()` builds the environment from `_ENV_ALLOWLIST`, drops anything
whose name looks like a credential, and raises if a caller tries to pass one in
explicitly.

Verified: child cannot read `ANTHROPIC_API_KEY`; 14 environment variables reach
it, none sensitive; a 30-second sleep under a 2-second timeout is killed and
reported as `timed_out`.

**`SANDBOX_CAVEAT` is not boilerplate.** This is process isolation with a
timeout. No syscall filter, no network block. Containerise for untrusted input.

### `validate.py`

Four tiers, run in order, stopping at the first failure:

| Level | What it does | Catches |
|---|---|---|
| `L0 SYNTAX` | `ast.parse`, plus finding a `Scene` subclass | syntax errors, no scene class |
| `L1 IMPORT` | executes the module, constructs the scene | bad imports, undefined names at module level |
| `L2 DRYRUN` | `manim --dry_run` — runs `construct()`, writes nothing | everything inside the animation |
| `L3 RENDER` | `manim -q{quality}` | rasterisation and encoding |

`validate(workdir, code, up_to=Level.DRYRUN)` runs the ladder.
`scene_class_names(code)` finds Scene subclasses by AST — the class name is not
pinned in the prompt, because the model names it after the topic and forcing a
name spends a repair round on nothing that matters.

Verified behaviour:

| Input | Stops at | Result | Message |
|---|---|---|---|
| valid scene | L2 | pass | |
| `class Demo(Scene)` missing colon | L0 | fail | `SyntaxError at line 2: expected ':'` |
| no Scene subclass | L0 | fail | `No Scene subclass found` |
| two Scene subclasses | L0 | fail | `Found 2 Scene subclasses (Demo, Extra)` |
| module-level `import` of a missing module | L1 | fail | `ModuleNotFoundError` |
| undefined name inside `construct` | L2 | fail | `NameError: name 'NoSuchThing' is not defined` |
| `1/0` inside `construct` | L2 | fail | `ZeroDivisionError` |
| valid scene, `up_to=RENDER` | L3 | pass | produced `Demo.mp4` |

Note the undefined-name case stops at **L2, not L1** — it is only referenced
inside `construct()`, which L1 does not run. That split is the point: L1
isolates "the file is broken" from "the animation is broken".

---

## 5. Measurements

On this machine, against a five-animation scene with one `MathTex`:

| Tier | Time |
|---|---|
| L0 `ast.parse` | 0.07 ms |
| L1 import + instantiate | ~1.0 s |
| L2 `manim --dry_run` | ~1.3 s |
| L3 `manim -ql` full render | ~1.2–1.6 s |

**Read these carefully — the obvious conclusion is wrong.** L1, L2 and L3 are
close together because roughly a second of each is fixed cost: starting Python
and importing manim. The rendering itself is nearly free *for a trivial scene*.
Two consequences:

- Syntax checking is free and should always run first. That part is solid.
- The gap between dry-run and render widens with scene complexity, and this
  scene is not complex. **Re-measure on real generated scenes** before assuming
  L2 saves much over L3.
- The fixed ~1 s per check is the thing worth attacking if the repair loop
  becomes slow: a persistent worker process that keeps manim imported removes
  it entirely. That is a real optimisation with a measurable before/after.

**Cold start is 9.2 s.** The first-ever manim invocation builds LaTeX and font
caches. Warm runs are the numbers above. Do not benchmark the first run.

### Prompt caching (measured 2026-09-09, `claude-opus-5`)

A 7 803-token preamble sent twice:

| | input | cache write | cache read | cost |
|---|---|---|---|---|
| first call | 14 | 7 803 | 0 | $0.0489 |
| second call | 14 | 0 | 7 803 | $0.0041 |

**The repeat costs 8.4% of the first** -- a cache read is a tenth of an input
token, a write a quarter more than one. That is the whole argument for putting
the Manim guidance and the IR spec in a cached preamble: they are identical on
every codegen call and they dominate the tokens.

Two things this measurement pins down. The minimum cacheable prefix is
model-dependent, roughly 512-4096 tokens, and a preamble under it silently
does not cache at all -- there is no error, only a bill. And cache entries
live about five minutes, which makes any caching test that assumes a cold
start flaky when run twice in that window; `tests/test_llm.py` puts a nonce in
the preamble for exactly that reason.

Frame geometry, needed by the IR validator and independent of quality:

```
frame_width = 14.222   ->  x in [-7.111, 7.111]
frame_height = 8.000   ->  y in [-4.000, 4.000]
```

---

## 6. What to build next, in order

1. **`explainer/llm.py`** — Anthropic client wrapper. `claude-opus-5`,
   adaptive thinking, token/cost accounting, typed error handling. Structured
   outputs (`output_config.format`) for anything with a schema. Read the
   `claude-api` skill before writing this; several API shapes changed in
   2025–26 and a recalled pattern is likely stale.
2. **`explainer/ir.py` + `ir_rules.py`** — IR dataclasses and the validation
   rules from `SCENE_IR.md`. **Build the rules before the generator.** They are
   pure functions, they are testable without an API key, and they are what
   tells you whether generation is working.
3. **`explainer/plan.py`** — Prompt to LessonPlan.
4. **`explainer/codegen.py`** — one IR scene to Manim source. Cache the static
   part of the prompt; the Manim guidance and IR spec are identical on every
   call and dominate the tokens.
5. **`explainer/repair.py` + `agent.py`** — the escalation ladder from
   `ARCHITECTURE.md`, with bounded rounds and a global budget.
6. **`explainer/video.py`** — ffmpeg concat.
7. **`cli.py`** — prompt in, mp4 out.

Then, and only then, the things that serve the job posting: render profiling,
and an HTML player page.

### Suggested first milestone

One hard-coded IR document, by hand, through codegen → validate → render →
concat. No planner, no LLM in the loop except codegen. That proves the spine
end to end and gives the repair loop something real to be measured against.

---

## 7. Testing

`tests/` is empty. What is worth testing, in rough priority:

- **IR rules** — pure functions, no API key, no subprocess. Highest value per
  line. One test per rule, each with a document that violates exactly that rule.
- **`sandbox.child_env`** — that no credential-shaped name survives. This is a
  security property; it deserves a test that fails loudly.
- **`validate` tiers** — that each tier catches what it claims and lets through
  what it should. The table in section 4 is the test matrix.
- **The escalation ladder** — with a fake codegen that fails a set number of
  times, assert that it repairs, then simplifies, then drops, and that bounds
  hold.

Anything touching the API should be behind a marker so the suite runs without
a key.

---

## 8. Open questions

- **Narration is text only.** No TTS, no audio track. Adding it inverts the
  timing relationship: `duration` becomes an output of the TTS rather than an
  input to it, and the `pacing` rule becomes an error rather than a warning.
- **Parallel scene rendering** is possible by construction (independent
  workdirs) but not implemented. Worth doing once render time is the
  bottleneck, and it makes a good profiling result.
- **The web renderer** consuming the same IR is the strongest available answer
  to the posting's web requirement, and also the largest piece of work. Decide
  early whether it is in scope; the IR is already renderer-agnostic either way.
- **No cost ceiling is enforced yet.** The agent loop should carry a dollar
  budget from the first version, not after the first expensive night.
