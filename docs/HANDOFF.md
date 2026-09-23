# Handoff

Everything a fresh session needs. Read this before writing code.

Written 2026-09-08. Anything dated here was true on that machine on that day —
re-run the checks in "Verify the environment" before trusting them.

---

## 1. What this project is for

A prompt goes in, a finished math explainer video comes out. Read
`ARCHITECTURE.md` for the six stages and `SCENE_IR.md` for the layout format.

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

**A relative path handed to a child with a different cwd is resolved twice.**
This cost two separate debugging sessions, in two modules, and neither was
caught by any test. `check_import` passed the driver's path and ffprobe was
passed the clip's, both while `cwd` was the directory those files were in, so
the child looked for `runs/x/scene/runs/x/scene/_import_check.py` and reported
a file nobody had named. Every test used pytest's `tmp_path`, which is
absolute, so nothing failed until the CLI built `runs/<slug>`. Both modules
now resolve paths before handing them over, `sandbox.run` resolves its own
cwd, and both have a regression test that runs from a relative directory.

**Structured outputs compile the schema into a grammar, and it has a ceiling.**
Four separate 400s, each with its own message, found while building the layout
stage:

- `minItems` is accepted only as `0` or `1`; any other value is rejected by
  value.
- `maxItems` is not supported at all.
- `additionalProperties` must be present on every object and must be `false`.
  An open object is not available, so a schema cannot say "these fields, plus
  whatever else".
- Size is the real limit. A nine-way tagged union of the IR's object types is
  "the compiled grammar is too large" at document *and* scene level; so is
  seven. Six compiles, at about 3.5 kB of schema JSON. Merging the nine into
  one object with every field optional is "Schema is too complex" -- the cost
  is in the optional properties, not the nesting.

`explainer/layout.py` works in six of the nine types because of that last one,
and `tests/test_layout.py` pins the size so widening the vocabulary fails
locally rather than as a 400 mid-run.

**The console is cp1252.** Any subprocess printing non-ASCII dies with
`UnicodeEncodeError`. `sandbox.py` sets `PYTHONIOENCODING=utf-8` for children.
Set it yourself when running ad-hoc scripts that print anything but ASCII.

---

## 4. What is built

```
explainer/
  sandbox.py     isolated subprocess execution   done, verified
  validate.py    tiered code validation          done, verified
  ir.py          Scene IR, parser, schema        done, tested
  ir_rules.py    12 validation rules             done, tested
  llm.py         Anthropic client + costing      done, verified live
  plan.py        prompt -> LessonPlan            done, verified live
  speech.py      narration -> audio + timings    done, verified live
  layout.py      LessonPlan -> SceneIR           done, verified live
  codegen.py     IR scene -> Manim source        done, verified live
  repair.py      simplify_ir + escalation ladder done, tested
  video.py       concat, subtitles, sound        done, tested
  agent.py       state, dispatch, dollar budget  done, verified live
  metrics.py     attempt log + run summary       done, tested
  spine.py       document -> mp4, no agent       done, verified live
cli.py           prompt -> mp4                   done, verified live
```

**The pipeline runs end to end.** `python cli.py "..."` takes a prompt to a
lesson.mp4. What is left is not plumbing: the ablations that make the metrics
mean something, render profiling, and the web coverage. See `ROADMAP.md`.

`spine.py` stays free of the repair loop on purpose -- it is the no-repair
baseline the ablation needs, already written and already tested.

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

### Speech, and the stage that had to move (2026-09-23)

The lesson has a voice. `edge-tts` speaks through the endpoint Microsoft Edge's
read-aloud uses: free, no key, no account, and **not an official API** -- it can
change or disappear without notice and it needs a network connection. Every
failure path in `speech.py` returns a `Spoken` that says it did not work, and
the pipeline falls back to the silent-with-subtitles lesson it made before.

**The timing inversion was real, and the fix was to move the stage, not to
adjust afterwards.** Without speech, a scene's narration *estimates* how long
that scene should run. With speech the synthesiser *decides*. Two orderings
were available:

- Speak last, then reconcile. Either stretch the video to meet the audio
  (quality loss, and manim has already encoded) or pad the audio and let them
  drift apart. Both are corrections applied to a mismatch that has already
  happened.
- Speak *between* `plan` and `layout`, and hand the layout the measured
  seconds as its budget. The words are final once the plan is, so nothing is
  re-synthesised when a layout is retried or a scene repaired -- it runs once,
  before either can happen.

The second is what `agent.run` does. The audio and the animation line up by
construction. `plan.Beat.seconds` carries the measurement, `Beat.spoken_seconds`
returns it when present and the word-count estimate otherwise, and
`layout.plan_prompt` states which of the two the model is being given -- an
estimate is a target to come close to, a measurement is a length to fill.

**Telling the layout the duration is not enough; it has to be fitted.** The
first voiced run showed this straight away, which is what `Run.drift` was added
to reveal. Every scene was told its measured length and every scene came in
short:

```
scene              asked   clip  voice     gap
the_claim           44.5   44.7   46.9   +2.14
tear_the_corners    42.5   42.7   48.9   +6.21
parallel_line       41.5   41.9   46.5   +4.59
landing             50.5   50.7   54.0   +3.27
```

Sixteen seconds of voice past the end of a 180-second video, and the last
scene's closing words cut off. `pacing` passes all four, because each one
individually is within 30%. The lesson is still wrong.

The fix is `layout.fit_to_speech`, and it is arithmetic rather than another
model call: each scene's step durations are scaled uniformly so the total
equals the measured narration. **This is the clearest payoff the IR stage has
produced.** The layout's output is data, so a systematic error in it can be
corrected exactly, once, in six lines -- with generated code there would be
nothing to scale.

The same prompt, run again with the fit in place:

```
scene                  asked   clip  voice     gap
the_claim               52.4   52.9   52.4   -0.50
tear_and_rearrange      55.2   55.6   55.1   -0.45
parallel_line_proof     73.1   73.4   73.1   -0.30
where_it_fails          58.6   58.5   58.6   +0.09
```

Worst gap +6.20s before, **+0.09s after**. What is left is manim rounding
`run_time` up to whole frames, which is a third of a second a scene and not
worth chasing. Video 240.43s against 239.18s of voice: nothing is cut.

Uniform, so which beat gets more time stays the layout's decision. Putting the
whole difference into a trailing `wait` was the alternative and is worse: the
animation finishes early and the viewer watches a still frame while the voice
catches up. Scales outside 0.5-2.0 are reported as a `fits-narration` warning
and left alone -- a sixfold stretch is slow motion, not a correction.

**The speaking rate was wrong and is now measured.** `WORDS_PER_MINUTE` was
150.0, a plausible guess. Two invented samples suggested 164.5, which was also
wrong -- a 16-word sample never pauses between sentences. Measured against the
27 real narrations from the runs on disk, spoken by the voice that actually
speaks them:

```
27 narrations   4151 words   1593.7s of audio   156.3 wpm pooled
per scene       min 133.7    median 156.5       max 175.3
```

The constant is now 156.0. At that value the median per-scene error is 5.4% and
the worst is 13.7%, comfortably inside the 30% `PACING_TOLERANCE`. It still
only *estimates*: when speech runs, the real duration is known per scene and is
what the layout is given. What it still decides is how many words a scene may
carry (`plan.MIN_WORDS` / `MAX_WORDS`, both derived) and the subtitle timings
of a lesson rendered with `--no-audio`.

**Word boundaries must be asked for.** `edge_tts.Communicate` defaults to
`boundary="SentenceBoundary"`, which returns one timing per sentence -- and a
sentence too long for one subtitle then has nowhere to be split. Pass
`boundary="WordBoundary"`. Offsets come back in **100-nanosecond ticks**, an
unusual unit and an easy one to read as milliseconds, which would put every
subtitle ten thousand times early.

**Audio is placed, not concatenated.** Each scene's speech is delayed to its
own start offset with `adelay` and mixed with `amix`, rather than joined end to
end. Concatenating would make one scene's overrun push every later scene out of
step, and the error would accumulate down the lesson; placing bounds it to the
scene that caused it, which bleeds slightly into the next one -- what a person
reading aloud does anyway. Pass `normalize=0` to `amix`, or one voice gets
quieter the more scenes the lesson has.

Start offsets come from probing the rendered clips, not from the IR: what a
scene asked for and what manim produced differ slightly every time.

**Order: burn, then mux.** Burning subtitles re-encodes the picture; muxing
copies it. The other order pays for a second video encode or loses the audio.

### Which model should generate the code (measured 2026-09-23)

Paired: the same 27 IR scenes, planned and laid out once on Opus and cached,
handed to three generators. Nothing but the generator changes. Rendering is
stubbed -- what is being measured is whether generated code *runs*, which L2
settles, and rasterising it would add minutes and change no number here.

| model | L2 first try | rendered | repairs | simplifications | dropped | cost | time |
|---|---|---|---|---|---|---|---|
| `claude-opus-5` | 27/27 | 27/27 | 0 | 0 | 0 | $1.223 | 577 s |
| `claude-sonnet-5` | 27/27 | 27/27 | 0 | 0 | 0 | $0.482 | 513 s |
| `claude-haiku-4-5` | 25/27 | 27/27 | 2 | 0 | 0 | $0.558 | 821 s |

Three things worth taking from this.

**Sonnet is the default.** It matched Opus exactly -- 27/27 on the first
attempt, no repairs -- for 39% of the cost and slightly less wall time. There
is no measured reason to pay for Opus at this stage. The planning and layout
stages still run on Opus; they were not varied here.

**Haiku costs more than Sonnet despite half the token price.** Haiku 4.5 has no
adaptive thinking, so `llm.Capabilities` gives it a fixed 4,000-token budget,
and a fixed budget spends itself whether the scene needs it or not. Adaptive
thinking does not. The cheaper model was 16% more expensive and 60% slower.
*The per-token price is not the price.*

**Both Haiku failures were the same failure**, and it is the kind the ladder
exists for:

```
TypeError: Mobject.__init__() got an unexpected keyword argument 'center'
```

A plausible-looking keyword that manim does not have. L2 caught both, one
repair round fixed both, and neither reached a render. This is the escalation
ladder doing exactly the job it was built for -- and it is also why the
41/41-first-try figure from the eight real runs should not be read as "the
ladder is unnecessary". It is unnecessary *for Opus and Sonnet on these
prompts*. Change the model and it starts earning its place immediately.

**Where the money actually goes.** Planning and laying out five lessons cost
$2.75. Generating code for all 27 scenes three times over cost $2.26. Codegen
is not the expensive stage; the two structured-output stages that run once per
lesson are. Optimising codegen further would be optimising the smaller half.

### Eight runs, forty-one scenes (measured 2026-09-23)

Five prompts across five areas of mathematics -- calculus, probability, linear
algebra, algebra, series -- run in parallel, plus the three earlier runs:

```
8 runs, 41 scenes
  code passed L2 first try   41/41 = 100%
  code repairs               0
  simplifications            0
  scenes dropped             0
  layout retries             1 of 8 runs
  cost                       $6.05 total, $0.136 a scene, $0.61-1.11 a lesson
```

**The code repair ladder has never fired.** Not once in forty-one scenes. The
most likely reason is not that the model is flawless but that the IR stage has
made codegen easy: translating a validated declarative document with six
object types and eight actions into Manim is close to mechanical. The open
decisions -- what to show and where -- happen in the layout stage, and that is
where the one repair loop that *has* fired was needed.

That has a direct consequence for the ablation in commit 28. Running the same
prompts with `--no-repair` would show no difference at all, which is an honest
result and a useless experiment. The informative version varies the *model*
instead: generate code on Sonnet 5 or Haiku 4.5 and measure how much the
ladder recovers as the generator gets weaker. That also answers a question
worth money -- whether a cheap model plus a repair loop beats an expensive one
alone.

**Five lessons in parallel took 394 s, about the same as one.** Time is
dominated by waiting on the model, not by local CPU, so throughput scales with
parallel lessons. That settles the parallel-rendering open question below:
parallelise at the lesson level, not the scene level.

### The MathTex estimate, measured and replaced (2026-09-23)

177 distinct expressions were harvested from the 41 generated scenes -- what
the model actually writes, rather than what a test author imagines -- and
rendered to measure against.

| estimate | median ratio to manim | worst | over 1.5x |
|---|---|---|---|
| `0.16 * len(source)` | 1.59 | 9.42 | 97/177 |
| `0.22 * visible_length` | 0.98 | 2.70 | 3/177 |

Height was wrong in both directions at once: one constant of 0.60 was 2.1x too
tall for a plain expression (median 0.284, n=134) and too short for the tallest
fraction by nearly half (median 0.770, max 0.978, n=43). It is two values now,
chosen just above each group's p90.

Re-judging the warnings eight real runs reported: **42 of 50 disappear,
including every one of the 23 overlaps**. The eight that remain are text
genuinely placed in the subtitle band, which is the rule working.

That re-check needed the IR documents and the run records did not have them,
so the layouts had to be reconstructed from the generated Python. The record
now keeps the document -- `metrics.py` claims a summary should be recomputable
and this is what that costs when it is not.

### The MathTex extent estimate is where the warnings come from

Those 25 scenes produced 45 layout warnings, and **41 of them (91%) name a
MathTex object**; all 23 `no-overlap` warnings do. The two most-warned scenes
were then inspected frame by frame, from a still moment, and both were clean.

The cause is stated in `ir_rules.py` and was not treated as urgent until now:
MathTex width is estimated from the length of the *LaTeX source*.
`+\frac{1}{2}` is eleven characters that render as one narrow stacked
fraction; `\begin{bmatrix}3 & 1\\0 & 2\end{bmatrix}` is thirty-odd that
render as a small block. So the estimator inflates anything with markup, and
the rule reports overlaps that are not there.

This has to be fixed before warnings are allowed to drive `simplify_ir`. Wiring
them up as they stand would tear apart good scenes on the strength of a bad
estimate. The fix is the same shape as the one in 688c8b3: strip LaTeX markup
before counting, then calibrate against manim with a slow test.

### The layout retry, firing for the first time (2026-09-20)

Same prompt, after the subtitle band was reserved:

```
6/6 scenes in 508s for $1.1107 over 9 calls
IR clean first try      no  (2 layout attempts)
code passed L2 first    100%
rules fired             none
```

Nine calls rather than seven: the first layout came back with errors, the
retry carried the issues back, and the second was clean. That is the first
time any repair loop in this project has fired on a real run and worked, and
the run it produced has **zero** layout warnings against five on the previous
attempt at the same prompt.

One gap it exposed: only the *final* attempt's issues are recorded, so what
the first layout got wrong is lost. The metric ARCHITECTURE.md asks for --
which rule caught which defect -- needs the issues from every attempt, not the
last.

### Prompt to MP4 (measured 2026-09-20)

`python cli.py "explain why the angles of a triangle add to 180 degrees"`:

```
5/5 scenes in 328s for $0.7019 over 7 calls
code passed L2 first    100%
escalated to simplify   0%
repair rounds           0: 5
time                    328s total, 39s rendering
cost                    $0.7019, $0.1404 a rendered scene
```

226 seconds of video. Seven model calls: one plan, one layout, five codegen --
the repair loop was never needed, which is the best possible result and also
the least informative one. The ablations are what turn a run like this into a
number worth quoting.

Note where the time goes: 39 s of the 328 is rendering. The rest is the model
thinking, so the persistent-worker optimisation aimed at manim's start-up cost
would move about a tenth of the wall clock. Worth knowing before optimising the
wrong end.

### End to end (measured 2026-09-09)

`docs/examples/pythagoras.json`, three scenes, hand-written, through codegen ->
validate -> render -> concat with no repair loop:

```
3/3 scenes rendered in 32.3s for $0.0337
  ok   statement  (8.4s)
  ok   squares   (15.0s)
  ok   conclusion (8.8s)
```

All three passed L0 through L2 on the first attempt. That is three samples, not
a success rate -- the ablations in commit 28 are what turn this into a number
worth quoting -- but it means the repair loop will be built against a pipeline
that already works rather than one that depends on it.

Of the 32.3 s, generation is most of it; the renders are the second-and-a-bit
each that the tier table below predicts.

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

The *usable* frame is smaller and not centred. Burned-in subtitles occupy
`y -3.25 .. -2.52`, measured on an 854x480 render with a two-line cue, so the
`in-frame` rule allows `y` from -2.4 to 3.8. The top was 3.6 until a real run
warned five times about titles reaching 3.68 that were not clipped at all.

---

## 6. What to build next, in order

**Items 1-6 below are done** (commits 12-20; see `ROADMAP.md` for what each
one settled). They are kept here because the reasoning still applies to what
wraps them.

1. ~~**`explainer/llm.py`**~~ — Anthropic client wrapper. `claude-opus-5`,
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

- ~~**Narration is burned in as subtitles, not spoken.**~~ -- done. The
  lesson has a voice. The timing inversion this warned about was real and was
  resolved by **moving the stage rather than adjusting afterwards**: speech
  runs between `plan` and `layout`, so the synthesiser's measured duration is
  what the layout is given as its budget. See §5, "Speech".
- **`pacing` still checks the estimate, not the measurement.** The layout is
  told a scene's measured length, but `ir_rules.pacing` compares the step
  durations against `words / 156 * 60`, because the IR scene does not carry
  the synthesised duration -- only `plan.Beat` does. The two disagree by about
  5-6% typically and 14% at worst, well inside the rule's 30% tolerance, so it
  does not misfire; it is simply checking a slightly different thing from what
  the layout was asked to do. Threading the measurement into `ir_rules.check`
  would close it, and would let `pacing` become an error rather than a warning
  when audio is on -- which is the upgrade SCENE_IR.md anticipated.
- **Nothing acts on a layout warning.** `in-frame` and `no-overlap` are
  recorded and then ignored. SCENE_IR.md says they may drive `simplify_ir`;
  the ladder only simplifies when code fails to compile, never when a scene
  compiles into something that looks wrong. That is the gap between a lesson
  that renders and a lesson worth watching, and the ablations should measure
  it before anything is built to close it.
- **Read frames from a still moment.** Sampling a rendered lesson on a fixed
  interval catches `Write` animations half drawn, and at thumbnail scale a
  half-drawn line looks exactly like two lines on top of each other. A whole
  round of "the layout is overlapping" turned out to be that; the settled
  frames were clean and the estimator was within 12% of manim. Grab a frame
  during a `wait`, at full resolution, before believing a layout is broken.
- ~~**Parallel scene rendering**~~ -- answered by measurement. Five lessons
  run in parallel finished in the time of one, because the wall clock is model
  latency rather than local CPU. Parallelism belongs at the lesson level;
  parallel *scenes* would optimise the 10-20% of a run that is rendering.
- **The web renderer** consuming the same IR is the strongest available answer
  to the posting's web requirement, and also the largest piece of work. Decide
  early whether it is in scope; the IR is already renderer-agnostic either way.
- **No cost ceiling is enforced yet.** The agent loop should carry a dollar
  budget from the first version, not after the first expensive night.
