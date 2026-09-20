# MathExplainer

Turn a one-line prompt into a finished math explainer video.

```
"explain why the derivative of sin is cos"  ->  lesson.mp4
```

The system is an agent: it plans the lesson, lays out each scene as structured
data, compiles that to Manim code, runs the code in a sandbox, repairs what
fails, renders, and stitches the result together. Failures escalate rather than
crash the run -- a scene whose code will not compile gets its code repaired,
then its layout simplified, and only then is it dropped.

```
Prompt -> Lesson Plan -> Scene IR -> Manim Code -> Render -> MP4
             (LLM)        (LLM)       (LLM)        (manim)   (ffmpeg)
                            |            |
                      validated      validated in tiers,
                     structurally     repaired on failure
```

## Status

It runs. A prompt goes in and a lesson comes out.

```bash
python cli.py "explain why the angles of a triangle add to 180 degrees"
```

Measured on that prompt, 2026-09-20: five scenes, all five rendered, no
repairs and no simplifications, 328 s and $0.70 for 226 s of video. One run is
not a success rate -- the ablations that turn these into an argument are still
to come -- but every stage is built and every one has tests.

| Component | State |
|---|---|
| `sandbox.py` -- isolated subprocess execution | done |
| `validate.py` -- tiered code validation | done |
| `ir.py` / `ir_rules.py` -- Scene IR, parser, 12 rules | done |
| `llm.py` -- Anthropic client, caching, cost accounting | done |
| `plan.py` -- prompt to lesson plan | done |
| `layout.py` -- lesson plan to Scene IR | done |
| `codegen.py` -- IR scene to Manim source | done |
| `repair.py` -- repair, simplification, escalation ladder | done |
| `video.py` -- ffmpeg concatenation | done |
| `agent.py` -- state, dispatch, dollar budget | done |
| `metrics.py` -- attempt log and run summary | done |
| `cli.py` -- prompt in, mp4 out | done |
| Ablations, render profiling, a web player | not started |

`spine.py` runs a hand-written IR document through the same renderer with no
agent and no repair loop, which makes it the baseline the ablations need.

**Start here if you are picking this up:** [`docs/HANDOFF.md`](docs/HANDOFF.md).
It carries the verified environment facts, the measurements the design depends
on, and the gotchas that cost time to rediscover.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | What exists, what is measured, what to build next, known traps |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The five stages, the agent loop, the escalation ladder, metrics |
| [`docs/SCENE_IR.md`](docs/SCENE_IR.md) | Scene IR schema and its validation rules |

## Setup

Requires Python 3.10+, plus two system dependencies Manim shells out to.

```bash
pip install -r requirements.txt
```

| Dependency | Why | Check |
|---|---|---|
| **ffmpeg** | Manim encodes frames to MP4 with it; also used to concatenate scenes | `ffmpeg -version` |
| **LaTeX** (TeX Live / MiKTeX) | `MathTex` renders formulas through it | `latex --version` |

Set an API key for the generation stages:

```bash
export ANTHROPIC_API_KEY=...        # Windows: setx ANTHROPIC_API_KEY ...
```

Verify the render toolchain end to end before anything else -- if this fails,
nothing downstream can work:

```bash
python -m manim -ql --disable_caching docs/examples/smoke.py Smoke
```

Use `python -m manim`, not the `manim` command: pip installs the executable to a
Scripts directory that is often not on PATH.

## Security

The pipeline executes Python written by a language model. `explainer/sandbox.py`
runs it as a child process with a hard timeout and an environment built from an
allowlist, so generated code never sees `ANTHROPIC_API_KEY` or any other
credential in the parent environment.

**That is isolation, not a security boundary.** There is no syscall filter and
no network block; generated code can still reach the network and any file the
user can reach. Run the whole pipeline inside a container if the prompt source
is untrusted.
