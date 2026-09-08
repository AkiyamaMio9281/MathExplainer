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

Early. The execution substrate is built and verified; the generation stages are
not written yet.

| Component | State |
|---|---|
| `explainer/sandbox.py` -- isolated subprocess execution | done, verified |
| `explainer/validate.py` -- tiered code validation | done, verified |
| Lesson Plan / Scene IR / codegen / agent loop | not started |

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
