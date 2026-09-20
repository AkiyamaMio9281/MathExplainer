# Architecture

## The five stages

```
  Prompt                 "explain why the derivative of sin is cos"
    |
    |  LLM
    v
  Lesson Plan            pedagogy: what to teach, in what order, in what words
    |                    { title, audience, scenes: [{ id, beat, narration }] }
    |  LLM
    v
  Scene IR               layout: what is on screen, where, and when
    |                    declarative, renderer-agnostic, independently validated
    |  LLM
    v
  Manim Code             one Python file per scene, one Scene subclass each
    |
    |  manim (sandboxed)
    v
  Per-scene MP4s
    |
    |  ffmpeg concat, then the narration burned in as subtitles
    v
  lesson.mp4
```

### Why Scene IR is its own stage

Going straight from a lesson plan to Manim code is one long jump, and when the
output is wrong there is no way to tell whether the *idea* was bad or the
*code* was. Splitting them buys three things:

1. **Validation before execution.** The IR is data, so it can be checked
   without running anything: dangling references, objects off the edge of the
   frame, objects that overlap, narration whose length does not match the
   animation's. Those are the defects that survive a clean compile and only
   show up when a human watches the video. See `SCENE_IR.md`.
2. **Failures localise.** A broken scene is repaired at the layer that broke.
   Bad code gets the code regenerated; a layout that cannot be rendered gets
   the IR simplified. Neither touches the lesson plan, so the explanation does
   not silently change while chasing a compile error.
3. **A second renderer stays possible.** The IR does not mention Manim. A web
   renderer (Canvas/SVG) consuming the same IR is additive work, not a rewrite
   — and that comparison is the strongest available evidence for the rendering
   and web-profiling parts of the target role.

### Scene granularity

**One Manim `Scene` per IR scene, rendered independently, concatenated at the
end.** Not one large Scene for the whole lesson.

- A failure is contained: scene 4 failing does not cost scenes 1-3.
- Repair regenerates one small file instead of the whole lesson.
- Scenes can render in parallel later; a single Scene cannot.
- The cost is a concatenation step, and ffmpeg is already a dependency.

Each scene gets its own working directory containing a `scene.py`, which is
what `explainer/validate.py` already expects.

## The agent loop

The pipeline is not a fixed script. It holds state, chooses the next action
from that state, and escalates when an action keeps failing.

```
state = {
    prompt,
    plan       | None,
    ir         | None,
    scenes: [ { ir, code, checks[], video, status } ],
    attempts: [ ... ],          # every tool call, its outcome, its cost
    budget:   { llm_calls, wall_seconds, dollars },
}
```

### Tools

| Tool | Signature | Notes |
|---|---|---|
| `plan` | `prompt -> LessonPlan` | structured output |
| `build_ir` | `LessonPlan -> SceneIR` | structured output |
| `validate_ir` | `SceneIR -> [Issue]` | pure; no LLM, no subprocess |
| `codegen` | `SceneIR.scene -> str` | one scene at a time |
| `validate_code` | `(code, Level) -> Check` | `explainer/validate.py` |
| `repair_code` | `(code, error) -> str` | error text goes back to the model |
| `simplify_ir` | `(scene, reason) -> scene` | escalation target |
| `render` | `code -> Path` | `explainer/validate.py::render` |
| `concat` | `[Path] -> Path` | ffmpeg |
| `subtitle` | `(Path, [(narration, clip)]) -> Path` | ffmpeg; timed by each clip's measured length |

### The escalation ladder

This is the part that makes it an agent rather than a retry loop. When a scene
will not compile, the response depends on how many times it has already
failed, and each rung changes a *different* layer:

```
  codegen produces a scene
        |
        v
  validate_code  ---- passes ----> render
        |
      fails
        |
        v
  repair_code  (<= REPAIR_ROUNDS, default 3)
        |                    feeding the exact traceback back each round
      still failing
        |
        v
  simplify_ir  (<= SIMPLIFY_ROUNDS, default 2)
        |       drop the most complex object, shorten the step list,
        |       then regenerate code from the simpler IR
      still failing
        |
        v
  drop the scene, record why, continue the lesson without it
```

Graceful degradation matters more than it sounds: a six-scene lesson that
delivers five scenes and an explicit note about the sixth is a usable result,
whereas an exception after four minutes of rendering is not.

Every rung is bounded, and the run also has a global budget. An agent that can
loop forever on a scene the model cannot get right is a way to spend money
overnight.

### Validation order inside a scene

Cheap checks first, and never render to find out something a parse would have
caught. Measured tier costs are in `HANDOFF.md`; the short version is that
syntax checking is effectively free and everything above it costs about a
second of fixed overhead before it does any work.

```
L0 syntax (ast)  ->  L1 import  ->  L2 dry run  ->  [repair loop ends here]  ->  L3 render
```

The repair loop stops at L2. Rendering happens once, after the scene is known
to execute.

## What to record

The metrics are the research output; without them this is a demo. Record per
run, and keep the raw attempt log so the numbers can be recomputed:

| Metric | Why it matters |
|---|---|
| IR valid on first attempt (%) | how well the model does the layout task |
| Code passes L2 on first attempt (%) | the headline generation-quality number |
| Repair rounds to success (distribution) | how far a wrong answer is from a right one |
| Escalation rate to `simplify_ir` (%) | how often local repair is not enough |
| Scenes dropped (%) | the failure rate a viewer would actually notice |
| Which IR rule caught which defect | shows the validator earns its place |
| Render seconds per scene, and per animation | the input to any performance work |
| LLM calls, tokens, dollars per finished video | whether the thing is affordable |

Two comparisons make these numbers mean something: the same prompts with the
repair loop disabled, and with IR validation disabled. Both are one flag each
and turn a pile of statistics into an argument.

## Layout for the code to come

```
MathExplainer/
  cli.py                  prompt in, mp4 out
  explainer/
    sandbox.py            done -- isolated subprocess execution
    validate.py           done -- L0/L1/L2/L3 code checks
    llm.py                Anthropic client, retries, token accounting
    plan.py               Prompt -> LessonPlan
    ir.py                 IR dataclasses, parser, and its own JSON schema
    ir_rules.py           IR validation rules (pure functions)
    layout.py             LessonPlan -> SceneIR
    codegen.py            IR scene -> Manim source
    spine.py              document -> mp4, with no agent around it
    repair.py             repair + escalation ladder
    agent.py              state, tool dispatch, budget
    video.py              ffmpeg concat
    metrics.py            attempt log, run summary
  docs/
  tests/
```

## Model usage

Default to `claude-opus-5` with adaptive thinking. Use structured outputs
(`output_config.format` with a JSON schema) for Lesson Plan and Scene IR --
both are schemas, and parsing prose into them is a failure mode worth
designing out rather than repairing. Codegen returns source, so it stays plain
text with a fenced-block extractor.

Cache the parts of the prompt that do not change between scenes -- the Manim
API guidance and the IR spec are the same on every codegen call, and they are
the bulk of the tokens.
