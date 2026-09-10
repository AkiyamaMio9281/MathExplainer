# Scene IR

A declarative description of what is on screen and when. It is data, not code:
it can be validated, diffed, simplified, and rendered by something other than
Manim.

## Coordinate system

Manim units, taken from the default config and **independent of render
quality** (`-ql` changes pixels, not units):

```
        y = +4.000
            |
x = -7.111 -+- x = +7.111        frame_width  = 14.222
            |                    frame_height =  8.000
        y = -4.000
```

Origin is the centre of the frame. `[0, 0]` is the middle, `[0, 3]` is near the
top, `[-6, 0]` is near the left edge.

Keep a margin: the validator flags anything whose bounding box extends past
`|x| > 6.6` or `|y| > 3.6`, because text rendered at a position near the edge
spills past it.

## Document shape

```json
{
  "version": "1",
  "title": "The Pythagorean Theorem",
  "scenes": [
    {
      "id": "intro",
      "narration": "Every right triangle hides one simple relationship between its sides.",
      "objects": [
        {"id": "title", "type": "text", "content": "The Pythagorean Theorem",
         "position": [0, 3], "font_size": 48},
        {"id": "tri", "type": "polygon",
         "points": [[-3, -1], [0, -1], [0, 1.5]], "color": "BLUE"},
        {"id": "eq", "type": "mathtex", "content": "a^2 + b^2 = c^2",
         "position": [3.2, 0], "font_size": 56}
      ],
      "steps": [
        {"action": "write",    "target": "title", "duration": 1.5},
        {"action": "create",   "target": "tri",   "duration": 2.0},
        {"action": "write",    "target": "eq",    "duration": 1.5},
        {"action": "indicate", "target": "eq",    "duration": 1.0},
        {"action": "wait",                        "duration": 1.0}
      ]
    }
  ]
}
```

`objects` declares everything the scene can show. `steps` is the timeline. An
object is not visible until a step introduces it.

## Object types

Every object has `id` (unique within the scene) and `type`. `color` accepts
Manim colour names (`WHITE`, `BLUE`, `YELLOW`, `GREEN`, `RED`, `ORANGE`,
`PURPLE`, `GREY`) and defaults to `WHITE`.

| `type` | Required | Optional |
|---|---|---|
| `text` | `content`, `position` | `font_size` (default 36), `color` |
| `mathtex` | `content` (LaTeX, no `$`), `position` | `font_size` (default 48), `color` |
| `polygon` | `points` (>= 3 `[x,y]`) | `color`, `fill_opacity` (default 0) |
| `circle` | `center`, `radius` | `color`, `fill_opacity` |
| `rectangle` | `center`, `width`, `height` | `color`, `fill_opacity` |
| `line` | `start`, `end` | `color`, `stroke_width` |
| `arrow` | `start`, `end` | `color` |
| `axes` | `x_range` `[min,max,step]`, `y_range` | `position` (default `[0,0]`) |
| `plot` | `axes` (id of an `axes`), `expression` | `color`, `x_range` |

`plot.expression` is a Python expression in `x`, e.g. `"np.sin(x)"` or
`"x**2 - 1"`. Codegen wraps it in a lambda; nothing else in the IR contains
executable text.

## Actions

Every step has `action` and `duration` (seconds, > 0). All except `wait` have
`target`.

| `action` | Applies to | Effect |
|---|---|---|
| `write` | `text`, `mathtex` | draws it on, stroke by stroke |
| `create` | shapes, `axes`, `plot` | draws it on |
| `fade_in` | any | fades it in |
| `fade_out` | any | fades it out; it is gone afterwards |
| `transform` | any, plus `into` (another object id) | morphs target into `into` |
| `move` | any, plus `to` (`[x,y]`) | translates it |
| `indicate` | any | brief attention flash, leaves it on screen |
| `wait` | -- | holds the frame |

`write` / `create` / `fade_in` are **introductions**: they are what makes an
object visible. `transform` consumes its target and leaves `into` on screen in
its place.

## Validation rules

Pure functions over the IR. No LLM, no subprocess -- these run in
milliseconds, and every defect caught here is one that would otherwise survive
a clean compile and only surface when a human watched the video.

**Errors** (the IR is rejected and regenerated):

| Rule | Check |
|---|---|
| `unique-ids` | object ids are unique within a scene, and scene ids within the document |
| `safe-ids` | every id is a lowercase identifier: `[a-z][a-z0-9_]*`, not a Python keyword |
| `known-type` | `type` is in the table above |
| `required-fields` | every required field for that type is present |
| `reference-integrity` | every `target` / `into` names a declared object, and `plot.axes` a declared `axes` |
| `introduced-before-use` | no action on an object before it is introduced |
| `not-after-removal` | no action on an object after `fade_out`, or after it is a `transform` target |
| `positive-duration` | every `duration` > 0 |
| `non-empty` | the document has >= 1 scene; each scene has >= 1 object and >= 1 step |
| `action-applies` | `write` only on `text`/`mathtex`; `create` only on the rest |
| `document-shape` | the document, and its `scenes`/`objects`/`steps`, are the right kind of container |
| `known-version` | `version` is the one this document describes |
| `known-action` | `action` is in the table above |
| `field-shape` | each field is the right kind of value: a point is two numbers, a range is three |

Four of these are decided while the document is being typed, and so live in
`explainer/ir.py` rather than in `ir_rules.py`: `known-type`, `known-action`,
`required-fields` and `field-shape` all decide whether a typed object can be
built at all, and the rest need a built one in order to run. The parser
collects them rather than raising, so one repair round carries every defect
instead of the first; anything it could not type is dropped from the document
and reported, which is why a document with errors is a subset of what the
model wrote and is never rendered in that state.

`safe-ids` is the one rule here with a consequence outside the video. A scene
id becomes a working directory and an object id becomes a local variable in
the generated Python, so both are model-written text that ends up somewhere it
can do damage -- an id of `../../elsewhere` writes outside the directory the
run was given. Lowercase is not a style preference either: it is what keeps an
id clear of Manim's CamelCase classes and its ALLCAPS colour constants.

**Warnings** (recorded, may trigger `simplify_ir`, do not block rendering):

| Rule | Check | Why |
|---|---|---|
| `in-frame` | every position / point within `|x| <= 6.6`, `|y| <= 3.6` | objects placed off the edge are invisible in the output but render without error |
| `no-overlap` | approximate bounding boxes of simultaneously-visible objects do not intersect | overlapping text is the single most common way a generated scene looks broken |
| `unknown-field` | a field the object type or action does not use | usually the model confusing two types; the value is reported and dropped rather than silently carried |
| `pacing` | `len(narration.split()) / 150 * 60` within 30% of `sum(step durations)` | narration and animation drift apart; 150 wpm is a normal explainer pace |
| `scene-length` | total duration between 5 s and 90 s | very short scenes look like glitches, very long ones lose the viewer |

`in-frame` and `no-overlap` are layout-defect detection on a declarative
document -- structurally the same problem as finding layout defects in a
rendered DOM, and worth naming that way when writing this up.

### Bounding boxes

Exact text extents need a font engine, which the validator deliberately does
not load. Approximate instead, and treat `no-overlap` as a warning because the
estimate is rough. Per `font_size / 36`, centred on `position`:

| Type | Width | Height |
|---|---|---|
| `text` | `0.25 * len(content)` | `0.50` |
| `mathtex` | `0.16 * len(content)` | `0.60` |
| shapes | exact, from their points / centre and radius | |
| `axes` | `12`, whatever the range | `6` |
| `plot` | the box of the `axes` it is drawn on | |

**These constants are measured, against ManimCE 0.21.0.** The first draft of
this document guessed `0.55` per character and `1.0` tall, which is about
twice what manim actually produces: under those numbers the title in the
example above came out 16.9 units wide against a 14.2-unit frame, so
`in-frame` rejected this document's own example. Measured, a character is
0.21-0.26 units wide and a line 0.24-0.49 tall per `font_size / 36`; the
values in the table sit just above the means, because an estimate slightly too
large turns a near miss into a warning while one that is too large by a factor
of two turns every scene into one. `tests/test_ir_rules.py` re-measures against
manim, so these do not quietly rot.

Two corrections worth stating separately. **Axes do not take their size from
`x_range` / `y_range`** -- manim sizes them from the frame and uses the range
for tick labels, so an axes is 12 by 6 whatever it plots. And because that is
most of the frame, **`axes` and `plot` are left out of `no-overlap`**: they are
backdrops that other objects are meant to be drawn over, and counting them
would fire on nearly every scene that plots anything. `no-overlap` exists to
catch overlapping text.

`move` is modelled as translating an object's box so its centre lands on `to`,
which works for every type without asking which of `position`, `center`,
`start` or `points` it happens to carry.

## Simplification

`simplify_ir` is the escalation rung between repairing code and dropping a
scene. It reduces what a scene asks for, in this order, and the code is
regenerated afterwards:

1. Drop `plot` and `axes` objects (the most failure-prone type) and any steps
   targeting them.
2. Drop `transform` steps, replacing them with `fade_out` then `fade_in`.
3. Reduce to at most three objects, keeping the ones with the most steps.
4. Flatten to introductions only -- one `write`/`create` per object, then
   `wait`.

Step 4 always produces something renderable, so the ladder terminates.

## Open questions for the implementer

- **Narration is text only.** No TTS, and no audio track in the MP4. If audio
  gets added, `duration` becomes an output of the TTS rather than an input to
  it, and `pacing` becomes an error rather than a warning.
- **No camera moves, no 3D.** `ThreeDScene` and camera work are out of scope
  for v1; a `camera` action would be the natural v2 extension.
- **Colour is a name, not a value.** Hex would be more expressive but makes the
  contrast checking a future validator would want harder to skip.
