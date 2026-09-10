"""One IR scene to one Manim source file.

The prompt is the work here, and it has two halves that are deliberately kept
apart. `PREAMBLE` is the Manim guidance and the IR vocabulary -- identical on
every call, several thousand tokens, and cached. The scene itself is the user
message and is all that changes. Measured on this model, a repeated preamble
costs about a twelfth of its first send, and the preamble dominates the
tokens, so the split is most of what codegen costs.

**The IR is rendered to JSON rather than described in prose.** The model
already receives the vocabulary in the preamble; sending the scene as the same
JSON the validator reads means there is exactly one description of a scene in
the system, and a repair round argues about the code rather than about what
the scene was supposed to be.

Output is source, not a schema, so it comes back as plain text and is pulled
out of a fenced block. `extract_code` is written for what models actually
emit -- a fence with or without a language tag, occasionally prose either side
of it, occasionally no fence at all -- because every shape it fails to handle
costs a repair round on a file that was already correct.

Nothing here validates the result. `validate.py` owns that, and keeping the
two apart is what lets the repair loop feed a traceback back into `repair`
without regenerating from the IR.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from . import ir, llm

SCENE_CLASS_HINT = "Scene"

#: The half of the prompt that never changes. Cached.
PREAMBLE = """\
You write Manim Community Edition scenes. You are given one scene of a lesson,
described in a small declarative format, and you return one Python file that
renders it.

# The format

A scene has `objects` -- everything it can show -- and `steps`, the timeline.
An object is not visible until a step introduces it.

Coordinates are Manim units. The frame is 14.222 by 8.0, so x runs from -7.111
to 7.111 and y from -4.0 to 4.0, with [0, 0] at the centre. A position of
[0, 3] is near the top. Positions are 2D; Manim wants 3D points, so write
`[x, y, 0]`.

Object types and the fields they carry:

- `text`: content, position, font_size (default 36), color -> `Text`
- `mathtex`: content (LaTeX without $), position, font_size (default 48),
  color -> `MathTex`
- `polygon`: points (3 or more [x, y]), color, fill_opacity -> `Polygon`
- `circle`: center, radius, color, fill_opacity -> `Circle`
- `rectangle`: center, width, height, color, fill_opacity -> `Rectangle`
- `line`: start, end, color, stroke_width -> `Line`
- `arrow`: start, end, color -> `Arrow`
- `axes`: x_range [min, max, step], y_range, position -> `Axes`
- `plot`: axes (the id of an axes object), expression, color, x_range ->
  `axes_object.plot(lambda x: ..., x_range=...)`

`plot.expression` is a Python expression in `x`, such as `np.sin(x)` or
`x**2 - 1`. Wrap it in a lambda. It is the only executable text in the input.

Actions, each with a duration in seconds:

- `write` (text and mathtex only) -> `self.play(Write(obj), run_time=d)`
- `create` (everything else) -> `self.play(Create(obj), run_time=d)`
- `fade_in` -> `FadeIn`
- `fade_out` -> `FadeOut`; the object is gone afterwards
- `transform` with `into` -> `self.play(Transform(obj, into), run_time=d)`;
  the target is consumed and `into` is left in its place
- `move` with `to` -> `self.play(obj.animate.move_to([x, y, 0]), run_time=d)`
- `indicate` -> `self.play(Indicate(obj), run_time=d)`; the object stays
- `wait` -> `self.wait(d)`

Colours are Manim constants: WHITE, BLUE, YELLOW, GREEN, RED, ORANGE, PURPLE,
GREY.

# What to return

One Python file, in a single ```python fenced block, and nothing else.

- Start with `from manim import *`. Add `import numpy as np` only if a plot
  expression needs it.
- Define exactly one class inheriting from `Scene`. More than one is rejected
  before it is ever rendered. Name it after the topic in CamelCase.
- Put every object in `construct`, as a local variable named after its id.
- Follow the steps in order. Every `self.play` takes `run_time` from the
  step's duration.
- Set each object's position when it is constructed, using `.move_to`,
  `.shift`, or the constructor's own arguments.
- Use only Manim's public API. Do not read files, import anything beyond manim
  and numpy, or use `os`, `sys`, or `subprocess`.
- No comments explaining what the code obviously does. No `if __name__`
  block. No `config` changes.

Return the file and nothing else -- no explanation before or after it.
"""

_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def scene_prompt(scene: ir.Scene, title: str = "") -> str:
    """The user half: this scene, as the JSON the validator reads."""
    document = {"title": title, "scene": _scene_to_dict(scene)}
    lines = [
        "Render this scene.",
        "",
        json.dumps(document, indent=2),
    ]
    if scene.narration:
        lines += [
            "",
            "The narration below is spoken over the animation. It is context for "
            "what the scene is about; do not put it on screen.",
            "",
            scene.narration,
        ]
    return "\n".join(lines)


def _scene_to_dict(scene: ir.Scene) -> dict:
    # ir.Document.to_dict is the canonical renderer; borrow it for one scene
    # so there is no second description of a scene to drift from the first.
    return ir.Document(scenes=(scene,)).to_dict()["scenes"][0]


def extract_code(text: str) -> str:
    """The Python out of a model reply.

    Written for what actually comes back rather than for the documented ideal:
    a fenced block with or without a language tag, sometimes with prose either
    side, sometimes no fence at all. Each shape this fails to handle costs a
    repair round on a file that was already correct.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        # The longest, not the first: a reply that opens with a one-line
        # snippet before the file would otherwise hand back the snippet.
        return max(blocks, key=len).strip()
    return text.strip()


def generate(
    client: llm.Client,
    scene: ir.Scene,
    *,
    title: str = "",
    effort: str | None = None,
) -> tuple[str, llm.Reply]:
    """Manim source for one scene, and what the call cost."""
    reply = client.complete(
        scene_prompt(scene, title), preamble=PREAMBLE, effort=effort
    )
    return extract_code(reply.text), reply


def repair_prompt(code: str, error: str, limit: int = 4000) -> str:
    """Ask for a fix, given the code and what the checker said.

    The traceback goes in whole and last. Summarising it is how a repair round
    gets spent on the wrong line.
    """
    if len(error) > limit:
        error = "...\n" + error[-limit:]
    return "\n".join(
        [
            "This file failed to run. Return the corrected file, in one",
            "```python fenced block, with nothing else.",
            "",
            "```python",
            code,
            "```",
            "",
            "It failed with:",
            "",
            "```",
            error,
            "```",
        ]
    )


def repair(
    client: llm.Client,
    code: str,
    error: str,
    *,
    effort: str | None = None,
) -> tuple[str, llm.Reply]:
    """One repair round. The preamble is the same, so it stays cached."""
    reply = client.complete(
        repair_prompt(code, error), preamble=PREAMBLE, effort=effort
    )
    return extract_code(reply.text), reply


def scene_ids(scenes: Iterable[ir.Scene]) -> list[str]:
    return [scene.id for scene in scenes]
