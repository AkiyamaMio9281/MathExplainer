"""Toolchain smoke test.

Exercises the three things that must work before anything else does: Cairo
rasterisation, the LaTeX path (MathTex shells out to latex), and ffmpeg
encoding. If this renders, the environment is sound.

    python -m manim -ql --disable_caching docs/examples/smoke.py Smoke
"""

from manim import *


class Smoke(Scene):
    def construct(self):
        title = Text("Pythagorean Theorem", font_size=40)
        formula = MathTex(r"a^2 + b^2 = c^2", font_size=56)   # the LaTeX path
        triangle = Polygon([-2, -1, 0], [1, -1, 0], [1, 1, 0], color=BLUE)

        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        self.play(Create(triangle))
        self.play(Write(formula.next_to(triangle, RIGHT, buff=1)))
        self.wait(0.5)
