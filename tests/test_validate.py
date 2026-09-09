"""Each validation tier catches what it claims and lets through what it should.

The matrix is the table in HANDOFF.md section 4. The row that carries the
design is the undefined name used only inside ``construct()``: it passes L1 and
fails at L2, because L1 executes the module and builds the Scene without
running the animation. That split is the entire justification for having two
subprocess tiers instead of one, so it is asserted from both sides here --
import succeeds, dry run fails -- rather than by checking the level a single
``validate`` call happened to stop at.

Everything that spawns a subprocess is marked ``slow``; the tier costs about a
second of fixed Python start-up before it does any work. Tests call the tier
functions directly where they can, so a test about L2 does not pay for L1.

Scenes here use ``Text`` rather than ``MathTex``: the LaTeX path is already
covered by docs/examples/smoke.py, and it is the slowest thing in the
toolchain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explainer.validate import (
    SCENE_FILE,
    Level,
    check_dry_run,
    check_import,
    check_syntax,
    render,
    scene_class_names,
    validate,
)

VALID = """\
from manim import *


class Demo(Scene):
    def construct(self):
        self.play(Write(Text("hi")))
        self.wait(0.2)
"""

# The missing colon is on line 2, which is what the reported line number is
# checked against below.
MISSING_COLON = """\
from manim import *
class Demo(Scene)
    def construct(self):
        pass
"""

NO_SCENE = """\
from manim import *


def construct():
    pass
"""

TWO_SCENES = VALID + """

class Extra(Scene):
    def construct(self):
        self.wait(0.1)
"""

# Fails while the module is executing, so it never reaches construct().
BAD_IMPORT = """\
from manim import *
import mx_no_such_module


class Demo(Scene):
    def construct(self):
        self.wait(0.1)
"""

# Parses, imports, and constructs cleanly. Only running the animation finds it.
UNDEFINED_NAME = """\
from manim import *


class Demo(Scene):
    def construct(self):
        self.play(Write(NoSuchThing()))
        self.wait(0.1)
"""

DIVIDE_BY_ZERO = """\
from manim import *


class Demo(Scene):
    def construct(self):
        self.wait(1 / 0)
"""


def write_scene(workdir: Path, code: str) -> str:
    """Put *code* in *workdir* as scene.py; return its Scene class name."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / SCENE_FILE).write_text(code, encoding="utf-8")
    return scene_class_names(code)[0]


# --------------------------------------------------------------------------
# L0: free, and therefore always first
# --------------------------------------------------------------------------


def test_a_valid_scene_passes_syntax_and_names_its_class():
    result = check_syntax(VALID)
    assert result.ok
    assert result.level == Level.SYNTAX
    assert result.scene_name == "Demo"


def test_missing_colon_fails_at_syntax_with_a_line_number():
    result = check_syntax(MISSING_COLON)
    assert result.failed
    assert result.level == Level.SYNTAX
    # The line number is this project's contribution to the message; the text
    # after it is CPython's and varies between versions, so it is not pinned.
    assert "SyntaxError at line 2" in result.error


def test_a_file_with_no_scene_subclass_fails_at_syntax():
    result = check_syntax(NO_SCENE)
    assert result.failed
    assert "No Scene subclass found" in result.error


def test_two_scene_subclasses_fail_at_syntax():
    result = check_syntax(TWO_SCENES)
    assert result.failed
    assert "Found 2 Scene subclasses" in result.error
    # Naming them is the point: the repair prompt has to know which to drop.
    assert "Demo" in result.error and "Extra" in result.error


def test_syntax_level_writes_nothing_and_spawns_nothing(tmp_path):
    # L0 costs 0.07 ms only if it really does stop before touching the disk.
    result = validate(tmp_path, VALID, up_to=Level.SYNTAX)
    assert result.ok
    assert result.level == Level.SYNTAX
    assert not (tmp_path / SCENE_FILE).exists()


# --------------------------------------------------------------------------
# scene_class_names: found by AST, not by pinning a name in the prompt
# --------------------------------------------------------------------------


def test_scene_classes_are_found_through_a_dotted_base():
    assert scene_class_names("import manim\n\nclass A(manim.Scene):\n    pass\n") == ["A"]


def test_scene_subclass_kinds_other_than_scene_count():
    code = "from manim import *\n\nclass A(ThreeDScene):\n    pass\n"
    assert scene_class_names(code) == ["A"]


def test_non_scene_classes_are_ignored():
    code = "from manim import *\n\nclass Helper:\n    pass\n\nclass A(Scene):\n    pass\n"
    assert scene_class_names(code) == ["A"]


def test_unparseable_code_yields_no_scene_names():
    assert scene_class_names(MISSING_COLON) == []


# --------------------------------------------------------------------------
# L1 vs L2: the split that justifies two subprocess tiers
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_module_level_import_error_stops_at_import(tmp_path):
    name = write_scene(tmp_path, BAD_IMPORT)
    result = check_import(tmp_path, name)

    assert result.failed
    assert result.level == Level.IMPORT
    assert "mx_no_such_module" in result.error


@pytest.mark.slow
def test_an_undefined_name_in_construct_passes_import_and_fails_at_dry_run(tmp_path):
    name = write_scene(tmp_path, UNDEFINED_NAME)

    # L1 executes the module and constructs the Scene, but not construct(),
    # so it cannot see this.
    imported = check_import(tmp_path, name)
    assert imported.ok, imported.error

    # L2 runs the animation, and this is where it surfaces.
    dry_run = check_dry_run(tmp_path, name)
    assert dry_run.failed
    assert dry_run.level == Level.DRYRUN
    assert "NameError" in dry_run.error
    assert "NoSuchThing" in dry_run.error


@pytest.mark.slow
def test_an_exception_in_construct_fails_at_dry_run(tmp_path):
    name = write_scene(tmp_path, DIVIDE_BY_ZERO)
    result = check_dry_run(tmp_path, name)

    assert result.failed
    assert "ZeroDivisionError" in result.error


@pytest.mark.slow
def test_a_valid_scene_reaches_dry_run(tmp_path):
    result = validate(tmp_path, VALID, up_to=Level.DRYRUN)

    assert result.ok, result.error
    assert result.level == Level.DRYRUN
    assert result.scene_name == "Demo"
    # --dry_run is the tier that writes nothing.
    assert not list(tmp_path.rglob("Demo.mp4"))


# --------------------------------------------------------------------------
# L3: the only tier that produces a file
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_rendering_produces_a_video(tmp_path):
    result = validate(tmp_path, VALID, up_to=Level.RENDER)

    assert result.ok, result.error
    assert result.level == Level.RENDER
    assert result.output is not None
    assert result.output.name == "Demo.mp4"
    assert result.output.stat().st_size > 0


@pytest.mark.slow
def test_rendering_ignores_a_video_left_by_an_earlier_run(tmp_path):
    # The workdir is reused across repair rounds by design, so a leftover file
    # is exactly the situation where the wrong video would be returned.
    name = write_scene(tmp_path, VALID)
    stale = tmp_path / "media" / "videos" / "leftover" / "480p15" / "Demo.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale" * 100)

    result = render(tmp_path, name)

    assert result.ok, result.error
    assert not stale.exists(), "the earlier render was left in place"
    assert result.output.read_bytes()[:5] != b"stale"
