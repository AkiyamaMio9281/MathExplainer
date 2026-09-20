"""Tiered validation of generated Manim code.

The repair loop re-checks the code after every model edit, so how fast a check
runs decides how many repair rounds are affordable. Measured on this machine
against a five-animation scene:

    L0 syntax (ast.parse)        0.07 ms
    L1 import + instantiate      0.97 s
    L2 manim --dry_run           1.31 s
    L3 manim -ql (full render)   1.57 s

Two things follow. Syntax checking is free, so it always runs first and a
malformed edit never costs a subprocess. And L1-L3 differ by far less than
their absolute cost -- roughly a second of that is Python start-up plus
importing manim, which every tier pays. The gap between dry-run and render
widens with scene complexity (this scene is trivial), so the loop still stops
at L2 and renders once at the end, but the saving is smaller than the tier
names suggest. Re-measure on real scenes before assuming otherwise.
"""

from __future__ import annotations

import ast
import enum
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Completed, run_python

SCENE_FILE = "scene.py"


class Level(enum.IntEnum):
    """Validation depth. Higher levels subsume the ones below."""

    SYNTAX = 0     # parses as Python
    IMPORT = 1     # module executes; a Scene subclass exists and constructs
    DRYRUN = 2     # construct() runs to completion; nothing is written
    RENDER = 3     # rasterised and encoded to a video file


@dataclass(frozen=True)
class Check:
    level: Level
    ok: bool
    error: str = ""
    seconds: float = 0.0
    scene_name: str | None = None
    output: Path | None = None

    @property
    def failed(self) -> bool:
        return not self.ok


def scene_class_names(code: str) -> list[str]:
    """Class names in *code* that look like Manim scenes.

    Found by AST rather than by fixing the name in the prompt: the model names
    the class after the topic often enough that pinning it is a repair round
    spent on nothing that matters.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
            if base_name.endswith("Scene"):
                names.append(node.name)
                break
    return names


def check_syntax(code: str) -> Check:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        where = f"line {exc.lineno}" if exc.lineno else "unknown line"
        return Check(Level.SYNTAX, False, f"SyntaxError at {where}: {exc.msg}")
    names = scene_class_names(code)
    if not names:
        return Check(
            Level.SYNTAX,
            False,
            "No Scene subclass found. Define exactly one class inheriting from Scene.",
        )
    if len(names) > 1:
        # One Manim Scene per IR scene is an architectural invariant, not a
        # style preference: manim renders the class it is told to and the
        # others silently never appear. Rejecting it here costs 0.07 ms;
        # discovering it after the render costs a scene.
        return Check(
            Level.SYNTAX,
            False,
            f"Found {len(names)} Scene subclasses ({', '.join(names)}). "
            "Define exactly one class inheriting from Scene.",
        )
    return Check(Level.SYNTAX, True, scene_name=names[0])


# Executes the module and constructs the scene without running construct(),
# which separates "the file is broken" from "the animation is broken".
_IMPORT_DRIVER = """
import json, sys, traceback
path, name = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
ns = {}
try:
    exec(compile(src, path, "exec"), ns)
    ns[name]()
except Exception:
    print(json.dumps({"ok": False, "error": traceback.format_exc()}))
    sys.exit(1)
print(json.dumps({"ok": True}))
"""


def check_import(workdir: Path, scene_name: str, timeout: float = 120.0) -> Check:
    workdir.mkdir(parents=True, exist_ok=True)
    driver = workdir / "_import_check.py"
    driver.write_text(_IMPORT_DRIVER, encoding="utf-8")
    # By name, not by path. The child's cwd *is* workdir, so a relative path
    # built from this process's cwd gets resolved a second time against it --
    # runs/x/scene/runs/x/scene/_import_check.py, and an error about a file
    # nobody named. Absolute workdirs hide it, which is why every test that
    # used pytest's tmp_path passed while the CLI did not.
    done = run_python(
        [driver.name, SCENE_FILE, scene_name], cwd=workdir, timeout=timeout
    )
    if done.ok:
        return Check(Level.IMPORT, True, seconds=done.seconds, scene_name=scene_name)
    return Check(
        Level.IMPORT, False, _driver_error(done), done.seconds, scene_name=scene_name
    )


def check_dry_run(workdir: Path, scene_name: str, timeout: float = 300.0) -> Check:
    done = run_python(
        ["-m", "manim", "-ql", "--dry_run", "--disable_caching", SCENE_FILE, scene_name],
        cwd=workdir,
        timeout=timeout,
    )
    return Check(
        Level.DRYRUN,
        done.ok,
        "" if done.ok else done.failure_text(),
        done.seconds,
        scene_name=scene_name,
    )


def render(
    workdir: Path,
    scene_name: str,
    quality: str = "l",
    timeout: float = 1800.0,
) -> Check:
    """Render to a video file. *quality* is manim's l/m/h/p/k."""
    # manim never cleans media/, so a video left by an earlier render of
    # this workdir is indistinguishable from this run's output, and picking
    # the newest match is a guess rather than an answer. Clearing costs
    # nothing: --disable_caching means there is no cache to lose.
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(workdir / "media", ignore_errors=True)
    done = run_python(
        ["-m", "manim", f"-q{quality}", "--disable_caching", SCENE_FILE, scene_name],
        cwd=workdir,
        timeout=timeout,
    )
    if not done.ok:
        return Check(
            Level.RENDER, False, done.failure_text(), done.seconds, scene_name=scene_name
        )
    produced = _find_video(workdir, scene_name)
    if produced is None:
        return Check(
            Level.RENDER,
            False,
            "manim exited 0 but no video file was produced.",
            done.seconds,
            scene_name=scene_name,
        )
    return Check(
        Level.RENDER, True, seconds=done.seconds, scene_name=scene_name, output=produced
    )


def validate(workdir: Path, code: str, up_to: Level = Level.DRYRUN) -> Check:
    """Run each tier in order, returning the first failure or the last success."""
    result = check_syntax(code)
    if result.failed or up_to == Level.SYNTAX:
        return result

    scene_name = result.scene_name or ""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / SCENE_FILE).write_text(code, encoding="utf-8")

    result = check_import(workdir, scene_name)
    if result.failed or up_to == Level.IMPORT:
        return result

    result = check_dry_run(workdir, scene_name)
    if result.failed or up_to == Level.DRYRUN:
        return result

    return render(workdir, scene_name)


def _driver_error(done: Completed) -> str:
    """Pull the traceback out of the import driver's JSON, or fall back to raw
    output when it died before it could report."""
    for line in reversed(done.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
        except ValueError:
            continue
        if isinstance(payload, dict) and "error" in payload:
            return str(payload["error"])
    return done.failure_text()


def _find_video(workdir: Path, scene_name: str) -> Path | None:
    """The video this run produced. ``render`` clears media/ beforehand,
    so anything found here belongs to the render that just finished rather
    than to whatever happened to be newest."""
    candidates = sorted((workdir / "media" / "videos").rglob(f"{scene_name}.mp4"))
    if not candidates:
        return None
    return candidates[0]
