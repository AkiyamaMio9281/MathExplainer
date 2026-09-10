"""Dataclasses for the Scene IR, and the parser that produces them.

`docs/SCENE_IR.md` is the specification; this is its typed form. The IR is data
rather than code so that it can be validated, diffed, simplified, and rendered
by something that is not Manim -- so nothing in this module imports Manim or
names it.

Three decisions here shape everything downstream.

**Parsing collects issues instead of raising.** A model that produced a
malformed document gets one repair round, and that round should carry every
structural defect rather than the first one; regenerating once per missing
field is an expensive way to discover there were three. `parse_document`
therefore always returns a `ParseResult` -- the parts it could type, plus an
`Issue` for each part it could not.

**Anything that cannot be typed is dropped rather than half-built.** An object
missing its `content` is not represented as a `Text` with ``content=None``; it
is recorded as an issue and left out. Everything downstream -- the rules, the
code generator, simplification -- therefore sees only well-formed objects and
never has to test a field for None. The cost is that a document with errors is
a *subset* of what the model wrote, which is why such a document is never
meant to be rendered: check `ParseResult.ok` first.

**Required fields are the fields without defaults.** The table in SCENE_IR.md
and the dataclasses below are the same statement, and `_required_names` reads
it off the dataclass rather than repeating it in a second list that can drift.

The rules needing a whole typed document -- reference integrity, the timeline,
the frame, pacing -- live in `ir_rules.py`. The split is not structural versus
semantic; it is "rules that decide whether a typed document can exist at all"
against "rules that need one in order to run".

Colour is deliberately not validated. `docs/SCENE_IR.md` lists the names Manim
accepts, but a wrong one fails at L1 with a plain NameError naming the
constant, so a rule here would only duplicate a check the code tiers already
do well.
"""

from __future__ import annotations

import enum
import json
import keyword
import re
from dataclasses import MISSING, dataclass
from dataclasses import fields as dc_fields
from typing import Any, ClassVar, Iterable, Mapping, Sequence

VERSION = "1"
DEFAULT_COLOR = "WHITE"

#: Ids are not free-form labels. A scene id becomes a working directory, and
#: an object id becomes a local variable in generated Python, so both are
#: model-supplied text that ends up somewhere it can do damage. Lowercase is
#: not style: it is what keeps an id from colliding with Manim's CamelCase
#: classes or its ALLCAPS colour constants.
SAFE_ID = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def is_safe_id(name: str) -> bool:
    """Whether *name* can be a directory and a Python local without trouble."""
    return bool(SAFE_ID.match(name)) and not keyword.iskeyword(name)

Point = tuple[float, float]
Range = tuple[float, float, float]

# Returned by the coercers when a value is present but the wrong shape. None
# cannot serve here: it is a legitimate value for an absent optional field.
_INVALID = object()


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """One defect, located well enough to repair without guessing."""

    rule: str
    message: str
    where: str
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        return f"{self.where}: {self.message} [{self.rule}]"


class Action(str, enum.Enum):
    WRITE = "write"
    CREATE = "create"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    TRANSFORM = "transform"
    MOVE = "move"
    INDICATE = "indicate"
    WAIT = "wait"


#: The actions that make an object visible. An object is not on screen until
#: one of these introduces it, which is what `introduced-before-use` checks.
INTRODUCTIONS = frozenset({Action.WRITE, Action.CREATE, Action.FADE_IN})

#: Actions that take no target.
UNTARGETED = frozenset({Action.WAIT})


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class SceneObject:
    """Base for everything ``objects`` can hold.

    ``kw_only`` is what lets a subclass declare a field without a default
    after the base has declared one with a default.
    """

    TYPE: ClassVar[str] = ""

    id: str
    color: str = DEFAULT_COLOR


@dataclass(frozen=True, kw_only=True)
class Text(SceneObject):
    TYPE: ClassVar[str] = "text"

    content: str
    position: Point
    font_size: float = 36.0


@dataclass(frozen=True, kw_only=True)
class MathTex(SceneObject):
    TYPE: ClassVar[str] = "mathtex"

    content: str  # LaTeX, without surrounding $
    position: Point
    font_size: float = 48.0


@dataclass(frozen=True, kw_only=True)
class Polygon(SceneObject):
    TYPE: ClassVar[str] = "polygon"

    points: tuple[Point, ...]
    fill_opacity: float = 0.0


@dataclass(frozen=True, kw_only=True)
class Circle(SceneObject):
    TYPE: ClassVar[str] = "circle"

    center: Point
    radius: float
    fill_opacity: float = 0.0


@dataclass(frozen=True, kw_only=True)
class Rectangle(SceneObject):
    TYPE: ClassVar[str] = "rectangle"

    center: Point
    width: float
    height: float
    fill_opacity: float = 0.0


@dataclass(frozen=True, kw_only=True)
class Line(SceneObject):
    TYPE: ClassVar[str] = "line"

    start: Point
    end: Point
    # None means "whatever the renderer's default is". Putting Manim's 4 here
    # would write a Manim constant into a renderer-agnostic document.
    stroke_width: float | None = None


@dataclass(frozen=True, kw_only=True)
class Arrow(SceneObject):
    TYPE: ClassVar[str] = "arrow"

    start: Point
    end: Point


@dataclass(frozen=True, kw_only=True)
class Axes(SceneObject):
    TYPE: ClassVar[str] = "axes"

    x_range: Range
    y_range: Range
    position: Point = (0.0, 0.0)


@dataclass(frozen=True, kw_only=True)
class Plot(SceneObject):
    TYPE: ClassVar[str] = "plot"

    axes: str  # id of an Axes object
    # A Python expression in x. The only executable text in the whole IR;
    # codegen wraps it in a lambda.
    expression: str
    x_range: Range | None = None


OBJECT_TYPES: dict[str, type[SceneObject]] = {
    cls.TYPE: cls
    for cls in (Text, MathTex, Polygon, Circle, Rectangle, Line, Arrow, Axes, Plot)
}


# ---------------------------------------------------------------------------
# Steps, scenes, documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Step:
    """One entry on the timeline.

    A single shape rather than a class per action: the actions differ only in
    whether they carry ``into`` or ``to``, and the timeline walk in ir_rules
    wants uniform access to ``target``.
    """

    action: Action
    duration: float
    target: str | None = None
    into: str | None = None  # transform only
    to: Point | None = None  # move only


@dataclass(frozen=True, kw_only=True)
class Scene:
    """One IR scene. Not a Manim ``Scene`` -- always qualify as ``ir.Scene``."""

    id: str
    narration: str = ""
    objects: tuple[SceneObject, ...] = ()
    steps: tuple[Step, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Document:
    version: str = VERSION
    title: str = ""
    scenes: tuple[Scene, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "title": self.title,
            "scenes": [_scene_to_dict(scene) for scene in self.scenes],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class ParseResult:
    document: Document | None
    issues: tuple[Issue, ...] = ()

    @property
    def errors(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Issue, ...]:
        return tuple(i for i in self.issues if i.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        return self.document is not None and not self.errors

    def report(self) -> str:
        """The issues as text, for feeding back to the model."""
        return "\n".join(str(i) for i in self.issues)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonify(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    return value


def _scene_object_to_dict(obj: SceneObject) -> dict[str, Any]:
    out: dict[str, Any] = {"id": obj.id, "type": type(obj).TYPE}
    for f in dc_fields(obj):
        if f.name == "id":
            continue
        value = getattr(obj, f.name)
        if value is None:
            continue  # an absent optional stays absent
        out[f.name] = _jsonify(value)
    return out


def _step_to_dict(step: Step) -> dict[str, Any]:
    out: dict[str, Any] = {"action": step.action.value, "duration": step.duration}
    for name in ("target", "into", "to"):
        value = getattr(step, name)
        if value is not None:
            out[name] = _jsonify(value)
    return out


def _scene_to_dict(scene: Scene) -> dict[str, Any]:
    return {
        "id": scene.id,
        "narration": scene.narration,
        "objects": [_scene_object_to_dict(o) for o in scene.objects],
        "steps": [_step_to_dict(s) for s in scene.steps],
    }


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------

# Dispatch is by field name rather than by annotation: `from __future__ import
# annotations` makes annotations strings, and the spec uses a small, consistent
# vocabulary of field names. A new field needs an entry in exactly one of these.
_POINT_FIELDS = frozenset({"position", "center", "start", "end", "to"})
_RANGE_FIELDS = frozenset({"x_range", "y_range"})
_POINTS_FIELDS = frozenset({"points"})
_FLOAT_FIELDS = frozenset(
    {"radius", "width", "height", "font_size", "fill_opacity", "stroke_width", "duration"}
)
_STR_FIELDS = frozenset(
    {"id", "content", "color", "axes", "expression", "target", "into", "narration", "title"}
)


def _is_number(value: Any) -> bool:
    # bool is an int in Python, and `true` where a radius belongs is a defect
    # rather than a radius of 1.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any, where: str, issues: list[Issue]) -> Any:
    if not _is_number(value):
        issues.append(Issue("field-shape", f"expected a number, got {value!r}", where))
        return _INVALID
    return float(value)


def _as_str(value: Any, where: str, issues: list[Issue]) -> Any:
    if not isinstance(value, str):
        issues.append(Issue("field-shape", f"expected a string, got {value!r}", where))
        return _INVALID
    return value


def _as_tuple_of_numbers(value: Any, size: int, where: str, issues: list[Issue]) -> Any:
    if isinstance(value, str) or not isinstance(value, Sequence):
        issues.append(Issue("field-shape", f"expected a list of {size} numbers", where))
        return _INVALID
    if len(value) != size:
        issues.append(
            Issue("field-shape", f"expected {size} numbers, got {len(value)}", where)
        )
        return _INVALID
    if not all(_is_number(item) for item in value):
        issues.append(Issue("field-shape", f"expected numbers, got {value!r}", where))
        return _INVALID
    return tuple(float(item) for item in value)


def _as_points(value: Any, where: str, issues: list[Issue]) -> Any:
    if isinstance(value, str) or not isinstance(value, Sequence):
        issues.append(Issue("field-shape", "expected a list of [x, y] points", where))
        return _INVALID
    out = []
    for index, item in enumerate(value):
        point = _as_tuple_of_numbers(item, 2, f"{where}[{index}]", issues)
        if point is _INVALID:
            return _INVALID
        out.append(point)
    return tuple(out)


def _coerce(name: str, value: Any, where: str, issues: list[Issue]) -> Any:
    if name in _POINT_FIELDS:
        return _as_tuple_of_numbers(value, 2, where, issues)
    if name in _RANGE_FIELDS:
        return _as_tuple_of_numbers(value, 3, where, issues)
    if name in _POINTS_FIELDS:
        return _as_points(value, where, issues)
    if name in _FLOAT_FIELDS:
        return _as_float(value, where, issues)
    if name in _STR_FIELDS:
        return _as_str(value, where, issues)
    raise AssertionError(f"no coercer registered for field {name!r}")


def _required_names(cls: type) -> tuple[str, ...]:
    """The fields with no default, which is the spec's Required column."""
    return tuple(
        f.name
        for f in dc_fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_json(text: str) -> ParseResult:
    try:
        data = json.loads(text)
    except ValueError as exc:
        return ParseResult(None, (Issue("document-shape", f"invalid JSON: {exc}", "$"),))
    return parse_document(data)


def parse_document(data: Any) -> ParseResult:
    """Type *data* as far as it goes, recording every defect on the way."""
    issues: list[Issue] = []
    if not isinstance(data, Mapping):
        issues.append(Issue("document-shape", "the document must be an object", "$"))
        return ParseResult(None, tuple(issues))

    version = data.get("version", VERSION)
    if version != VERSION:
        issues.append(
            Issue("known-version", f"expected version {VERSION!r}, got {version!r}", "$.version")
        )

    title = data.get("title", "")
    if not isinstance(title, str):
        issues.append(Issue("field-shape", "expected a string", "$.title"))
        title = ""

    raw_scenes = data.get("scenes", [])
    if isinstance(raw_scenes, str) or not isinstance(raw_scenes, Sequence):
        issues.append(Issue("document-shape", "scenes must be a list", "$.scenes"))
        raw_scenes = []

    scenes = []
    for index, raw in enumerate(raw_scenes):
        scene = _parse_scene(raw, f"scenes[{index}]", issues)
        if scene is not None:
            scenes.append(scene)

    document = Document(
        version=str(version), title=title, scenes=tuple(scenes)
    )
    return ParseResult(document, tuple(issues))


def _parse_scene(raw: Any, where: str, issues: list[Issue]) -> Scene | None:
    if not isinstance(raw, Mapping):
        issues.append(Issue("document-shape", "a scene must be an object", where))
        return None

    scene_id = raw.get("id")
    if not isinstance(scene_id, str) or not scene_id:
        # Without an id the scene cannot be reported on, repaired, or rendered
        # into a file, so it is the one field that costs the whole scene.
        issues.append(Issue("required-fields", "a scene needs a non-empty id", where))
        return None

    narration = raw.get("narration", "")
    if not isinstance(narration, str):
        issues.append(Issue("field-shape", "expected a string", f"{where}.narration"))
        narration = ""

    objects = tuple(
        _iter_parsed(raw.get("objects", []), "objects", f"{where}.objects", issues, _parse_object)
    )
    steps = tuple(_iter_parsed(raw.get("steps", []), "steps", f"{where}.steps", issues, _parse_step))
    return Scene(id=scene_id, narration=narration, objects=objects, steps=steps)


def _iter_parsed(raw: Any, label: str, where: str, issues: list[Issue], parse) -> Iterable[Any]:
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        issues.append(Issue("document-shape", f"{label} must be a list", where))
        return
    for index, item in enumerate(raw):
        parsed = parse(item, f"{where}[{index}]", issues)
        if parsed is not None:
            yield parsed


def _parse_object(raw: Any, where: str, issues: list[Issue]) -> SceneObject | None:
    if not isinstance(raw, Mapping):
        issues.append(Issue("known-type", "an object must be an object", where))
        return None

    type_name = raw.get("type")
    cls = OBJECT_TYPES.get(type_name) if isinstance(type_name, str) else None
    if cls is None:
        issues.append(
            Issue(
                "known-type",
                f"unknown type {type_name!r}; expected one of "
                f"{', '.join(sorted(OBJECT_TYPES))}",
                where,
            )
        )
        return None

    required = set(_required_names(cls))
    known = {f.name for f in dc_fields(cls)} | {"type"}
    for key in raw:
        if key not in known:
            issues.append(
                Issue(
                    "unknown-field",
                    f"{cls.TYPE} has no field {key!r}",
                    where,
                    Severity.WARNING,
                )
            )

    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    broken = False
    for f in dc_fields(cls):
        if f.name not in raw:
            if f.name in required:
                missing.append(f.name)
                broken = True
            continue
        value = _coerce(f.name, raw[f.name], f"{where}.{f.name}", issues)
        if value is _INVALID:
            broken = True
            continue
        kwargs[f.name] = value

    if missing:
        issues.append(
            Issue("required-fields", f"{cls.TYPE} needs {', '.join(sorted(missing))}", where)
        )
    # The spec puts ">= 3 points" in the Required column, and a two-point
    # polygon is a defect rather than a degenerate shape worth rendering.
    if cls is Polygon and not broken and len(kwargs.get("points", ())) < 3:
        issues.append(Issue("required-fields", "polygon needs at least 3 points", where))
        broken = True

    return None if broken else cls(**kwargs)


#: The fields each action requires beyond ``duration``. This is the whole of
#: the per-action shape: anything in {target, into, to} that an action does not
#: name here is reported and dropped, because carrying it silently hides the
#: common case of the model confusing two actions.
_STEP_FIELDS: dict[Action, tuple[str, ...]] = {
    Action.WRITE: ("target",),
    Action.CREATE: ("target",),
    Action.FADE_IN: ("target",),
    Action.FADE_OUT: ("target",),
    Action.INDICATE: ("target",),
    Action.TRANSFORM: ("target", "into"),
    Action.MOVE: ("target", "to"),
    Action.WAIT: (),
}

_STEP_KEYS = frozenset({"action", "duration", "target", "into", "to"})


def _parse_step(raw: Any, where: str, issues: list[Issue]) -> Step | None:
    if not isinstance(raw, Mapping):
        issues.append(Issue("known-action", "a step must be an object", where))
        return None

    raw_action = raw.get("action")
    try:
        action = Action(raw_action)
    except ValueError:
        issues.append(
            Issue(
                "known-action",
                f"unknown action {raw_action!r}; expected one of "
                f"{', '.join(a.value for a in Action)}",
                where,
            )
        )
        return None

    if "duration" not in raw:
        # Deliberately not the positive-duration rule: that one needs a number
        # in hand to judge, and belongs with the rest of the rules.
        issues.append(Issue("required-fields", f"{action.value} needs duration", where))
        return None
    duration = _as_float(raw["duration"], f"{where}.duration", issues)
    if duration is _INVALID:
        return None

    expected = _STEP_FIELDS[action]
    for key in raw:
        if key not in _STEP_KEYS:
            issues.append(
                Issue("unknown-field", f"a step has no field {key!r}", where, Severity.WARNING)
            )
        elif key in ("target", "into", "to") and key not in expected:
            issues.append(
                Issue(
                    "unknown-field",
                    f"{action.value} takes no {key}; it is ignored",
                    where,
                    Severity.WARNING,
                )
            )

    fields: dict[str, Any] = {}
    missing: list[str] = []
    broken = False
    for name in expected:
        if name not in raw:
            missing.append(name)
            broken = True
            continue
        value = _coerce(name, raw[name], f"{where}.{name}", issues)
        if value is _INVALID:
            broken = True
            continue
        fields[name] = value

    if missing:
        issues.append(
            Issue("required-fields", f"{action.value} needs {', '.join(missing)}", where)
        )
    if broken:
        return None
    return Step(action=action, duration=duration, **fields)
