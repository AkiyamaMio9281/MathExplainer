"""Parsing the Scene IR: what types cleanly, what is reported, what is dropped.

Two properties are worth more than the individual cases and are asserted
directly. A well-formed document survives a round trip through ``to_dict``
unchanged, so the typed form and the JSON form say the same thing. And a
document with several independent defects produces several issues in one pass
-- the repair loop gets one round carrying everything, not one round per
missing field.

The boundary with ir_rules.py is asserted too, from this side. A duration of
zero and a duplicated id both parse without complaint here, because judging
them needs a typed document and they belong with the rules. Those tests exist
so that moving a check across the boundary breaks something visible.
"""

from __future__ import annotations

import pytest

from explainer import ir

FULL = {
    "version": "1",
    "title": "The Pythagorean Theorem",
    "scenes": [
        {
            "id": "intro",
            "narration": "Every right triangle hides one simple relationship.",
            "objects": [
                {
                    "id": "title",
                    "type": "text",
                    "content": "Pythagoras",
                    "position": [0, 3],
                    "font_size": 48,
                    "color": "BLUE",
                },
                {
                    "id": "eq",
                    "type": "mathtex",
                    "content": "a^2 + b^2 = c^2",
                    "position": [3.2, 0],
                },
                {
                    "id": "tri",
                    "type": "polygon",
                    "points": [[-3, -1], [0, -1], [0, 1.5]],
                    "color": "BLUE",
                    "fill_opacity": 0.2,
                },
                {"id": "circ", "type": "circle", "center": [-2, 2], "radius": 0.8},
                {
                    "id": "rect",
                    "type": "rectangle",
                    "center": [2, -2],
                    "width": 1.5,
                    "height": 1.0,
                },
                {
                    "id": "seg",
                    "type": "line",
                    "start": [-1, -3],
                    "end": [1, -3],
                    "stroke_width": 6,
                },
                {"id": "arr", "type": "arrow", "start": [-4, 0], "end": [-2, 0]},
                {
                    "id": "ax",
                    "type": "axes",
                    "x_range": [-3, 3, 1],
                    "y_range": [-2, 2, 1],
                    "position": [0, -1],
                },
                {
                    "id": "curve",
                    "type": "plot",
                    "axes": "ax",
                    "expression": "np.sin(x)",
                    "x_range": [-3, 3, 0.1],
                },
            ],
            "steps": [
                {"action": "write", "target": "title", "duration": 1.5},
                {"action": "create", "target": "tri", "duration": 2.0},
                {"action": "fade_in", "target": "circ", "duration": 1.0},
                {"action": "write", "target": "eq", "duration": 1.5},
                {"action": "indicate", "target": "eq", "duration": 1.0},
                {"action": "move", "target": "rect", "to": [2, 2], "duration": 1.0},
                {"action": "transform", "target": "seg", "into": "arr", "duration": 1.5},
                {"action": "fade_out", "target": "circ", "duration": 0.5},
                {"action": "wait", "duration": 1.0},
            ],
        }
    ],
}


def parsed(data) -> ir.Document:
    result = ir.parse_document(data)
    assert result.ok, result.report()
    assert result.document is not None
    return result.document


# ---------------------------------------------------------------------------
# The whole vocabulary, and the round trip
# ---------------------------------------------------------------------------


def test_every_object_type_and_action_parses():
    document = parsed(FULL)
    scene = document.scenes[0]

    assert {type(o).TYPE for o in scene.objects} == set(ir.OBJECT_TYPES)
    assert {s.action for s in scene.steps} == set(ir.Action)


def test_a_document_round_trips_through_to_dict():
    once = parsed(FULL)
    twice = parsed(once.to_dict())
    assert twice == once


def test_json_round_trips_too():
    once = parsed(FULL)
    result = ir.parse_json(once.to_json())
    assert result.ok, result.report()
    assert result.document == once


def test_values_are_typed_not_left_as_json():
    scene = parsed(FULL).scenes[0]
    by_id = {o.id: o for o in scene.objects}

    assert by_id["title"].position == (0.0, 3.0)
    assert by_id["tri"].points == ((-3.0, -1.0), (0.0, -1.0), (0.0, 1.5))
    assert by_id["ax"].x_range == (-3.0, 3.0, 1.0)
    assert scene.steps[0].action is ir.Action.WRITE


def test_defaults_match_the_specification():
    document = parsed(
        {
            "version": "1",
            "scenes": [
                {
                    "id": "s",
                    "objects": [
                        {"id": "t", "type": "text", "content": "x", "position": [0, 0]},
                        {"id": "m", "type": "mathtex", "content": "x", "position": [0, 0]},
                        {"id": "c", "type": "circle", "center": [0, 0], "radius": 1},
                        {"id": "l", "type": "line", "start": [0, 0], "end": [1, 1]},
                        {
                            "id": "a",
                            "type": "axes",
                            "x_range": [0, 1, 1],
                            "y_range": [0, 1, 1],
                        },
                    ],
                    "steps": [{"action": "wait", "duration": 1}],
                }
            ],
        }
    )
    by_id = {o.id: o for o in document.scenes[0].objects}

    assert by_id["t"].font_size == 36.0
    assert by_id["m"].font_size == 48.0
    assert by_id["t"].color == "WHITE"
    assert by_id["c"].fill_opacity == 0.0
    assert by_id["a"].position == (0.0, 0.0)
    # None rather than 4: the renderer's default, not a Manim constant baked
    # into a renderer-agnostic document.
    assert by_id["l"].stroke_width is None


# ---------------------------------------------------------------------------
# Defects are collected, not raised on
# ---------------------------------------------------------------------------


def one_scene(objects=(), steps=()):
    return {
        "version": "1",
        "scenes": [{"id": "s", "objects": list(objects), "steps": list(steps)}],
    }


def test_independent_defects_are_all_reported_in_one_pass():
    result = ir.parse_document(
        one_scene(
            objects=[
                {"id": "good", "type": "text", "content": "x", "position": [0, 0]},
                {"id": "a", "type": "sphere", "position": [0, 0]},
                {"id": "b", "type": "text"},
                {"id": "c", "type": "circle", "center": [0, "x"], "radius": 1},
            ]
        )
    )

    assert not result.ok
    rules = sorted(i.rule for i in result.errors)
    assert rules == ["field-shape", "known-type", "required-fields"]
    # The good object survives; only what could not be typed is dropped.
    assert [o.id for o in result.document.scenes[0].objects] == ["good"]


def test_a_missing_required_field_names_every_field_it_missed():
    result = ir.parse_document(one_scene(objects=[{"id": "b", "type": "text"}]))

    (issue,) = result.errors
    assert issue.rule == "required-fields"
    assert "content" in issue.message and "position" in issue.message
    assert issue.where == "scenes[0].objects[0]"


def test_an_unknown_type_lists_the_ones_that_exist():
    result = ir.parse_document(one_scene(objects=[{"id": "a", "type": "sphere"}]))

    (issue,) = result.errors
    assert issue.rule == "known-type"
    assert "mathtex" in issue.message and "polygon" in issue.message


@pytest.mark.parametrize(
    "position", [[0], [0, 1, 2], "0,1", {"x": 0}, [0, None]], ids=str
)
def test_a_point_of_the_wrong_shape_is_reported(position):
    result = ir.parse_document(
        one_scene(objects=[{"id": "t", "type": "text", "content": "x", "position": position}])
    )
    assert not result.ok
    assert all(i.rule == "field-shape" for i in result.errors)


def test_a_boolean_is_not_a_number():
    # isinstance(True, int) is True, so this needs saying explicitly.
    result = ir.parse_document(
        one_scene(objects=[{"id": "c", "type": "circle", "center": [0, 0], "radius": True}])
    )
    assert not result.ok
    assert result.errors[0].rule == "field-shape"


def test_a_polygon_needs_three_points():
    result = ir.parse_document(
        one_scene(objects=[{"id": "p", "type": "polygon", "points": [[0, 0], [1, 1]]}])
    )

    (issue,) = result.errors
    assert issue.rule == "required-fields"
    assert "at least 3" in issue.message


def test_an_unknown_action_lists_the_ones_that_exist():
    result = ir.parse_document(one_scene(steps=[{"action": "zoom", "duration": 1}]))

    (issue,) = result.errors
    assert issue.rule == "known-action"
    assert "indicate" in issue.message


@pytest.mark.parametrize(
    "step,missing",
    [
        ({"action": "transform", "target": "a", "duration": 1}, "into"),
        ({"action": "move", "target": "a", "duration": 1}, "to"),
        ({"action": "write", "duration": 1}, "target"),
        ({"action": "write", "target": "a"}, "duration"),
    ],
)
def test_a_step_missing_its_action_specific_field_is_reported(step, missing):
    result = ir.parse_document(one_scene(steps=[step]))

    (issue,) = result.errors
    assert issue.rule == "required-fields"
    assert missing in issue.message
    assert result.document.scenes[0].steps == ()


def test_a_scene_without_an_id_costs_the_scene_but_not_the_document():
    result = ir.parse_document(
        {"version": "1", "scenes": [{"narration": "x"}, {"id": "kept"}]}
    )

    assert not result.ok
    assert [s.id for s in result.document.scenes] == ["kept"]


def test_a_version_mismatch_is_reported_but_keeps_the_document():
    result = ir.parse_document({"version": "2", "scenes": []})

    assert not result.ok
    assert result.errors[0].rule == "known-version"
    assert result.document is not None


@pytest.mark.parametrize("data", ["a string", 3, None, [1, 2]], ids=str)
def test_a_document_that_is_not_an_object_yields_nothing(data):
    result = ir.parse_document(data)

    assert result.document is None
    assert result.errors[0].rule == "document-shape"


def test_invalid_json_is_an_issue_rather_than_an_exception():
    result = ir.parse_json("{not json")

    assert result.document is None
    assert result.errors[0].rule == "document-shape"


def test_report_locates_every_issue():
    result = ir.parse_document(one_scene(objects=[{"id": "b", "type": "text"}]))
    assert "scenes[0].objects[0]" in result.report()
    assert "required-fields" in result.report()


# ---------------------------------------------------------------------------
# Warnings: reported, but the document is still usable
# ---------------------------------------------------------------------------


def test_a_field_the_action_does_not_use_is_a_warning():
    result = ir.parse_document(
        one_scene(steps=[{"action": "wait", "duration": 1, "target": "a"}])
    )

    assert result.ok
    (warning,) = result.warnings
    assert warning.rule == "unknown-field"
    assert result.document.scenes[0].steps[0].target is None


def test_an_unknown_object_field_is_a_warning():
    result = ir.parse_document(
        one_scene(
            objects=[
                {
                    "id": "t",
                    "type": "text",
                    "content": "x",
                    "position": [0, 0],
                    "rotation": 45,
                }
            ]
        )
    )

    assert result.ok
    assert result.warnings[0].rule == "unknown-field"
    assert result.document.scenes[0].objects[0].id == "t"


# ---------------------------------------------------------------------------
# The boundary with ir_rules.py, asserted from this side
# ---------------------------------------------------------------------------


def test_a_non_positive_duration_parses_because_it_is_a_rule():
    result = ir.parse_document(one_scene(steps=[{"action": "wait", "duration": 0}]))

    assert result.ok, result.report()
    assert result.document.scenes[0].steps[0].duration == 0.0


def test_duplicate_ids_parse_because_uniqueness_is_a_rule():
    result = ir.parse_document(
        one_scene(
            objects=[
                {"id": "same", "type": "text", "content": "a", "position": [0, 0]},
                {"id": "same", "type": "text", "content": "b", "position": [0, 1]},
            ]
        )
    )

    assert result.ok, result.report()
    assert [o.id for o in result.document.scenes[0].objects] == ["same", "same"]


def test_a_dangling_reference_parses_because_integrity_is_a_rule():
    result = ir.parse_document(
        one_scene(steps=[{"action": "write", "target": "nothing", "duration": 1}])
    )

    assert result.ok, result.report()
    assert result.document.scenes[0].steps[0].target == "nothing"
