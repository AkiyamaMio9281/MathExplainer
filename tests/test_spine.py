"""The spine, with the model stubbed out.

The interesting property is graceful degradation, and it can be tested without
spending anything: a stub that returns working code for two scenes and broken
code for the third exercises codegen, validate, render and concat for real,
and asserts that one bad scene costs one scene rather than the lesson.

Only the model is stubbed. manim and ffmpeg run, because a spine test that
mocks the parts that actually fail would be testing the mock.

The hand-written document in docs/examples is loaded here too. It is the one
the milestone is about, and a rules check over it costs nothing -- if the
example ever stops satisfying the validator, that is worth finding here rather
than three minutes into a render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explainer import ir, ir_rules, llm, spine, video

EXAMPLE = Path(__file__).resolve().parents[1] / "docs" / "examples" / "pythagoras.json"

WORKS = """\
from manim import *


class Demo(Scene):
    def construct(self):
        self.play(Write(Text("{label}")), run_time=0.3)
        self.wait(0.2)
"""

BROKEN = """\
from manim import *


class Demo(Scene):
    def construct(self):
        self.play(Write(NoSuchThing()), run_time=0.3)
"""


class StubClient(llm.Client):
    """Returns fixed code per scene id. Never touches the network."""

    def __init__(self, broken: set[str] = frozenset()):
        super().__init__()
        self.broken = set(broken)
        self.prompts: list[str] = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        scene_id = next(
            (i for i in self.broken if f'"id": "{i}"' in prompt), None
        )
        body = BROKEN if scene_id else WORKS.format(label=len(self.prompts))
        return llm.Reply(
            text=f"```python\n{body}\n```",
            usage=llm.Usage(input_tokens=100, output_tokens=50, calls=1),
            seconds=0.0,
            model=llm.MODEL,
            stop_reason="end_turn",
        )


def tiny_document(*scene_ids: str) -> ir.Document:
    return ir.Document(
        title="Stub",
        scenes=tuple(
            ir.Scene(
                id=scene_id,
                narration="word " * 15,
                objects=(ir.Text(id="t", content="x", position=(0.0, 0.0)),),
                steps=(ir.Step(action=ir.Action.WRITE, target="t", duration=6.0),),
            )
            for scene_id in scene_ids
        ),
    )


# ---------------------------------------------------------------------------
# The document the milestone is about
# ---------------------------------------------------------------------------


def test_the_example_document_parses_and_satisfies_every_rule():
    result = ir.parse_json(EXAMPLE.read_text(encoding="utf-8"))

    assert result.ok, result.report()
    issues = ir_rules.check_document(result.document)
    # Errors and warnings both: a hand-written example that trips its own
    # validator is either a bad example or a bad rule, and it matters which.
    assert issues == (), ir_rules.report(issues)


def test_the_example_has_more_than_one_scene():
    # The concat step is only exercised by a document that needs it.
    document = ir.parse_json(EXAMPLE.read_text(encoding="utf-8")).document
    assert len(document.scenes) >= 2


# ---------------------------------------------------------------------------
# Refusing to spend
# ---------------------------------------------------------------------------


def test_an_ir_with_errors_never_reaches_the_model():
    client = StubClient()
    broken = ir.Document(scenes=(ir.Scene(id="s"),))  # no objects, no steps

    run = spine.render_document(broken, Path("."), client)

    assert not run.ok
    assert client.prompts == []
    assert run.usage.calls == 0
    assert "nothing was generated" in run.error


def test_an_unreadable_file_is_reported_rather_than_raised(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    run = spine.render_file(bad, tmp_path / "out")

    assert not run.ok
    assert "unreadable IR" in run.error


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_a_broken_scene_costs_one_scene_not_the_lesson(tmp_path):
    run = spine.render_document(
        tiny_document("one", "two", "three"), tmp_path, StubClient(broken={"two"})
    )

    assert run.ok, run.summary()
    assert [s.scene_id for s in run.rendered] == ["one", "three"]
    assert [s.scene_id for s in run.dropped] == ["two"]
    # The lesson exists and is made of the scenes that worked.
    assert run.video is not None and run.video.is_file()
    assert video.duration(run.video) is not None


@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_a_dropped_scene_says_which_tier_rejected_it(tmp_path):
    run = spine.render_document(tiny_document("only"), tmp_path, StubClient(broken={"only"}))

    (dropped,) = run.dropped
    # NoSuchThing is referenced only inside construct(), so it survives import
    # and fails at the dry run -- the split that justifies two tiers.
    assert "DRYRUN" in dropped.error
    assert "NoSuchThing" in dropped.error
    assert not run.ok
    assert run.error == "no scene rendered"


@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_each_scene_renders_in_its_own_directory(tmp_path):
    # Independent workdirs are what make the failures independent, and what
    # will let these run in parallel later.
    run = spine.render_document(tiny_document("alpha", "beta"), tmp_path, StubClient())

    assert (tmp_path / "alpha" / "scene.py").is_file()
    assert (tmp_path / "beta" / "scene.py").is_file()
    assert run.ok


@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_the_summary_names_what_was_dropped_and_what_it_cost(tmp_path):
    run = spine.render_document(
        tiny_document("kept", "lost"), tmp_path, StubClient(broken={"lost"})
    )
    summary = run.summary()

    assert "1/2 scenes rendered" in summary
    assert "DROP lost" in summary
    assert "$" in summary
    assert run.usage.calls == 2


@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_the_cli_writes_a_run_record(tmp_path):
    document = tmp_path / "doc.json"
    document.write_text(
        json.dumps(ir.Document(scenes=tiny_document("solo").scenes).to_dict()),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    assert spine.main([str(document), str(out)]) == 0

    record = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert record["ok"] is True
    assert record["scenes"][0]["status"] == "rendered"
    assert record["dollars"] >= 0


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.api
@pytest.mark.slow
@pytest.mark.skipif(not video.available(), reason="ffmpeg is required")
def test_the_hand_written_lesson_renders_end_to_end(tmp_path):
    run = spine.render_file(EXAMPLE, tmp_path)

    print("\n" + run.summary())
    assert run.ok, run.summary()
    assert len(run.rendered) == 3
    assert run.video.is_file()
    # Three scenes of 6, 9 and 7.5 seconds, joined.
    assert video.duration(run.video) == pytest.approx(22.5, abs=3.0)
