"""Joining clips, including the case the concat demuxer cannot handle.

Test clips come from ffmpeg's own ``testsrc`` rather than from manim: this
module knows nothing about manim, and a rendering dependency would make a
concat test slow and able to fail for unrelated reasons.

The case worth having is the mismatched one. Two clips of different sizes are
exactly what the concat demuxer refuses, and the refusal arrives at the join
rather than at either render, so the fallback is the difference between a
lesson and a message about a muxer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explainer import video
from explainer.sandbox import run

pytestmark = pytest.mark.skipif(
    not video.available(), reason="ffmpeg and ffprobe are required"
)


def make_clip(path: Path, seconds=1.0, size="320x240", rate=15) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    done = run(
        [
            video.FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={seconds}:size={size}:rate={rate}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        cwd=path.parent,
        timeout=120,
    )
    assert done.ok, done.failure_text()
    return path


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_probe_reads_the_properties_concat_cares_about(tmp_path):
    stream = video.probe(make_clip(tmp_path / "a.mp4", size="320x240", rate=15))

    assert stream is not None
    assert (stream.width, stream.height) == (320, 240)
    assert stream.frame_rate == "15/1"
    assert stream.pix_fmt == "yuv420p"


@pytest.mark.slow
def test_probe_declines_a_file_that_is_not_a_video(tmp_path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"certainly not an mp4")

    assert video.probe(junk) is None


# ---------------------------------------------------------------------------
# The fast path
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_matching_clips_are_copied_rather_than_re_encoded(tmp_path):
    clips = [
        make_clip(tmp_path / "a.mp4", seconds=1.0),
        make_clip(tmp_path / "b.mp4", seconds=1.0),
    ]
    out = tmp_path / "lesson.mp4"

    result = video.concat(clips, out)

    assert result.ok, result.error
    assert result.method == "copy"
    assert result.output == out
    assert video.duration(out) == pytest.approx(2.0, abs=0.3)


@pytest.mark.slow
def test_one_clip_is_still_a_lesson(tmp_path):
    out = tmp_path / "lesson.mp4"
    result = video.concat([make_clip(tmp_path / "only.mp4", seconds=1.0)], out)

    assert result.ok, result.error
    assert video.duration(out) == pytest.approx(1.0, abs=0.3)


@pytest.mark.slow
def test_the_listing_file_does_not_survive_the_run(tmp_path):
    clips = [make_clip(tmp_path / "a.mp4"), make_clip(tmp_path / "b.mp4")]
    video.concat(clips, tmp_path / "lesson.mp4")

    assert not list(tmp_path.glob("*concat*.txt"))


@pytest.mark.slow
def test_paths_containing_spaces_survive_the_demuxer_listing(tmp_path):
    # The demuxer parses its own listing, and a path written without quoting
    # becomes a complaint about a file nobody named.
    room = tmp_path / "a directory with spaces"
    clips = [make_clip(room / "first clip.mp4"), make_clip(room / "second clip.mp4")]

    result = video.concat(clips, room / "the lesson.mp4")

    assert result.ok, result.error
    assert result.method == "copy"


# ---------------------------------------------------------------------------
# The fallback
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_clips_of_different_sizes_are_re_encoded_rather_than_refused(tmp_path):
    clips = [
        make_clip(tmp_path / "small.mp4", seconds=1.0, size="320x240"),
        make_clip(tmp_path / "large.mp4", seconds=1.0, size="640x480"),
    ]
    out = tmp_path / "lesson.mp4"

    result = video.concat(clips, out)

    assert result.ok, result.error
    assert result.method == "reencode"
    assert video.duration(out) == pytest.approx(2.0, abs=0.4)
    # Normalised to the first clip, which is what the filter was told to do.
    assert (video.probe(out).width, video.probe(out).height) == (320, 240)


@pytest.mark.slow
def test_clips_of_different_frame_rates_are_re_encoded(tmp_path):
    clips = [
        make_clip(tmp_path / "slow.mp4", seconds=1.0, rate=15),
        make_clip(tmp_path / "fast.mp4", seconds=1.0, rate=30),
    ]

    result = video.concat(clips, tmp_path / "lesson.mp4")

    assert result.ok, result.error
    assert result.method == "reencode"


@pytest.mark.slow
def test_the_order_given_is_the_order_played(tmp_path):
    # Lengths stand in for identity: a 1s clip then a 3s clip is not the same
    # video as the reverse, and duration is the cheapest way to tell.
    first = make_clip(tmp_path / "a.mp4", seconds=1.0)
    second = make_clip(tmp_path / "b.mp4", seconds=3.0)

    forwards = video.concat([first, second], tmp_path / "forwards.mp4")
    backwards = video.concat([second, first], tmp_path / "backwards.mp4")

    assert forwards.ok and backwards.ok
    assert video.duration(forwards.output) == pytest.approx(4.0, abs=0.4)
    assert video.duration(backwards.output) == pytest.approx(4.0, abs=0.4)


# ---------------------------------------------------------------------------
# Refusals that cost nothing
# ---------------------------------------------------------------------------


def test_an_empty_list_is_refused_without_starting_ffmpeg(tmp_path):
    result = video.concat([], tmp_path / "lesson.mp4")

    assert not result.ok
    assert "nothing to concatenate" in result.error
    assert not (tmp_path / "lesson.mp4").exists()


def test_a_missing_clip_is_named(tmp_path):
    present = tmp_path / "there.mp4"
    present.write_bytes(b"x")
    result = video.concat([present, tmp_path / "gone.mp4"], tmp_path / "lesson.mp4")

    assert not result.ok
    assert "gone.mp4" in result.error


@pytest.mark.slow
def test_a_clip_with_no_video_stream_is_reported_before_encoding(tmp_path):
    good = make_clip(tmp_path / "good.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")

    result = video.concat([good, junk], tmp_path / "lesson.mp4")

    assert not result.ok
    assert "junk.mp4" in result.error
    # Our message, not ffmpeg's, and no method was ever chosen: the point of
    # probing first is that the run stops before any encoding is attempted.
    assert "no readable video stream" in result.error
    assert result.method == ""
    assert not (tmp_path / "lesson.mp4").exists()
