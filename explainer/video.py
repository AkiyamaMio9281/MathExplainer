"""Join per-scene clips into one lesson video.

One Manim ``Scene`` per IR scene, rendered independently, is what makes a
failure containable -- scene four failing does not cost scenes one to three --
and this is the step that pays for it. ffmpeg is already a dependency, because
manim encodes through it.

**The trap here is not the happy path.** ffmpeg's concat demuxer stitches
streams without re-encoding, which is fast and lossless, but it requires every
input to agree on resolution, frame rate, codec and pixel format. When they do
not agree it fails at the join, long after the renders that produced them
looked fine -- and the error names a muxer rather than the scene that differs.
So the inputs are probed first, and the answer decides the method:

* every stream identical -> the concat demuxer with ``-c copy``. No re-encode,
  a fraction of a second.
* anything differing -> the concat *filter*, scaling and resampling each input
  to match the first. Slower, lossy, and always correct.

Falling back rather than failing is the same choice made everywhere else here:
a lesson that renders is worth more than a clean error about a frame rate.

There is no audio track. Narration is text in the IR, so every stream is
video-only, and the concat filter is told so explicitly -- ``a=0`` rather than
letting ffmpeg look for audio that is not there.

Subprocesses go through ``sandbox.run``, so ffmpeg gets the same allowlisted
environment as generated Python. It has no business reading an API key either.

**Every path handed to a child is absolute.** The children here run with a
``cwd`` that is not this process's, so a relative path is resolved a second
time against it -- ``runs/x/scene/runs/x/scene/clip.mp4``, and an error about
a file nobody named. It cost a full pipeline run to find the first instance of
this and a second one to find the next, so the rule is stated rather than
remembered.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .sandbox import Completed, run

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


@dataclass(frozen=True)
class Stream:
    """The properties the concat demuxer insists on agreeing about."""

    width: int
    height: int
    frame_rate: str  # as ffprobe reports it, e.g. "30/1"
    codec: str
    pix_fmt: str


@dataclass(frozen=True)
class Concatenated:
    ok: bool
    output: Path | None = None
    method: str = ""  # "copy" or "reencode"
    seconds: float = 0.0
    error: str = ""


def available() -> bool:
    """Whether ffmpeg and ffprobe are both on PATH."""
    return shutil.which(FFMPEG) is not None and shutil.which(FFPROBE) is not None


def probe(clip: Path, timeout: float = 60.0) -> Stream | None:
    """The video stream of *clip*, or None when it has none or cannot be read."""
    done = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,codec_name,pix_fmt",
            "-of",
            "json",
            str(clip.resolve()),
        ],
        cwd=clip.parent,
        timeout=timeout,
    )
    if not done.ok:
        return None
    try:
        streams = json.loads(done.stdout).get("streams", [])
    except ValueError:
        return None
    if not streams:
        return None
    entry = streams[0]
    try:
        return Stream(
            width=int(entry["width"]),
            height=int(entry["height"]),
            frame_rate=str(entry["r_frame_rate"]),
            codec=str(entry["codec_name"]),
            pix_fmt=str(entry["pix_fmt"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def duration(clip: Path, timeout: float = 60.0) -> float | None:
    """Length of *clip* in seconds, or None when it cannot be read."""
    done = run(
        [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip.resolve()),
        ],
        cwd=clip.parent,
        timeout=timeout,
    )
    if not done.ok:
        return None
    try:
        return float(done.stdout.strip())
    except ValueError:
        return None


def concat(
    clips: Sequence[Path], output: Path, timeout: float = 900.0
) -> Concatenated:
    """Join *clips*, in order, into *output*."""
    if not clips:
        return Concatenated(False, error="nothing to concatenate")
    missing = [str(c) for c in clips if not c.is_file()]
    if missing:
        return Concatenated(False, error=f"missing clip(s): {', '.join(missing)}")
    if not available():
        return Concatenated(
            False, error=f"{FFMPEG} and {FFPROBE} must both be on PATH"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    clips = [clip.resolve() for clip in clips]
    streams = [probe(clip) for clip in clips]
    unreadable = [str(c) for c, s in zip(clips, streams) if s is None]
    if unreadable:
        return Concatenated(
            False, error=f"no readable video stream in: {', '.join(unreadable)}"
        )

    if len(set(streams)) == 1:
        return _copy(clips, output, timeout)
    return _reencode(clips, output, streams[0], timeout)


def _copy(clips: Sequence[Path], output: Path, timeout: float) -> Concatenated:
    listing = output.parent / f".{output.stem}-concat.txt"
    listing.write_text(_listing(clips), encoding="utf-8")
    done = run(
        [
            FFMPEG,
            "-y",
            "-f",
            "concat",
            # Absolute paths are "unsafe" to the demuxer unless this is set,
            # and every path here is absolute.
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(output),
        ],
        cwd=output.parent,
        timeout=timeout,
    )
    listing.unlink(missing_ok=True)
    return _result(done, output, "copy")


def _listing(clips: Sequence[Path]) -> str:
    # The demuxer's own quoting: single quotes around each path, and a literal
    # quote written as '\''. Forward slashes because a backslash is an escape
    # character here, which is how a Windows path silently becomes a parse
    # error about a file nobody named.
    lines = []
    for clip in clips:
        path = clip.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{path}'")
    return "\n".join(lines) + "\n"


def _reencode(
    clips: Sequence[Path], output: Path, target: Stream, timeout: float
) -> Concatenated:
    args = [FFMPEG, "-y"]
    for clip in clips:
        args += ["-i", str(clip.resolve())]

    # Each input is scaled and resampled to the first clip's geometry before
    # the concat filter sees it; setsar keeps a rescaled clip from carrying a
    # pixel aspect ratio that would squash it.
    parts = []
    for index in range(len(clips)):
        parts.append(
            f"[{index}:v]scale={target.width}:{target.height},"
            f"setsar=1,fps={target.frame_rate}[v{index}]"
        )
    joined = "".join(f"[v{i}]" for i in range(len(clips)))
    parts.append(f"{joined}concat=n={len(clips)}:v=1:a=0[out]")

    args += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[out]",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    done = run(args, cwd=output.parent, timeout=timeout)
    return _result(done, output, "reencode")


def _result(done: Completed, output: Path, method: str) -> Concatenated:
    if not done.ok:
        return Concatenated(
            False, method=method, seconds=done.seconds, error=done.failure_text()
        )
    if not output.is_file() or output.stat().st_size == 0:
        return Concatenated(
            False,
            method=method,
            seconds=done.seconds,
            error="ffmpeg exited 0 but produced no video",
        )
    return Concatenated(True, output=output, method=method, seconds=done.seconds)
