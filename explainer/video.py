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

**The clips themselves are silent** -- narration is text in the IR, so every
rendered stream is video-only, and the concat filter is told so explicitly
(``a=0`` rather than letting ffmpeg look for audio that is not there). Sound
arrives afterwards: ``speech`` synthesises each scene's narration, ``audio_track``
lays those files onto one timeline, and ``mux`` puts it on the joined lesson.

That ordering is why the audio is *placed* rather than concatenated. Each
scene's speech is delayed to its own start offset, so a scene whose voice runs
slightly long cannot shift every later scene out of step; the error stays local
instead of accumulating. The subtitles come from the same source -- the
synthesiser reports when each word is said, so ``spoken_cues`` puts a line up
on the syllable, where ``cues_for`` could only share the scene out by word
count.

The words still reach a viewer who has no sound, because the subtitles are
burned in either way. A silent lesson remains a working lesson: if speech
fails, nothing here does.

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
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

#: Words in one subtitle. Long enough to read as a phrase, short enough that a
#: line does not sit on screen after the animation has moved on.
WORDS_PER_CUE = 12

#: Characters before a cue wraps onto a second line.
WRAP_AT = 42

#: Sentence ends, for splitting narration before word count gets a say. A
#: cue running from the middle of one sentence into the middle of the next
#: is a caption only in the technical sense.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

NEWLINE = "\n"


@dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def _timestamp(seconds: float) -> str:
    seconds = max(seconds, 0.0)
    whole = int(seconds)
    milliseconds = int(round((seconds - whole) * 1000))
    if milliseconds == 1000:  # rounding up should not produce ",1000"
        whole, milliseconds = whole + 1, 0
    return (
        f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:"
        f"{whole % 60:02d},{milliseconds:03d}"
    )


def _wrap(text: str, width: int = WRAP_AT) -> str:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) <= 2:
        return NEWLINE.join(lines)
    # Never more than two lines: a third covers the animation it describes.
    return NEWLINE.join([" ".join(lines[:-1]), lines[-1]])


def cues_for(
    narration: str,
    start: float,
    duration: float,
    words_per_cue: int = WORDS_PER_CUE,
) -> list[Cue]:
    """Split one scene's narration across the time that scene actually runs.

    Time is shared out by word count, so a long phrase stays up longer than a
    short one. *duration* is the rendered clip's length rather than the IR's
    requested one: what was asked for and what manim produced differ slightly
    every scene, and over five scenes that drift is enough to put a line under
    the wrong animation.
    """
    words = narration.split()
    if not words or duration <= 0:
        return []

    chunks = _split(narration, words_per_cue)
    cues: list[Cue] = []
    spent = 0.0
    for index, chunk in enumerate(chunks):
        # The last cue takes whatever is left, so rounding cannot leave a gap
        # at the end of the scene or run past it.
        end = duration if index == len(chunks) - 1 else spent + duration * len(chunk) / len(words)
        cues.append(Cue(start + spent, start + end, _wrap(" ".join(chunk))))
        spent = end
    return cues


def _split(narration: str, words_per_cue: int) -> list[list[str]]:
    """Sentences first, word count second.

    Splitting purely by count produced lines like "rows of three. Twelve
    is flexible. Now take seven dots and try" -- three fragments of three
    different sentences, read off a real frame of a real run. A sentence
    shorter than the limit becomes one cue whatever its length; a longer
    one is broken up, and only then by count.
    """
    chunks: list[list[str]] = []
    for sentence in _SENTENCE_END.split(narration.strip()):
        words = sentence.split()
        if not words:
            continue
        for index in range(0, len(words), words_per_cue):
            chunks.append(words[index : index + words_per_cue])
    return chunks


def cues_from(scenes: Sequence[tuple[str, float]]) -> list[Cue]:
    """Lay several scenes' narration end to end on one timeline."""
    cues: list[Cue] = []
    at = 0.0
    for narration, duration in scenes:
        cues.extend(cues_for(narration, at, duration))
        at += max(duration, 0.0)
    return cues


def to_srt(cues: Sequence[Cue]) -> str:
    blocks = [
        f"{number}{NEWLINE}{_timestamp(cue.start)} --> {_timestamp(cue.end)}"
        f"{NEWLINE}{cue.text}{NEWLINE}"
        for number, cue in enumerate(cues, start=1)
    ]
    return NEWLINE.join(blocks)


def subtitle(
    video_path: Path, cues: Sequence[Cue], timeout: float = 900.0
) -> Concatenated:
    """Burn *cues* into *video_path*, in place.

    In place because the lesson should have one name whether it carries
    subtitles or not, and ffmpeg cannot read and write the same file -- so it
    writes a neighbour and replaces.

    The filter references the subtitle file by bare name, with the child's cwd
    set to the directory holding it. A filtergraph parses its own argument, so
    a Windows path inside one needs both its backslashes and the colon after
    the drive letter escaped; a chdir sidesteps the whole question.
    """
    if not cues:
        return Concatenated(False, method="subtitles", error="no narration to burn")
    if not video_path.is_file():
        return Concatenated(False, method="subtitles", error=f"missing: {video_path}")
    if not available():
        return Concatenated(False, method="subtitles", error="ffmpeg is not on PATH")

    video_path = video_path.resolve()
    srt = video_path.with_suffix(".srt")
    burned = video_path.with_name(f"{video_path.stem}-subtitled.mp4")
    srt.write_text(to_srt(cues), encoding="utf-8")

    done = run(
        [
            FFMPEG,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"subtitles={srt.name}:force_style='Fontsize=16,MarginV=24'",
            "-pix_fmt",
            "yuv420p",
            str(burned),
        ],
        cwd=srt.parent,
        timeout=timeout,
    )
    srt.unlink(missing_ok=True)

    if not done.ok or not burned.is_file() or burned.stat().st_size == 0:
        burned.unlink(missing_ok=True)
        return Concatenated(
            False, method="subtitles", seconds=done.seconds, error=done.failure_text()
        )

    burned.replace(video_path)
    return Concatenated(
        True, output=video_path, method="subtitles", seconds=done.seconds
    )


def subtitle_lesson(
    video_path: Path, scenes: Sequence[tuple[str, Path]], timeout: float = 900.0
) -> Concatenated:
    """Burn the narration of *scenes* onto the joined lesson.

    Each scene is timed by its own clip's measured length rather than by what
    the IR asked for, so the lines stay under the animation they describe.
    """
    timed = [(narration, duration(clip) or 0.0) for narration, clip in scenes]
    return subtitle(video_path, cues_from(timed), timeout=timeout)


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------

#: One track, one channel, at a rate everything downstream is happy with.
AUDIO_RATE = 44_100


def audio_track(
    placements: Sequence[tuple[Path, float]],
    output: Path,
    total: float,
    timeout: float = 900.0,
) -> Concatenated:
    """Lay each scene's narration at its own start time on one track.

    *placements* is ``(audio file, seconds into the lesson)``. Each clip is
    delayed to its slot rather than concatenated end to end, so alignment is
    anchored per scene and a scene whose speech runs a little long cannot push
    every later scene out of step -- it bleeds slightly into the next one
    instead, which is what a person reading aloud does anyway.
    """
    if not placements:
        return Concatenated(False, method="audio", error="nothing to say")
    if not available():
        return Concatenated(False, method="audio", error="ffmpeg is not on PATH")

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    args = [FFMPEG, "-y"]
    for path, _ in placements:
        args += ["-i", str(path.resolve())]

    parts = []
    for index, (_, start) in enumerate(placements):
        milliseconds = max(int(round(start * 1000)), 0)
        parts.append(f"[{index}:a]adelay={milliseconds}|{milliseconds}[d{index}]")
    mixed = "".join(f"[d{i}]" for i in range(len(placements)))
    # normalize=0 keeps one voice at one voice's volume rather than dividing
    # it by the number of inputs.
    parts.append(f"{mixed}amix=inputs={len(placements)}:normalize=0[out]")

    args += [
        "-filter_complex",
        ";".join(parts),
        "-map",
        "[out]",
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "1",
        "-t",
        f"{max(total, 0.1):.3f}",
        str(output),
    ]
    done = run(args, cwd=output.parent, timeout=timeout)
    return _result(done, output, "audio")


def mux(video_path: Path, audio_path: Path, timeout: float = 900.0) -> Concatenated:
    """Put *audio_path* onto *video_path*, in place.

    The video is always copied rather than re-encoded -- it has just been
    through the subtitle burn, and a second pass would cost quality for
    nothing. The audio is copied too when the container will take it as it is,
    which is the normal case here because `soundtrack` writes AAC in an M4A;
    a stream mp4 will not carry falls back to encoding, the same copy-first
    shape the concat path uses and for the same reason.

    That fallback matters because it avoids a *second* lossy encode. Speech at
    a low bitrate re-encoded once more is audibly worse, and the first version
    of this did it on every run.
    """
    if not video_path.is_file():
        return Concatenated(False, method="mux", error=f"missing: {video_path}")
    if not audio_path.is_file():
        return Concatenated(False, method="mux", error=f"missing: {audio_path}")

    video_path = video_path.resolve()
    combined = video_path.with_name(f"{video_path.stem}-sound.mp4")

    def attempt(codec: str) -> Completed:
        return run(
            [
                FFMPEG, "-y",
                "-i", str(video_path),
                "-i", str(audio_path.resolve()),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", codec,
                *(() if codec == "copy" else ("-b:a", "128k")),
                str(combined),
            ],
            cwd=video_path.parent,
            timeout=timeout,
        )

    method = "mux"
    done = attempt("copy")
    if not done.ok or not combined.is_file() or combined.stat().st_size == 0:
        combined.unlink(missing_ok=True)
        method = "mux-reencode"
        done = attempt("aac")

    if not done.ok or not combined.is_file() or combined.stat().st_size == 0:
        combined.unlink(missing_ok=True)
        return Concatenated(
            False, method=method, seconds=done.seconds, error=done.failure_text()
        )
    combined.replace(video_path)
    return Concatenated(True, output=video_path, method=method, seconds=done.seconds)


def soundtrack(
    video_path: Path,
    placements: Sequence[tuple[Path, float]],
    total: float,
    timeout: float = 900.0,
) -> Concatenated:
    """Build the narration track and put it on *video_path*, in place.

    Two ffmpeg passes rather than one: the track is written to a file first so
    that a failure to *build* it is distinguishable from a failure to attach
    it, and so the track survives the run for anyone wanting to listen to it
    on its own.
    """
    track = video_path.resolve().with_name(f"{video_path.stem}-narration.m4a")
    built = audio_track(placements, track, total, timeout=timeout)
    if not built.ok:
        return built
    return mux(video_path, track, timeout=timeout)


def spoken_cues(
    scenes: Sequence[tuple[Sequence[Any], float]],
    words_per_cue: int = WORDS_PER_CUE,
) -> list[Cue]:
    """Subtitles placed on the syllable, from the synthesiser's own timings.

    *scenes* is ``(words, seconds into the lesson)`` where each word carries
    ``start``, ``end`` and ``text``. Grouping is the same as for estimated
    cues -- sentences first, then a word limit -- but each cue now begins and
    ends when its words are actually said, rather than at a share of the scene
    proportional to its length. A long word and a short one do not take the
    same time to say, which is all the estimate could assume.
    """
    cues: list[Cue] = []
    for words, offset in scenes:
        for chunk in _group(words, words_per_cue):
            cues.append(
                Cue(
                    offset + chunk[0].start,
                    offset + chunk[-1].end,
                    _wrap(" ".join(word.text for word in chunk)),
                )
            )
    return cues


def _group(words: Sequence[Any], words_per_cue: int) -> list[list[Any]]:
    """`_split`'s rule -- sentences first, then a word limit -- over objects.

    Grouping the words themselves rather than joining their text and splitting
    it again. The two are the same for ordinary narration and stop being the
    same the moment one word carries a space: the chunks would still be right
    and the timings taken from them would be off by one from there on, for the
    rest of the scene.
    """
    chunks: list[list[Any]] = []
    current: list[Any] = []
    for word in words:
        if not word.text.strip():
            continue
        current.append(word)
        ends_sentence = word.text.rstrip().endswith((".", "!", "?"))
        if ends_sentence or len(current) == words_per_cue:
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks
