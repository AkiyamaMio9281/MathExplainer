"""Narration to audio, and the timings that come back with it.

Adding speech inverts the timing relationship the rest of the pipeline was
built on. Without it, the narration's word count *decides* how long a scene
runs. With it, the synthesiser decides, and the layout has to be told. So this
stage runs **immediately after planning and before layout** -- the words are
final once the plan is, so nothing has to be re-synthesised when a layout is
repaired.

That ordering is what makes the audio and the animation line up by
construction rather than by adjustment afterwards. The alternative -- render
first, then stretch or pad -- trades either quality or honesty for not having
to move a stage.

**Word timings come back from the synthesiser**, so subtitles can be placed on
the syllable rather than shared out by word count. The estimate that replaced
was reasonable and still wrong: a long word and a short one do not take the
same time to say.

## The dependency, stated plainly

`edge-tts` speaks through the endpoint Microsoft Edge's read-aloud uses. It is
free, needs no key, and is not an official API: it can change or stop without
notice, and it needs a network connection. A failure here is never fatal --
`speak` returns a `Spoken` that says it did not work, and the pipeline falls
back to the silent-with-subtitles lesson it made before.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

#: Microsoft's own read-aloud voices. Aria is the clearest of the US English
#: ones for technical narration; the rest are a `--voice` away.
VOICE = "en-US-AriaNeural"

#: The synthesiser reports offsets in 100-nanosecond units.
TICKS_PER_SECOND = 10_000_000

#: Ask for word-level timings rather than the default sentence ones. Sentences
#: would be simpler and are not enough: a sentence too long for one subtitle
#: has to be split, and splitting it needs to know where its words fall.
BOUNDARY = "WordBoundary"


@dataclass(frozen=True)
class Word:
    """One spoken word and when it is said, in seconds from the start."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Spoken:
    """One scene's narration as audio, or the reason there is none."""

    ok: bool
    path: Path | None = None
    seconds: float = 0.0
    words: tuple[Word, ...] = ()
    error: str = ""
    synthesis_seconds: float = 0.0

    @property
    def words_per_minute(self) -> float:
        return len(self.words) / self.seconds * 60.0 if self.seconds > 0 else 0.0


def available() -> bool:
    """Whether the synthesiser can be reached at all."""
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return False
    return True


def speak(text: str, path: Path, *, voice: str = VOICE) -> Spoken:
    """Say *text* into *path*, and report how long it took to say.

    Never raises. The pipeline has a silent lesson to fall back on, and losing
    the audio should not lose the video.
    """
    if not text.strip():
        return Spoken(False, error="nothing to say")
    if not available():
        return Spoken(False, error="edge-tts is not installed")

    import edge_tts

    started = time.perf_counter()
    audio = bytearray()
    words: list[Word] = []
    try:
        stream = edge_tts.Communicate(text, voice, boundary=BOUNDARY).stream_sync()
        for chunk in stream:
            kind = chunk.get("type")
            if kind == "audio":
                audio.extend(chunk["data"])
            elif kind == BOUNDARY:
                start = chunk["offset"] / TICKS_PER_SECOND
                words.append(
                    Word(
                        start=start,
                        end=start + chunk["duration"] / TICKS_PER_SECOND,
                        text=str(chunk["text"]),
                    )
                )
    except Exception as exc:  # the endpoint is unofficial; anything can happen
        return Spoken(False, error=f"{type(exc).__name__}: {exc}")

    if not audio:
        return Spoken(False, error="the synthesiser returned no audio")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(audio))

    # The last word's end is where speech stops; the file may carry a little
    # trailing silence past it, and the file's length is what has to be lined
    # up with, so it is measured rather than inferred.
    return Spoken(
        True,
        path=path,
        seconds=_length(path, fallback=words[-1].end if words else 0.0),
        words=tuple(words),
        synthesis_seconds=time.perf_counter() - started,
    )


def _length(path: Path, fallback: float = 0.0) -> float:
    from .video import duration

    measured = duration(path)
    return measured if measured else fallback


def narrate(
    scenes: list[tuple[str, str]], directory: Path, *, voice: str = VOICE
) -> dict[str, Spoken]:
    """Speak each ``(scene id, narration)``, into ``directory/<id>.mp3``."""
    directory.mkdir(parents=True, exist_ok=True)
    return {
        scene_id: speak(narration, directory / f"{scene_id}.mp3", voice=voice)
        for scene_id, narration in scenes
    }


def measured_rate(spoken: dict[str, Spoken]) -> float | None:
    """The voice's actual pace across everything that was said.

    Worth computing rather than assuming: every duration in the layout stage
    and every pacing rule is stated in words per minute, and the number they
    use should be the one the voice really speaks at.
    """
    words = sum(len(s.words) for s in spoken.values() if s.ok)
    seconds = sum(s.seconds for s in spoken.values() if s.ok)
    return words / seconds * 60.0 if seconds > 0 else None
