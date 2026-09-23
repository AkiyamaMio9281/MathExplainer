"""Narration to audio, without the network.

`edge_tts` speaks through an endpoint Microsoft does not document and does not
promise to keep. Testing against it would make this suite slow, flaky and
dependent on a third party's uptime, so the module is imported through
`sys.modules` and a fake is put there instead. What is being tested is what
this code does with what comes back -- the tick arithmetic, the failure paths,
and the promise that none of them raise.

That last one is the point of most of these. The pipeline has a silent lesson
to fall back on, and losing the audio must not lose the video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from explainer import speech

TICKS = speech.TICKS_PER_SECOND


# ---------------------------------------------------------------------------
# A synthesiser that never leaves the process
# ---------------------------------------------------------------------------


class FakeCommunicate:
    """Stands in for edge_tts.Communicate. Configured by the module below."""

    def __init__(self, text, voice, *, boundary=None, **kwargs):
        FakeCommunicate.asked.append((text, voice, boundary))
        self.text = text

    def stream_sync(self):
        if FakeCommunicate.raises is not None:
            raise FakeCommunicate.raises
        yield from FakeCommunicate.chunks


class FakeModule:
    Communicate = FakeCommunicate


def words_of(*pairs) -> list[dict]:
    """Boundary chunks as the endpoint sends them: offsets in 100ns ticks."""
    return [
        {
            "type": "WordBoundary",
            "offset": int(start * TICKS),
            "duration": int((end - start) * TICKS),
            "text": text,
        }
        for start, end, text in pairs
    ]


@pytest.fixture
def synthesiser(monkeypatch):
    """Installs the fake, and hands back a knob for what it returns."""
    FakeCommunicate.asked = []
    FakeCommunicate.raises = None
    FakeCommunicate.chunks = [
        {"type": "audio", "data": b"\x00" * 64},
        *words_of((0.0, 0.5, "A"), (0.5, 1.0, "prime"), (1.0, 2.0, "number")),
    ]
    monkeypatch.setitem(sys.modules, "edge_tts", FakeModule)
    # The file is written but never a real MP3, so its length cannot be probed.
    monkeypatch.setattr(speech, "_length", lambda path, fallback=0.0: fallback)
    return FakeCommunicate


# ---------------------------------------------------------------------------
# What comes back
# ---------------------------------------------------------------------------


def test_the_audio_is_written_and_the_words_come_back_in_seconds(synthesiser, tmp_path):
    out = tmp_path / "scene.mp3"
    spoken = speech.speak("A prime number", out)

    assert spoken.ok
    assert out.read_bytes() == b"\x00" * 64
    assert [(w.start, w.end, w.text) for w in spoken.words] == [
        (0.0, 0.5, "A"),
        (0.5, 1.0, "prime"),
        (1.0, 2.0, "number"),
    ]


def test_offsets_are_ticks_and_are_not_mistaken_for_anything_else(synthesiser, tmp_path):
    # A hundred-nanosecond unit is an unusual choice and an easy one to read as
    # milliseconds, which would make every subtitle ten thousand times early.
    synthesiser.chunks = [
        {"type": "audio", "data": b"x"},
        {"type": "WordBoundary", "offset": 12_500_000, "duration": 2_500_000, "text": "hi"},
    ]
    spoken = speech.speak("hi", tmp_path / "a.mp3")

    assert spoken.words[0].start == pytest.approx(1.25)
    assert spoken.words[0].end == pytest.approx(1.5)


def test_word_boundaries_are_asked_for_rather_than_taken(synthesiser, tmp_path):
    # The default is SentenceBoundary, which returns one timing per sentence.
    # A sentence too long for one subtitle then has nowhere to be split.
    speech.speak("A prime number", tmp_path / "a.mp3")

    assert synthesiser.asked[0][2] == "WordBoundary"
    assert speech.BOUNDARY == "WordBoundary"


def test_the_voice_is_passed_through(synthesiser, tmp_path):
    speech.speak("hello", tmp_path / "a.mp3", voice="en-GB-SoniaNeural")

    assert synthesiser.asked[0][1] == "en-GB-SoniaNeural"


def test_the_length_is_measured_from_the_file_not_the_last_word(
    synthesiser, tmp_path, monkeypatch
):
    # An MP3 usually carries a little silence past the last syllable, and it is
    # the file's length that has to line up with the clip, not the speech's.
    monkeypatch.setattr(speech, "_length", lambda path, fallback=0.0: 2.4)
    spoken = speech.speak("A prime number", tmp_path / "a.mp3")

    assert spoken.seconds == 2.4
    assert spoken.words[-1].end == 2.0


def test_an_unreadable_file_falls_back_to_the_last_word(synthesiser, tmp_path, monkeypatch):
    monkeypatch.setattr(speech, "_length", lambda path, fallback=0.0: fallback)
    spoken = speech.speak("A prime number", tmp_path / "a.mp3")

    assert spoken.seconds == 2.0


# ---------------------------------------------------------------------------
# Nothing here raises
# ---------------------------------------------------------------------------


def test_an_endpoint_that_throws_is_an_outcome_not_an_exception(synthesiser, tmp_path):
    synthesiser.raises = ConnectionError("no route to host")
    spoken = speech.speak("hello", tmp_path / "a.mp3")

    assert not spoken.ok
    assert "ConnectionError" in spoken.error
    assert "no route to host" in spoken.error
    assert not (tmp_path / "a.mp3").exists()


def test_a_stream_with_no_audio_is_a_failure_not_an_empty_file(synthesiser, tmp_path):
    # Boundaries without audio would otherwise write a zero-byte MP3 that
    # ffmpeg rejects several stages later, naming a file nobody wrote.
    synthesiser.chunks = words_of((0.0, 1.0, "hello"))
    spoken = speech.speak("hello", tmp_path / "a.mp3")

    assert not spoken.ok
    assert "no audio" in spoken.error
    assert not (tmp_path / "a.mp3").exists()


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_there_is_nothing_to_say(synthesiser, tmp_path, text):
    spoken = speech.speak(text, tmp_path / "a.mp3")

    assert not spoken.ok
    assert synthesiser.asked == []


def test_a_missing_dependency_is_reported_rather_than_imported(monkeypatch, tmp_path):
    monkeypatch.setattr(speech, "available", lambda: False)
    spoken = speech.speak("hello", tmp_path / "a.mp3")

    assert not spoken.ok
    assert "edge-tts" in spoken.error


def test_the_directory_is_made_rather_than_assumed(synthesiser, tmp_path):
    out = tmp_path / "deep" / "deeper" / "a.mp3"
    assert speech.speak("hello", out).ok
    assert out.is_file()


# ---------------------------------------------------------------------------
# A lesson's worth
# ---------------------------------------------------------------------------


def test_each_scene_is_spoken_into_its_own_file_and_keyed_by_id(synthesiser, tmp_path):
    spoken = speech.narrate(
        [("intro", "A prime number"), ("proof", "Two is prime")], tmp_path / "audio"
    )

    assert set(spoken) == {"intro", "proof"}
    assert spoken["intro"].path == tmp_path / "audio" / "intro.mp3"
    assert all(s.ok for s in spoken.values())


def test_one_scene_failing_leaves_the_others_spoken(synthesiser, tmp_path):
    spoken = speech.narrate([("intro", "hello"), ("proof", "   ")], tmp_path / "audio")

    assert spoken["intro"].ok
    assert not spoken["proof"].ok


def said(words: int, seconds: float) -> speech.Spoken:
    return speech.Spoken(
        True, seconds=seconds, words=tuple([speech.Word(0.0, 0.0, "w")] * words)
    )


def test_the_rate_is_measured_across_everything_that_was_said():
    # Pooled rather than averaged per scene: a short scene and a long one
    # should not count equally towards the rate. Here the two differ -- one
    # minute at 120 wpm and half a minute at 180 average to 150, but 210 words
    # really were said over 90 seconds, which is 140.
    spoken = {"a": said(120, 60.0), "b": said(90, 30.0)}

    assert spoken["a"].words_per_minute == pytest.approx(120.0)
    assert spoken["b"].words_per_minute == pytest.approx(180.0)
    assert speech.measured_rate(spoken) == pytest.approx(140.0)


def test_a_failed_scene_does_not_drag_the_rate_down(synthesiser):
    spoken = {
        "a": speech.Spoken(True, seconds=60.0, words=tuple([speech.Word(0, 0, "w")] * 150)),
        "b": speech.Spoken(False, error="the endpoint refused"),
    }

    assert speech.measured_rate(spoken) == pytest.approx(150.0)


def test_nothing_spoken_has_no_rate_rather_than_a_zero(synthesiser):
    assert speech.measured_rate({}) is None
    assert speech.measured_rate({"a": speech.Spoken(False)}) is None


def test_the_rate_a_scene_reports_matches_the_one_pooled_over_it():
    spoken = speech.Spoken(
        True, seconds=30.0, words=tuple([speech.Word(0, 0, "w")] * 75)
    )

    assert spoken.words_per_minute == pytest.approx(150.0)
    assert speech.measured_rate({"a": spoken}) == pytest.approx(150.0)


def test_a_scene_with_no_audio_has_no_rate_rather_than_a_division_by_zero():
    assert speech.Spoken(False).words_per_minute == 0.0
