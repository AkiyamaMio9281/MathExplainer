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
            # Absolute, for the same reason the module is: this helper made
            # the identical mistake first time round.
            str(path.resolve()),
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


@pytest.mark.slow
def test_relative_clip_paths_work(tmp_path, monkeypatch):
    # The children run with a cwd that is not this process's, so a relative
    # path gets resolved against it a second time. Every test above uses
    # pytest's absolute tmp_path; the pipeline builds runs/<slug>, which found
    # this the hard way.
    monkeypatch.chdir(tmp_path)
    room = Path("runs") / "lesson"
    clips = [make_clip(room / "a.mp4"), make_clip(room / "b.mp4")]

    result = video.concat(clips, room / "lesson.mp4")

    assert result.ok, result.error
    assert video.probe(Path("runs/lesson/lesson.mp4")) is not None
    assert video.duration(clips[0]) == pytest.approx(1.0, abs=0.3)


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def test_a_scene_is_split_into_readable_cues():
    cues = video.cues_for("one two three " * 10, start=0.0, duration=30.0)

    assert len(cues) == 3  # thirty words at twelve to a cue
    assert all(len(c.text.split()) <= video.WORDS_PER_CUE for c in cues)


def test_time_is_shared_out_by_word_count():
    # A long phrase stays up longer than a short one.
    cues = video.cues_for("a " * 18, start=0.0, duration=30.0)

    first, second = cues
    assert (first.end - first.start) == pytest.approx(20.0)
    assert (second.end - second.start) == pytest.approx(10.0)


def test_the_last_cue_lands_exactly_on_the_end_of_the_scene():
    # Exactly, not approximately. The last cue is given what is left rather
    # than its proportional share, because four float shares that should sum
    # to the duration do not, and a subtitle that outlasts its clip by a
    # rounding error is a subtitle over the next scene.
    # 14 words over 45.7s is one of the 577 combinations in a 1393-case sweep
    # where the naive sum lands on 45.70000000000001 instead.
    cues = video.cues_for("word " * 14, start=0.0, duration=45.7)

    assert cues[0].start == 0.0
    assert cues[-1].end == 45.7


def test_scenes_are_laid_end_to_end():
    cues = video.cues_from([("first scene words", 4.0), ("second scene words", 6.0)])

    assert cues[0].start == 0.0
    assert cues[0].end == pytest.approx(4.0)
    assert cues[-1].end == pytest.approx(10.0)


def test_a_scene_with_no_narration_contributes_nothing_but_still_advances_time():
    cues = video.cues_from([("", 4.0), ("spoken words here", 6.0)])

    assert len(cues) == 1
    assert cues[0].start == pytest.approx(4.0)


def test_a_scene_of_no_length_is_skipped_rather_than_dividing_by_zero():
    assert video.cues_for("some words", start=0.0, duration=0.0) == []


def test_a_long_cue_wraps_onto_at_most_two_lines():
    # A third line covers the animation it is describing.
    cue = video.cues_for("supercalifragilistic " * 12, 0.0, 10.0)[0]

    assert cue.text.count("\n") <= 1


def test_the_srt_is_shaped_the_way_players_expect():
    text = video.to_srt(video.cues_from([("hello there", 2.5)]))

    assert text.startswith("1\n00:00:00,000 --> 00:00:02,500\nhello there")


def test_timestamps_round_without_producing_a_thousand_milliseconds():
    # 0.9999 rounds to 1000ms, which is not a timestamp.
    text = video.to_srt([video.Cue(0.0, 0.9999, "x")])

    assert "00:00:01,000" in text
    assert ",1000" not in text


@pytest.mark.slow
def test_burning_subtitles_replaces_the_lesson_in_place(tmp_path):
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    before = clip.stat().st_size

    result = video.subtitle(clip, video.cues_from([("a spoken line", 2.0)]))

    assert result.ok, result.error
    assert result.method == "subtitles"
    # Same name, different bytes, still a video of the same length.
    assert result.output == clip.resolve()
    assert clip.stat().st_size != before
    assert video.duration(clip) == pytest.approx(2.0, abs=0.3)


@pytest.mark.slow
def test_the_subtitle_file_does_not_survive_the_run(tmp_path):
    clip = make_clip(tmp_path / "lesson.mp4", seconds=1.0)
    video.subtitle(clip, video.cues_from([("a line", 1.0)]))

    assert not list(tmp_path.glob("*.srt"))
    assert not list(tmp_path.glob("*-subtitled.mp4"))


@pytest.mark.slow
def test_a_path_with_spaces_and_a_drive_letter_still_burns(tmp_path):
    # A filtergraph parses its own argument, so a Windows path inside one needs
    # its backslashes and its drive-letter colon escaped. The filter is handed
    # a bare filename with cwd set alongside it instead.
    room = tmp_path / "a directory with spaces"
    clip = make_clip(room / "the lesson.mp4", seconds=1.0)

    result = video.subtitle(clip, video.cues_from([("a line", 1.0)]))

    assert result.ok, result.error


@pytest.mark.slow
def test_subtitle_lesson_times_itself_to_the_clips_that_were_rendered(tmp_path):
    # Not to what the IR asked for: what was requested and what manim produced
    # differ a little every scene, and the drift accumulates.
    first = make_clip(tmp_path / "a.mp4", seconds=1.0)
    second = make_clip(tmp_path / "b.mp4", seconds=3.0)
    joined = video.concat([first, second], tmp_path / "lesson.mp4")

    result = video.subtitle_lesson(
        joined.output, [("first words", first), ("second words", second)]
    )

    assert result.ok, result.error


def test_nothing_to_say_is_refused_without_starting_ffmpeg(tmp_path):
    result = video.subtitle(tmp_path / "missing.mp4", [])

    assert not result.ok
    assert "no narration" in result.error


@pytest.mark.slow
def test_a_failed_burn_leaves_the_original_lesson_untouched(tmp_path):
    # Losing the subtitles must not lose the lesson.
    clip = tmp_path / "lesson.mp4"
    clip.write_bytes(b"not a video")

    result = video.subtitle(clip, video.cues_from([("a line", 1.0)]))

    assert not result.ok
    assert clip.read_bytes() == b"not a video"
    assert not list(tmp_path.glob("*-subtitled.mp4"))


def test_a_cue_does_not_run_across_a_sentence_boundary():
    # Read off a real frame before this existed: "rows of three. Twelve is
    # flexible. Now take seven dots and try" -- three fragments of three
    # different sentences, which is a caption only in the technical sense.
    narration = (
        "Twelve dots can be four rows of three. Twelve is flexible. "
        "Now take seven dots and try to make a rectangle."
    )
    cues = video.cues_for(narration, 0.0, 20.0)

    assert [c.text.replace(video.NEWLINE, " ") for c in cues] == [
        "Twelve dots can be four rows of three.",
        "Twelve is flexible.",
        "Now take seven dots and try to make a rectangle.",
    ]


def test_a_sentence_longer_than_the_limit_is_still_broken_up():
    cues = video.cues_for("word " * 30, 0.0, 10.0)

    assert len(cues) == 3
    assert all(len(c.text.split()) <= video.WORDS_PER_CUE for c in cues)


def test_a_short_sentence_stays_whole_however_short():
    cues = video.cues_for("Yes. No. Maybe so.", 0.0, 6.0)

    assert [c.text for c in cues] == ["Yes.", "No.", "Maybe so."]


# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------


def make_tone(path: Path, seconds=1.0, frequency=440) -> Path:
    """A clip of audible sine, for checking where it ended up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done = run(
        [
            video.FFMPEG, "-y", "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:duration={seconds}",
            str(path.resolve()),
        ],
        cwd=path.parent,
        timeout=120,
    )
    assert done.ok, done.failure_text()
    return path


def loudness(path: Path, start: float, end: float) -> float:
    """Mean volume in dB over one window. Digital silence reports about -91."""
    done = run(
        [
            video.FFMPEG, "-i", str(path.resolve()),
            "-af", f"atrim=start={start}:end={end},volumedetect",
            "-f", "null", "-",
        ],
        cwd=path.parent,
        timeout=120,
    )
    assert done.ok, done.failure_text()
    for line in (done.stdout + done.stderr).splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    raise AssertionError(f"volumedetect said nothing:\n{done.stdout}{done.stderr}")


SILENT = -80.0  # anything quieter than this is silence, not speech


@pytest.mark.slow
def test_each_scenes_narration_lands_at_its_own_start(tmp_path):
    # The property the whole audio stage rests on. Two one-second tones, the
    # second placed at five seconds, must be audible there and nowhere else --
    # concatenating them instead would put the second at one second.
    first = make_tone(tmp_path / "a.mp3")
    second = make_tone(tmp_path / "b.mp3")

    built = video.audio_track([(first, 0.0), (second, 5.0)], tmp_path / "t.m4a", 8.0)

    assert built.ok, built.error
    assert loudness(built.output, 0.0, 1.0) > SILENT      # first scene
    assert loudness(built.output, 1.5, 4.5) < SILENT      # the gap between
    assert loudness(built.output, 5.0, 6.0) > SILENT      # second scene
    # The track ends with the last word rather than being padded out to the
    # video's length: a player carries on through silence it does not have.
    assert video.duration(built.output) == pytest.approx(6.0, abs=0.15)


@pytest.mark.slow
def test_a_scene_that_runs_long_does_not_shift_the_ones_after_it(tmp_path):
    # Placement rather than concatenation is what bounds the error. A scene
    # whose voice overruns bleeds into the next one; it does not push every
    # later scene out by the overrun, and it does not accumulate.
    long_one = make_tone(tmp_path / "long.mp3", seconds=3.0)
    late = make_tone(tmp_path / "late.mp3", seconds=1.0)

    built = video.audio_track([(long_one, 0.0), (late, 5.0)], tmp_path / "t.m4a", 8.0)

    assert built.ok, built.error
    # The late scene is still at five seconds, not at three.
    assert loudness(built.output, 3.5, 4.8) < SILENT
    assert loudness(built.output, 5.1, 5.9) > SILENT


@pytest.mark.slow
def test_the_track_is_cut_to_the_length_of_the_video(tmp_path):
    # An audio stream longer than the video makes a file whose scrubber runs
    # past the last frame, and whichever player is asked disagrees about how
    # long the lesson is.
    tone = make_tone(tmp_path / "a.mp3", seconds=10.0)

    built = video.audio_track([(tone, 0.0)], tmp_path / "t.m4a", 4.0)

    assert built.ok, built.error
    assert video.duration(built.output) == pytest.approx(4.0, abs=0.15)


@pytest.mark.slow
def test_one_voice_stays_one_voice_however_many_scenes_there_are(tmp_path):
    # ffmpeg's amix normalises by input count unless told not to, so the same
    # narration would get quieter the more scenes a lesson had.
    tone = make_tone(tmp_path / "a.mp3")
    alone = video.audio_track([(tone, 0.0)], tmp_path / "one.m4a", 8.0)
    among = video.audio_track(
        [(tone, 0.0), (tone, 2.0), (tone, 4.0), (tone, 6.0)], tmp_path / "four.m4a", 8.0
    )

    assert alone.ok and among.ok
    assert loudness(among.output, 0.0, 1.0) == pytest.approx(
        loudness(alone.output, 0.0, 1.0), abs=0.5
    )


@pytest.mark.slow
def test_the_sound_reaches_the_lesson_without_re_encoding_the_picture(tmp_path):
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    before = video.probe(clip)
    tone = make_tone(tmp_path / "a.mp3", seconds=2.0)

    result = video.soundtrack(clip, [(tone, 0.0)], total=2.0)

    assert result.ok, result.error
    assert result.output == clip
    assert has_audio(clip)
    # The picture is the same picture: copied, not encoded a second time.
    assert video.probe(clip) == before


@pytest.mark.slow
def test_the_track_survives_the_run_as_a_file_of_its_own(tmp_path):
    # Two passes rather than one, so a track that would not build is
    # distinguishable from one that would not attach -- and so it can be
    # listened to on its own.
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    video.soundtrack(clip, [(make_tone(tmp_path / "a.mp3", 2.0), 0.0)], total=2.0)

    assert (tmp_path / "lesson-narration.m4a").is_file()


@pytest.mark.slow
def test_a_lesson_keeps_its_name_when_it_gains_a_voice(tmp_path):
    # ffmpeg cannot read and write one file, so this writes a neighbour and
    # replaces. The neighbour must not survive: two lessons in the directory
    # and only one of them with sound is worse than either.
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    video.soundtrack(clip, [(make_tone(tmp_path / "a.mp3", 2.0), 0.0)], total=2.0)

    assert clip.is_file()
    assert not (tmp_path / "lesson-sound.mp4").exists()


def has_audio(path: Path) -> bool:
    done = run(
        [
            video.FFPROBE, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0",
            str(path.resolve()),
        ],
        cwd=path.parent,
        timeout=60,
    )
    return "audio" in done.stdout


def test_nothing_to_say_is_reported_rather_than_run(tmp_path):
    assert not video.audio_track([], tmp_path / "t.m4a", 4.0).ok


def test_muxing_onto_a_lesson_that_is_not_there_says_which_file(tmp_path):
    result = video.mux(tmp_path / "gone.mp4", tmp_path / "also-gone.m4a")

    assert not result.ok
    assert "gone.mp4" in result.error


@pytest.mark.slow
def test_a_track_that_will_not_build_is_not_muxed(tmp_path):
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    result = video.soundtrack(clip, [(tmp_path / "missing.mp3", 0.0)], total=2.0)

    assert not result.ok
    assert not has_audio(clip)


# ---------------------------------------------------------------------------
# Subtitles from what was actually said
# ---------------------------------------------------------------------------


def spoken(*pairs) -> list:
    from explainer.speech import Word

    return [Word(start, end, text) for start, end, text in pairs]


def test_a_cue_starts_and_ends_when_its_words_are_said():
    words = spoken((0.0, 0.4, "Two"), (0.4, 0.9, "is"), (1.6, 2.4, "prime."))

    cues = video.spoken_cues([(words, 0.0)])

    assert len(cues) == 1
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(2.4)
    assert cues[0].text == "Two is prime."


def test_the_timings_are_the_synthesisers_not_a_share_of_the_scene():
    # The estimate this replaces gave every word the same length. Here the
    # narration is said in the first two seconds of a thirty-second scene, and
    # the cue must come down when the voice does rather than hold for thirty.
    cues = video.spoken_cues([(spoken((0.0, 0.5, "Hello"), (0.5, 2.0, "there.")), 0.0)])

    assert cues[0].end == pytest.approx(2.0)


def test_each_scenes_words_are_offset_to_where_that_scene_starts():
    # The synthesiser times each scene from its own zero.
    words = spoken((0.0, 1.0, "Hello."))

    cues = video.spoken_cues([(words, 0.0), (words, 12.5)])

    assert [c.start for c in cues] == pytest.approx([0.0, 12.5])
    assert [c.end for c in cues] == pytest.approx([1.0, 13.5])


def test_a_sentence_too_long_for_one_cue_is_split_on_its_own_timings():
    words = spoken(*[(i * 0.5, i * 0.5 + 0.4, f"w{i}") for i in range(20)])

    cues = video.spoken_cues([(words, 0.0)], words_per_cue=12)

    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(11 * 0.5 + 0.4)   # the twelfth word
    assert cues[1].start == pytest.approx(12 * 0.5)       # the thirteenth
    assert cues[1].end == pytest.approx(19 * 0.5 + 0.4)


def test_cues_break_at_sentence_ends_here_too():
    words = spoken(
        (0.0, 0.4, "One."), (1.0, 1.4, "Two."), (2.0, 2.4, "Three.")
    )

    cues = video.spoken_cues([(words, 0.0)])

    assert [c.text for c in cues] == ["One.", "Two.", "Three."]


def test_a_scene_with_nothing_said_contributes_no_cues():
    cues = video.spoken_cues([([], 0.0), (spoken((0.0, 1.0, "Hello.")), 5.0)])

    assert len(cues) == 1
    assert cues[0].start == pytest.approx(5.0)


def test_spoken_cues_wrap_the_same_way_estimated_ones_do():
    words = spoken(*[(i * 0.3, i * 0.3 + 0.2, "supercalifragilistic") for i in range(4)])

    cue = video.spoken_cues([(words, 0.0)])[0]

    assert cue.text.count(video.NEWLINE) == 1


@pytest.mark.slow
def test_the_narration_is_copied_rather_than_encoded_a_second_time(tmp_path):
    # `soundtrack` writes AAC in an M4A, which mp4 takes as it is. Encoding it
    # again would be a second lossy pass over speech that is already at a low
    # bitrate.
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    result = video.soundtrack(clip, [(make_tone(tmp_path / "a.mp3", 2.0), 0.0)], total=2.0)

    assert result.ok, result.error
    assert result.method == "mux"


@pytest.mark.slow
def test_a_stream_mp4_will_not_carry_is_encoded_rather_than_refused(
    tmp_path, monkeypatch
):
    # A copy the container rejects is a fallback, not a failure -- the same
    # shape the concat path uses when the clips disagree.
    #
    # The refusal has to be induced. This ffmpeg build muxes PCM, Vorbis, Opus
    # and MP3 into mp4 without complaint, so there is no audio file to hand it
    # that would make the copy fail; what is being tested is that a failed
    # copy is retried as an encode, so the first copy is made to fail.
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    tone = make_tone(tmp_path / "a.mp3", seconds=2.0)
    real = video.run
    seen = []

    def once(args, **kwargs):
        seen.append(args)
        if len(seen) == 1:
            assert "copy" in args, "the first attempt should be the copy"
            return real([video.FFMPEG, "-thereisnosuchflag"], **kwargs)
        return real(args, **kwargs)

    monkeypatch.setattr(video, "run", once)
    result = video.mux(clip, tone)

    assert result.ok, result.error
    assert result.method == "mux-reencode"
    assert len(seen) == 2
    assert has_audio(clip)


@pytest.mark.slow
def test_a_mux_that_fails_both_ways_leaves_the_lesson_as_it_was(tmp_path, monkeypatch):
    clip = make_clip(tmp_path / "lesson.mp4", seconds=2.0)
    before = clip.read_bytes()
    real = video.run
    monkeypatch.setattr(
        video, "run", lambda args, **kw: real([video.FFMPEG, "-nope"], **kw)
    )

    result = video.mux(clip, make_tone(tmp_path / "a.mp3", 2.0))

    assert not result.ok
    assert clip.read_bytes() == before
    assert not (tmp_path / "lesson-sound.mp4").exists()


def test_a_word_carrying_a_space_does_not_slide_every_later_cue():
    # The first version joined the word texts and split the string again,
    # which desynchronises from the timings at the first word containing
    # whitespace -- and stays wrong for the rest of the scene.
    words = spoken(
        (0.0, 1.0, "New York"), (1.0, 2.0, "is"), (2.0, 3.0, "big.")
    )

    cues = video.spoken_cues([(words, 0.0)])

    assert len(cues) == 1
    assert cues[0].end == pytest.approx(3.0)
    assert cues[0].text == "New York is big."


def test_an_empty_word_is_skipped_rather_than_given_a_cue():
    words = spoken((0.0, 1.0, "Hello."), (1.0, 1.0, "  "), (1.2, 2.0, "There."))

    cues = video.spoken_cues([(words, 0.0)])

    assert [c.text for c in cues] == ["Hello.", "There."]


@pytest.mark.parametrize(
    "narration",
    [
        "One two three. Four five six.",
        "A sentence with rather a lot of words in it indeed, going on and on past twelve.",
        "Short. Then a longer one that needs breaking up because it has many words.",
    ],
)
def test_the_two_splitters_group_the_same_words_the_same_way(narration):
    # `cues_for` splits a string, `spoken_cues` groups objects, and they are
    # meant to be the same rule. If they drift apart a lesson's captions change
    # shape depending on whether the narration was spoken, which is not a
    # difference anyone asked for.
    from explainer.speech import Word

    tokens = narration.split()
    words = [Word(i, i + 1, t) for i, t in enumerate(tokens)]

    by_string = video._split(narration, video.WORDS_PER_CUE)
    by_object = video._group(words, video.WORDS_PER_CUE)

    assert [[w.text for w in chunk] for chunk in by_object] == by_string
