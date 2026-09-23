# Examples

Four files, each for a different purpose.

## `triangle.mp4` — a finished lesson, narrated

What the pipeline produces now. Generated on 2026-09-23 by:

```bash
python cli.py "explain why the angles of a triangle add to 180 degrees"
```

```
4/4 scenes in 306s for $0.5631 over 6 calls
IR clean first try      yes  (1 layout attempt)
code passed L2 first    100%
rules fired             none
worst audio/video gap   +0.09s
```

240 seconds, spoken aloud, subtitles placed from the synthesiser's own word
timings. **Turn the sound on** -- that is the point of this one.

The last line of that summary is the one worth reading. The narration is
synthesised *before* the layout is asked for, so the layout is told how many
seconds each scene must fill rather than guessing from a word count, and the
step durations are then scaled to match exactly. The worst scene ends 0.09s
away from its voice, and what remains is manim rounding animation lengths up
to whole frames. The same prompt before that fitting step was out by 6.2s and
lost its closing words off the end.

## `primes.mp4` — a finished lesson, silent

The same thing one stage earlier, kept because the two together show what the
audio work changed -- and because this run is the one where the layout retry
fired for real.

What the pipeline produces, checked in so it can be watched rather than
described. Generated on 2026-09-20 by:

```bash
python cli.py "explain what a prime number is and why there are infinitely many"
```

```
6/6 scenes in 508s for $1.1107 over 9 calls
IR clean first try      no  (2 layout attempts)
code passed L2 first    100%
escalated to simplify   0%
rules fired             none
```

344 seconds, no audio, narration burned in as subtitles.

Nine calls rather than the seven a clean run needs: the first layout came back
with rule errors, the retry carried them back, and the second was clean. It is
the first run in which a repair loop fired for real and worked, and the lesson
it produced has zero layout warnings.

The mathematics is the model's, not a template. It builds two different factor
trees for 60 and lands both on the same four primes, then gives Euclid's
argument with a worked instance -- 30030 + 1 = 30031 = 59 x 509, neither
factor on the list it started from.

One sample is not a success rate. The ablations in `ROADMAP.md` are what would
turn runs like this into a number worth quoting.

## `pythagoras.json` — a hand-written Scene IR document

Three scenes, written by hand rather than generated, and clean against all
twelve validation rules with pacing drift of 11-15% against a 30% tolerance.

It is what `explainer/spine.py` renders, and `spine.py` is deliberately the
path with no agent and no repair loop around it -- which makes it the baseline
the ablations need. The test suite checks this document against the rules on
every run: a hand-written example that trips its own validator is either a bad
example or a bad rule, and it matters which.

```bash
python -m explainer.spine docs/examples/pythagoras.json runs/spine
```

## `smoke.py` — the toolchain test

Three system-level things have to work before any generated scene can render,
and each fails in a way that looks like something else. This exercises all
three at once: `Text` goes through Cairo rasterisation, `MathTex` shells out
to a LaTeX distribution, and the encode goes through ffmpeg.

```bash
python -m manim -ql --disable_caching docs/examples/smoke.py Smoke
```

Run it before debugging anything downstream. Note `python -m manim` rather than
`manim`: pip installs the executable into a per-user Scripts directory that is
frequently not on PATH, so the bare command reports a missing program on a
machine where the package is installed perfectly well.
