# 06 – Listening for a model that cannot hear

The person driving this project asked: when the AI puts a sound in Live, how can it know whether it is the sound in the song, using text or some other kind of data?

Claude reads text and looks at images. It does not take audio in. So the listening has to happen in code, and what comes out has to be something a model can read, check and act on. Three kinds of output did that here.

## 1. Numbers in a producer's units

`describe.describe` and `synth808.compare` give, per sound:

```
reference: peak 1.7 ms, decay -6 dB 31 ms, -20 dB 47 ms, centroid 3678 Hz, noisiness 0.022
candidate: peak 6.2 ms, decay -6 dB 25 ms, -20 dB 49 ms, centroid 3367 Hz, noisiness 0.029
808: pitch C1 +26 cents (33.2 Hz), starts at 44 Hz; harmonics 2-6 [-36, -24, -35, -36, -51] dB
```

A number is only as good as how it was read. `describe` reads pitch with pYIN, which reads 14–27 cents sharp under 50 Hz. It still says C1 +26 cents for PIRAMIT's new 808 template, whose sustain settles at C1 +11.8 cents by QIFFT ([01](01-pipeline.md#808-notes-bass808transcribe)). The plan tunes the 808 from the QIFFT reading.

**Each number maps to a knob:**

| difference | knob |
|---|---|
| pitch off by cents | Detune |
| decay too long | Simpler decay, or a gate |
| 3rd harmonic 9 dB high | less drive |
| centroid low | EQ shelf |

A model can turn these into concrete edits, apply them, and measure again.

## 2. Pictures

Spectrograms side by side, plus a difference map (red: the candidate is louder there), as in `figures/piramit_sounds.png`. Looking at that figure, a model can say:
- in the hat band, the library snare stops at about 60 ms while the reference rings on (blue after 60 ms in the difference map); measured, they reach −40 dB at 60 and 70 ms;
- the library hat is a little brighter at its start (red above 4 kHz) and the reference holds more below 2 kHz;
- the synthesized 808 lacks the reference's upper harmonics.

Before replica mode the figure showed 300 ms of the −1 snare. A model read off it a tail to about 150 ms, and a hat 220 ms after every snare (pattern, not the snare). Measured on the levels in the snare band, the tail difference is 9 ms at −40 dB ([02](02-case-study-piramit.md#drums)): a 70 dB colour scale makes a quiet tail look long.

**Pictures and numbers fail in different places:**
- On a 70 dB color scale, the library 808 *looked* just like the reference. The numbers said its 6th–12th harmonics were 12–17 dB too strong (before replica mode).
- The waveform-only synth had the best number (NCC 0.969) and *looks* plainly wrong.

So use both.

## 3. Calibrated verdicts

A score is only meaningful with a threshold checked on cases with a known answer. `calibrate` builds those cases from your own library ([03](03-calibration.md)), and the verdict rules come from them ([04](04-identifying-samples.md)). They tell a model whether "0.95" means "the same file" (noisy sounds, clear margin) or merely "an 808" (every 808 scores that).

`refdissect suite` goes further: 48 beats in 22 scenarios with every hit, 808 note, melodic note and chord known. It scores what the pipeline heard and what Live will play (the plan, rendered offline) on the same strict terms, so a model can see which step loses a sound or a few milliseconds. On the merged replica-mode code: drum hits and 808 onsets F1@20 0.886 heard, 0.667 played (m2g; [03](03-calibration.md)).

## Rules for comparing fairly

- **Average many hits** before judging one sound.
- **Line up and band-limit** both sides the same way: the same attack point, the same filter, enough lag to cover onset uncertainty (±30 ms).
- **Match levels** before comparing spectra. Compare loudness separately.
- **Compare like with like** ("symmetric comparison"). A sound from a mixed, mastered, mp3'd, separated song should be compared with a candidate that went through the same chain. `refdissect compare` measures two stem sets separated by the same model the same way, against the reference's own spread ([11](11-style-beat-lessons.md)). Rendering the Live build and mastering it like the reference is on the [roadmap](10-roadmap.md).
- **Use several measures and require them to agree;** show the picture when they disagree.
- **The ear decides last.** The Live build keeps the original stems muted under the rebuilt parts, so A/B is two clicks.

## A listening report, as a format

What a model should get for each sound:

```json
{"sound": "snare (odd bars)", "verdict": "same sound as the snare of bars 24-31, a semitone down",
 "match": {"sample": "…", "semitones": -1.0025, "transpose": -1, "detune_cents": 0, "ncc": 0.923,
           "margin": 0.025, "per_hit_wins": "28/28", "null_test_db": -8.3},
 "differences": {"decay_-20dB_ms": "-0.2", "centroid_hz": "-362", "tail_-40dB_ms": "-9"},
 "suggested": ["a little brighter: a high shelf, or less filter"],
 "picture": "figures/…png"}
```

The numbers are PIRAMIT's (2026-10-22); the candidate's minus the reference's. `matches.json` already holds the identification part. Turning descriptor differences into suggested edits, and the edit → render → measure loop around them, is the next step ([10](10-roadmap.md)).
