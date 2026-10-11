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
- the library snare stops at 75 ms, while the reference rings to about 150 ms (needs a tail);
- a hat sits 220 ms after every snare (pattern, not the snare);
- the synthesized 808 lacks the reference's upper harmonics.

**Pictures and numbers fail in different places:**
- On a 70 dB color scale, the library 808 *looks* just like the reference. The numbers say its 6th–12th harmonics are 12–17 dB too strong.
- The waveform-only synth had the best number (NCC 0.969) and *looks* plainly wrong.

So use both.

## 3. Calibrated verdicts

A score is only meaningful with a threshold checked on cases with a known answer. `calibrate` builds those cases from your own library ([03](03-calibration.md)), and the verdict rules come from them ([04](04-identifying-samples.md)). They tell a model whether "0.95" means "the same file" (noisy sounds, clear margin) or merely "an 808" (every 808 scores that).

## Rules for comparing fairly

- **Average many hits** before judging one sound.
- **Line up and band-limit** both sides the same way: the same attack point, the same filter, enough lag to cover onset uncertainty (±30 ms).
- **Match levels** before comparing spectra. Compare loudness separately.
- **Compare like with like** ("symmetric comparison"). A sound from a mixed, mastered, mp3'd, separated song should be compared with a candidate that went through the same chain. Render the Live version, master it the same way, separate it, and compare stems with stems. This is on the [roadmap](10-roadmap.md).
- **Use several measures and require them to agree;** show the picture when they disagree.
- **The ear decides last.** The Live build keeps the original stems muted under the rebuilt parts, so A/B is two clicks.

## A listening report, as a format

What a model should get for each sound:

```json
{"sound": "snare (even bars)", "verdict": "same sample",
 "match": {"sample": "…", "semitones": -1, "ncc": 0.953, "margin": 0.075, "per_hit_wins": "27/27",
           "null_test_db": -10.4},
 "differences": {"decay_-20dB_ms": "+2", "centroid_hz": "-311", "tail": "reference rings to ~150 ms"},
 "suggested": ["add a short reverb or lengthen the decay to ~150 ms"],
 "picture": "figures/…png"}
```

`matches.json` already holds the identification part. Turning descriptor differences into suggested edits, and the edit → render → measure loop around them, is the next step ([10](10-roadmap.md)).
