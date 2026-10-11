# 03 – Calibration: beats with a known answer

A similarity score means nothing until you know what it does when the answer is known. `refdissect calibrate` builds that answer key from your own library and scores the pipeline against it.

## The beats

`calibrate.build_beat(seed)` picks, at random from the library inventory:
- a kick;
- a snare or clap;
- a closed hat;
- an open hat;
- an 808 (0.8–4 s);
- a long melodic loop and an acapella, as bleed.

It plays them as a trap beat at 136 BPM for 16 bars:

| sound | pattern (two bars of sixteenths) |
|---|---|
| kick | 0, 7, 10, 16, 23, 26, 30 |
| 808 | with each kick, notes 0, 0, +3, 0, +1, −2, +7 semitones, each cut at the next |
| snare | 8, 24 (beat 3: half time) |
| hats | every eighth except the open hat's, with 16th rolls; velocity 0.55–1.0 |
| open hat | 14 |

**Mastering, like the references.** Mix to −12 dBFS RMS, then +5 dB into a tanh soft clipper. That lands at about −8.6 dB RMS, within the −7 to −10 LUFS range measured on the reference tracks. It then goes through mp3 at 256 kb/s (ffmpeg) and Demucs `htdemucs`.

**Every beat is hard in the same ways:**
- the snare always lands on a hat;
- the 808 starts with every kick;
- the loop and the vocal bleed into every stem.

The six beats used here drew 30 sounds from 8,854 one-shots, plus 865 808s.

## Identification at known hit times (`oracle`)

**Method.** The true hit times are what a perfect drum transcription would give.
1. The hits at those times are averaged.
2. The average is matched against the whole library.
3. The test: is the true sample (or a copy of it under another name) the top match?

**27 of 30.**

| sound | found | NCC of the true sample | margin of the top match over the next different sound |
|---|---|---|---|
| snare / clap | 6/6 | 0.96–0.99 | 0.12–0.78 |
| open hat | 6/6 | 0.93–0.97 | 0.16–0.87 |
| hat | 5/6 | 0.53–0.99 | 0.02–0.71 |
| kick | 4/6 | 0.41–0.99 | 0.00–0.07 |
| 808 (±12 semitones) | 6/6 | 0.90–0.97 | 0.03–0.08 |

![top match NCC against margin](../figures/calibration_identification.png)

**What the plot shows:**
- **Noisy sounds** (snare, clap, hats) are unique as waveforms. The right match stands 0.1–0.9 above everything else.
- **Tonal sounds** (kick, 808) are mostly decaying sines. Any two at the same pitch correlate 0.9+, so even the right match leads by less than 0.08. For them, the waveform tells the *family* reliably and the *exact file* only weakly. Pitch glide, decay and harmonics have to decide ([05](05-808-synthesis.md)).

**The three misses:**
- **A kick that never matched (0.41).** Its file has a long quiet lead-in; Demucs split it oddly.
- **A hat that lost to the snare.** Hats on the eighth grid include beat 3, where the snare always sounds, so the averaged "hat" carried the snare.
- **A kick whose true file came second** by 0.026.

## The whole pipeline, blind (`blind`)

Nothing known:
1. fit the grid (found 135.99–136.02 BPM on all six);
2. discover templates;
3. detect hits;
4. match.

Each template is assigned to the true sound most of its hits belong to (precision ≥ 0.5). Its hits are scored after removing a constant offset: the truth is where the sample file starts, a detection is where the attack is. A sample with 20 ms of quiet lead-in is found 20 ms late every time.

| sound | transcription F1 per beat | mean | identified |
|---|---|---|---|
| snare | 1.00, 1.00, 1.00, 1.00, 1.00, 1.00 | **1.00** | 6/6 templates (NCC 0.955–0.984) |
| open hat | 1.00, 0.00, 1.00, 1.00, 1.00, 1.00 | 0.83 | 5/5 |
| hat | 0.52, 0.94, 0.71, 0.68, 0.94, 0.84 | 0.77 | 7/10 |
| kick | 0.97, 0.00, 0.87, 0.93, 0.67, 0.75 | 0.70 | 5/10 |

Hats and kicks are the weak part, for the reason above: they always play together with something else. Beat 2 is the beat with the odd kick and the snare-covered hat.

## How the numbers moved while building the pipeline

Each fix below was found by these scores, which is the point of having them:

| change | effect |
|---|---|
| hits grouped naively in the full band (first attempt) | 11/24 drum sounds identified blind |
| detection on the attack (40–80 ms) instead of 300 ms | snare detections had landed 352 ms early, locked to the following kick; now F1 1.00 |
| duplicates need the same waveform in *both* bands | snares had been dropped as "the hat" (F1 0 in 5 of 6 beats) |
| scoring after removing a constant offset | a 21 ms lead-in had scored a perfect snare as F1 0 |
| matching over ±30 ms of lag | blind snare identification 0/6 → 6/6 |
| low band (< 200 Hz) for kicks, with tail suppression | synthetic kick F1 0.67 → 1.00 |

## Running it

```bash
uv run refdissect calibrate "$W/work/inventory.json" --library "$W/work/library.npz" \
    --library-808 "$W/work/library_808.npz" --beats 6 --out "$W/work/calibration"
```

Beats already built and separated are reused. `results/calibration/summary.json` is this run, made with `scripts/export_results.py`.

**Limits:**
- Six beats and one genre: thresholds from 30 sounds are a starting point, not a law.
- The beats have no per-track processing (EQ, reverb, saturation on the snare), which real mixes do. Expect real NCCs a little lower for the same sample: PIRAMIT's snare matched at 0.953.
