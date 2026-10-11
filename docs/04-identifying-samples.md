# 04 – Identifying samples: what "the same sound" means

The model doing this work cannot hear. Every judgement it makes about sound has to come from something it can read. So "is this the same sound?" has to become a measurement, with a threshold that was checked against cases where the answer is known.

## Four meanings of "the same"

| level | meaning | how to tell | where it holds |
|---|---|---|---|
| L0 – the same file | bit for bit | null test: subtract, nothing is left | never inside a mixed and mastered song |
| L1 – the same recording | the same sample, processed by the mix | high NCC, a clear margin over every other file, every hit agreeing | noisy sounds (snare, clap, hat) |
| L2 – sounds the same | indistinguishable in the mix | descriptors within tolerances: pitch ±10 cents, decay ±10 %, harmonics ±3 dB, level ±1 dB | tonal sounds (kick, 808), synthesized replacements |
| L3 – the same kind | same role and character | embeddings (CLAP), names, band energy | searching a library by description |

The pipeline answers L1 where it can and falls back to L2. CLAP is in ableton-live-mcp's `find_similar_sounds` and is good at L3 (P@10 0.72 for text queries over ten drum classes). It was not built to tell two snares of the same kind apart.

## Normalized cross-correlation (NCC)

**Definition.** Two waveforms are lined up (best lag within ±30 ms) and correlated:

```
NCC = <t, c> / (|t| |c|)   in [-1, 1]
```

Both are first filtered to the role's band, so that what plays along (an 808 under a snare) counts for less. 1 means the same shape at some level; −1 means the same shape inverted (still the same sound).

**What makes a high NCC mean "the same recording":**
- **Noise is unique.** Two different snares never share their noise, so their NCC stays low. Two takes of the *same* snare file correlate however they were processed, as long as the processing is mostly linear (EQ, level) and the timing holds.
- **Averaging.** A single hit in a mix is buried; 26 hits lined up and averaged leave the snare and only 1/√26 of what played with each.
- **Bands.** Filtering both sides keeps what does not belong to the sound out of the score.

**What lowers it for the same file:**
- **Transposition.** The noise decorrelates within a few ms when the speed differs. The right file pitched −1 scored 0.15 at 0 semitones and 0.95 at −1, and 0.13 at ±0.25. That is why a semitone-band shortlist proposes transpositions first.
- **Time stretching.**
- **Distortion and heavy compression.**
- **Something that always plays with the hit.** A hat on every snare stays in the average.

## The margin, not the score

The calibration ([03](03-calibration.md)) shows that the score alone cannot decide:
- a wrong hat match reached 0.82 (it had found the snare that always played with it);
- right kicks went as low as 0.84.

What separates right from wrong for noisy sounds is the **margin**: the top match's NCC minus the best NCC of a *different* sound. That means removing copies of the top match first: kits copy each other's files under new names, and 95 % correlation between two library entries is treated as one recording.
- Right noisy matches led by 0.02–0.87, most by more than 0.1.
- Tonal sounds never lead by much: kicks 0.00–0.07, 808s 0.03–0.08.

## More evidence for a candidate

**Per-hit votes.** Each hit on its own is compared with the top three candidates over its first 70 ms. A real match wins nearly every hit, not just the average:

| PIRAMIT sound | wins |
|---|---|
| hat | 187 of 189 |
| even-bar snare | 27 of 27 |
| odd-bar snare (no match) | votes split |

**Null test.** Subtract the candidate (aligned, best gain) and measure what is left, in dB. It is the "same file" test with the mix's leftovers in it:
- the hat leaves −14.3 dB, the next candidate −7.4 dB;
- the snare leaves −10.4 dB, the next −6.4 dB.

**Shape of the transposition curve.** A noisy sound matched at the right transposition falls off within a quarter of a semitone. A broad maximum means a tone, not a sample.

## Verdicts

`library.verdict` turns these into words a person (or a model) acts on:
- **same sample**
- **candidate: maybe the same sample, processed (check by ear)**
- **same family**
- **no close match**

The rules are in [01](01-pipeline.md#match). They come from 30 calibrated sounds, so they are conservative:
- In the last calibration run, all 15 blind templates of noisy sounds that met the "same sample" rule were right, and no oracle case broke it.
- A "candidate" is sometimes wrong.

## Several sounds in one template

A template is the average of hits on one grid position. If a hat plays on every kick, the template holds both: in the hat band it looks like the hat, in the low band like the kick. `match` scores every template in every band and hands out identities strongest first:
- the hat goes to the template where it is strongest (0.98);
- the kick template, whose hat-band match is that same hat at 0.80, takes its best match in another band instead.

A sample matched at ≥ 0.9 by two templates is one sound found twice (`same_as`). PIRAMIT's hat was found three times.

## Things that broke identification here, in order

1. **The lag sign.** `ncc` returns lag d when t is c delayed by d, and `shift(c, d)` lines c up with t. Averaging with the opposite sign blurred every template. Now there is a test.
2. **Different onset definitions.** Templates were aligned to onset-strength peaks, library files to their first rise above −30 dB. Some samples have a quiet lead-in, so the two sat 12–21 ms apart, outside a ±8–10 ms lag window. The true snare scored 0.15 instead of 0.85. Now: templates are aligned on their attack and the lag window is ±30 ms.
3. **Long windows.** A 300 ms snare template held the louder kick two sixteenths later, and that kick decided the match. Now: 150 ms for snares, 80 ms for hats, and only the attack (40–80 ms) for detection.
4. **Transposed samples never tried.** Only the top 200 at 0 semitones were tried at other pitches, and a pitched snare is not among them. Now: the semitone-band shortlist.
