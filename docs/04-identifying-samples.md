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
- **Transposition.** The noise decorrelates as soon as the speed differs. The right file pitched −1 scored 0.15 at 0 semitones and 0.95 at −1, and 0.13 at ±0.25. Within a semitone it is a matter of cents: the true file's NCC halves within 1–5 cents (an open hat 0.97 at 0 and 0.59 at 1 cent; a snare 0.98, and 0.56 at 5 cents). That is why the transposition is proposed first, by a semitone-band shortlist and by the speckle below.
- **Time stretching.**
- **Distortion and heavy compression.**
- **Something that always plays with the hit.** A hat on every snare stays in the average.

## Samples at any pitch: the speckle

A sample is often played between semitones: a snare 0.70 semitones down was at best found at −1, and played there, 30 cents off.

**The speckle** is the fine structure of a noisy one-shot's magnitude spectrum: dB on a 1-cent grid, minus its 150-cent moving average. It is fixed by the recording, and played faster or slower the whole spectrum moves along the log-frequency axis. So, per band (snare 800 Hz–14 kHz over 150 ms, hat 3.5–16 kHz over 80 ms):
1. Every library sound of a noisy role has its speckle in the index.
2. A template's speckle is correlated with all of them over ±12 semitones at once, through the FFT. The peak gives the file; its position gives the transposition. No time alignment and no resampling are needed.
3. The 20 best by speckle and the 20 best of the whole-semitone search are tried by NCC at their speckle transposition, ±1.5 cents in quarter-cent steps. NCC still gives the verdict.

The whole-semitone search stays in, because a template that holds a louder sound hides its own speckle. The drum_pitch hat (+25 cents) ranked 1,090th by speckle and 8th by NCC at 0 semitones; at its speckle transposition, NCC found it at 0.81.

**Index.** `index-library` writes `library_speckle.npz` next to `library.npz`; `index-speckle` adds one to a library indexed before.
- 6,718 of 8,854 one-shots: every role but kick, 808 and melodic/loop.
- 102 MB, built in 4.6–6.0 s.
- `Library.load` takes it only when its version, settings and entries match. Library and index together peak at 690 MB.

**Measured** (calibration v2, htdemucs stems, templates of the hits at the true times):

| test | whole semitones | speckle + NCC |
|---|---|---|
| 105 drum sounds of 20 beats, each in its own band | 62 | 71 |
| drum_pitch beats (snare −0.70, hat +0.25, open hat −2.00 semitones) | 2 of 8 | 7 of 8 |
| producer beats (pitched copies, −6 to +8 semitones) | 9 of 22 | 14 of 22 |
| 57 noisy sounds, each played 0, +32.5, −47.5, −100, +240 cents off | 65 of 285 | 191 of 285 |
| of those: snares / open hats / hats / claps | 28/70, 17/80, 6/80, 4/20 | 65/70, 70/80, 31/80, 15/20 |

- When right, the transposition is within 1.5 cents (median 0, 90th percentile 0.5 cents; 186 cases).
- A query takes 1.9 s instead of 1.5 s (median of both bands, measured with one other job on the machine).
- The plan plays the found pitch: Transpose in whole semitones, Detune in whole cents. On the suite no pad plays its sound's file off pitch any more (m2g, m3); with whole semitones the drum_pitch snare and hat played 30 and 25 cents off.

**Tonal sounds keep whole semitones.** The kick band uses the semitone search: other kicks at fractional transpositions fit a kick better than its own file. A template found in the low band is still matched at any pitch in the snare and hat bands, because the low band also finds snares: held to whole semitones, drum_pitch-2's snare came out at −1 instead of −0.70.

**What still fails.**
- Hat templates averaged over every hat carry the snare on beat 3: 13 of 26 hats found at the true times.
- Kits keep near-copies. An open hat matched another kit's file at +0.47 semitones (NCC 0.92) better than its own (0.84). The two agree over 80 ms (0.91) but not over 150 ms (0.60).
- Rims are partly tonal: 4 of 10 against 8 of 10 by whole semitones. The rim's template held the song's hat, which the fine search then matched at the right pitch.

## The margin, not the score

The calibration ([03](03-calibration.md)) shows that the score alone cannot decide:
- a wrong hat match reached 0.82 (it had found the snare that always played with it);
- right kicks went as low as 0.84.

What separates right from wrong for noisy sounds is the **margin**: the top match's NCC minus the best NCC of a *different* sound. That means removing copies of the top match first: kits copy each other's files under new names, and 95 % correlation between two library entries is treated as one recording.
- Right noisy matches led by 0.02–0.87, most by more than 0.1.
- Tonal sounds never lead by much: kicks 0.00–0.07, 808s 0.03–0.08.
- A search at any pitch finds more near neighbours. PIRAMIT's hat still matches its file at 0.984 and wins 310 of its 327 hits, but another kit's hat 3 semitones up reaches 0.958. The margin fell from 0.078 to 0.025, and the verdict from "same sample" to "same family".

### Copies at another pitch

`distinct` compares two library entries, each played at the transposition it was found at. A kit's copy 0.22 semitones down tied the original at NCC 0.84 and left the verdict no margin. Two entries found at different pitches count as one recording only when they also agree above 0.95 in the snare band.

In v1orig-6, a wrong kick 2 semitones down agreed with the true kick at 0.952 over the full band, so the true kick was dropped as its copy. The snare band gave 0.83, and the hat band 0.11. A kit's copy a semitone up agreed in both bands (0.997, 0.992).

The 20 beats' 291 rankings (97 templates, 3 bands) were checked with the full band alone and then with the snare band added. 4 top-five lists changed. Truth files in the top five went from 208 to 209, and none was lost.

## More evidence for a candidate

**Per-hit votes.** Each hit on its own is compared with the top three candidates over its first 70 ms. A real match wins nearly every hit, not just the average:

| PIRAMIT sound | wins | before replica mode |
|---|---|---|
| hat | 310 of 327 | 187 of 189 |
| the −1 snare (odd bars) | 28 of 28 | 27 of 27 |
| the same snare file at its own pitch (bars 24–31) | 30 of 30 | – |
| the unmatched snare (even bars, NCC 0.47) | 26 of 27 | 25 of 26 |

Votes only rank the top three, so a template far from all of them still has a winner: the unmatched snare's top candidate wins its votes at NCC 0.47. Before replica mode this page said its votes split; they did not. The null test is what says no.

**Null test.** Subtract the candidate (aligned, best gain) and measure what is left, in dB. It is the "same file" test with the mix's leftovers in it:
- the hat leaves −14.9 dB, the next candidate −10.9 dB (−14.3 and −7.4 before replica mode);
- the snare at its own pitch leaves −15.7 dB, the next −11.0;
- the −1 snare leaves −8.3 dB, the next −7.1 (−10.4 and −6.4 before);
- the unmatched snare leaves −1.1 dB.

**Shape of the transposition curve.** A noisy sound matched at the right transposition falls off within a few cents. A broad maximum means a tone, not a sample.

## Verdicts

`library.verdict` turns the NCC, the margin and the share of hits won into words a person (or a model) acts on; the null test is reported beside the verdict, not used by it:
- **same sample**
- **candidate: maybe the same sample, processed (check by ear)**
- **same family**
- **no close match**

The rules are in [01](01-pipeline.md#match). They come from 30 calibrated sounds, so they are conservative:
- In the last calibration run, all 15 blind templates of noisy sounds that met the "same sample" rule were right, and no oracle case broke it.
- A "candidate" is sometimes wrong.

On the suite's 8 identification scenarios (20 beats), the templates whose top match is their truth sound's file went from 49 of 78 (m1) to 75 of 95 (m2g).

## Several sounds in one template

A template is the average of hits on one grid position. If a hat plays on every kick, the template holds both: in the hat band it looks like the hat, in the low band like the kick. `match` scores every template in every band and hands out identities strongest first:
- the hat goes to the template where it is strongest (0.98);
- the kick template, whose hat-band match is that same hat at 0.80, takes its best match in another band instead.

A sample matched at ≥ 0.9 by two templates is one sound found twice (`same_as`). Found at another pitch, it is the same file played there: PIRAMIT's snare file plays at its own pitch in bars 24–31 and a semitone down on beat 3 of the odd bars, and the plan gives each its own pad. Before replica mode PIRAMIT's hat was found three times. Now the joint fit hands the hits of a hat template found in the snare band to the main hat (a merge), so the hat plays once ([01](01-pipeline.md#hits-detectdetect-layersexplain)).

**Layers.** A clap on every snare is in the snare's average too. `match` fits the template's top match together with each of 60 candidates: the template's 30 best, and the 30 best for what the top match leaves. Both gains must be positive. The best partner is the layer when:
- it takes the residual down by 3 dB or more;
- it is no sound another template was given, and no copy of one;
- it does not correlate with the top match (≥ 0.5 means the same sound again, at another pitch);
- it is at most 10 dB louder than the top match;
- it is not heard at more moments of the drum stem away from the template's hits than at them. A hat on every snare is a sound of its own, even when no template was given it.

`matches.json` gives the layer under `layers` (path, transposition, `gain_db` against the top match, `lag_ms`) and the evidence under `layer_search`. Kick-band templates are not searched, because any sine-like file takes energy off a kick. The plan does not play layers yet.

| beats (hits at the true times) | layered | both sounds found | plain templates given a layer |
|---|---|---|---|
| 12 snares with a clap at a fixed offset, 12 plain (v1 recipe) | 12 | 10 | 0 of 12 snares |
| calibration v2 `layers` (clap moved 0–8 ms per hit) | 2 | 0 | – |

A clap that moves from hit to hit is smeared in the average: the pair search takes only 1.3–1.5 dB off. Finding it needs the hits lined up on the residual, one by one. The 3 dB rule was chosen on the same 24 fixed-offset cases, so 10 of 12 is optimistic.

**Kicks under an 808.** Demucs puts part of a kick's body in the bass stem. With `match --kick-sources`, a template given the kick band is also read from drums + bass, and from the mix the stems came from, with the 808 taken out: the analysis's 808 notes played by fit-808's one-shot, lined up within 5 ms and scaled by least squares. The source whose best match is closest wins (`source` and `kick_source` in matches.json).

| 808 taken out with | kicks found (20 beats, kick band, true kick times) |
|---|---|
| nothing (drum stem only) | 12 |
| fit-808's synth one-shot | 13 |
| fit-808's 808 template | 12 |
| the true 808 file at the true notes (an oracle) | 17 |

The rule is only as good as the 808 it takes out, and it is off by default. match runs before fit-808, so a full run has no one-shot to take out. A run that reused fit-808's one-shot switched 2 of its 26 kick templates, and neither switch found the right file.

## Things that broke identification here, in order

1. **The lag sign.** `ncc` returns lag d when t is c delayed by d, and `shift(c, d)` lines c up with t. Averaging with the opposite sign blurred every template. Now there is a test.
2. **Different onset definitions.** Templates were aligned to onset-strength peaks, library files to their first rise above −30 dB. Some samples have a quiet lead-in, so the two sat 12–21 ms apart, outside a ±8–10 ms lag window. The true snare scored 0.15 instead of 0.85. Now: templates are aligned on their attack and the lag window is ±30 ms.
3. **Long windows.** A 300 ms snare template held the louder kick two sixteenths later, and that kick decided the match. Now: 150 ms for snares, 80 ms for hats, and only the attack (40–80 ms) for detection.
4. **Transposed samples never tried.** Only the top 200 at 0 semitones were tried at other pitches, and a pitched snare is not among them. Now: the semitone-band shortlist.
5. **Whole semitones.** A snare 0.70 semitones down was found at −1 and played 30 cents off. Now: the speckle, and Detune per pad.
6. **A copy at another pitch.** A kit's copy 0.22 semitones down tied the original, and a wrong kick 2 semitones down hid the true one as its "copy". Now: copies are compared as played, and at different pitches also in the snare band.
7. **A role from the kick band.** v1orig-3's open hat was named fx from a kick-band match at 0.957; its attack was then read on the next kick, 220 ms later, and the beat's analysis F1 fell from 0.789 to 0.578. Now a close match names the role only in the snare and hat bands.
8. **A rule keyed on where a template was found.** match kept whole semitones for templates found in the low band, which had been kicks. The attack-based discovery finds snares there too, and drum_pitch-2's snare came out 30 cents off. Now only the kick band keeps whole semitones.
9. **A silent file.** The index normalized every file to full scale and left out only all-zero ones, and NCC is scale-free. A float WAV of the library holding a recording about 430 dB down matched v1orig-3's hat template at 0.961 ("same sample"), and its pad played the beat's 144 hats as silence (the hat −455 dB against a perfect kit). It was the top match of nine other templates per model too, and 4 of m2g-vel's 199 pads played it. Now a file whose indexed start peaks under −60 dBFS is never a match; an index built before that floor reads each listed entry's file once to check. The hat pad plays its true file (0.940, in the hat band), at +3.5 dB against the kit.
10. **A tied band.** A kick's click is all the snare and hat bands hold of it, and every file that starts at once matches that. full-1's kick met 20 kick, 808 and synth-bass files within 0.0002 of NCC 0.9796 in the hat band, so a synth-bass one-shot played its kicks and the beat's plan levels went 2.60 → 6.88 dB from a perfect kit. Now a band whose five best different sounds lie within 0.002 NCC names none of them, and the template is matched in its own band, where its file is first at 0.960, 0.116 over the next sound; the beat's plan levels are 2.92 dB from the kit. On the suite the rule moves 4 templates on htdemucs and 2 on htdemucs_ft, all kicks.

What still plays a file of another role, at m3 (not judged by ear): dynamics-2's kick pad plays a snare file (NCC 0.813), melodic-1's snare pad a hat file (0.928), tempo160-1's snare pads a hat file (0.696), and on htdemucs_ft rolls-1's hat template, matched as a kick, plays the truth's hat file on kick pads (plan kick F1@20 0.466 there, as at m2g-ft-vel). Library files over full scale are played unscaled: 1 % of the entries peak over +6.4 dBFS, the loudest +19.4.
