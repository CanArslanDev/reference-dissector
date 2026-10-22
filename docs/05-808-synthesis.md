# 05 – Synthesizing the 808

**Why synthesize it.**
- An 808 is the one sound in trap you can rebuild from first principles: a sine, a pitch drop at the start, a decay, some saturation.
- Its waveform cannot single out one file anyway: all 808s at one pitch correlate 0.9+ ([03](03-calibration.md)).
- So the question becomes *can we make one that sounds the same*.

## The voice

```
f(t)   = f_end + (f_start − f_end) · exp(−t / glide)
amp(t) = (1 − exp(−t / attack)) · exp(−t / decay)
y(t)   = tanh(drive · (amp(t) · sin(phase + 2π∫f) + asym)) − tanh(drive · asym)
```

Eight numbers, each a knob on Operator/Simpler + Saturator:

| parameter | knob |
|---|---|
| pitch | coarse/fine tune (Simpler: Transpose, Detune) |
| glide | pitch envelope amount and decay |
| attack, decay | amp envelope |
| drive, asym | Saturator drive, DC offset or bias |

The search ranges are fixed (`synth808.BOUNDS`): f_start 25–120 Hz, glide 2–200 ms, attack 0.3–200 ms, decay 0.1–6 s, drive 0.1–8. `synth808.one_shot` renders the voice at any pitch as a sample for Simpler.

## The reference

**Building it** (`fit_808`):
1. **Pick notes.** The 808 template's notes (`bass808.json`): the long notes of its most common waveform, learnt from those that start on a kick first ([01](01-pipeline.md#808-notes-bass808transcribe)). PIRAMIT has 53 such C1 notes, all on a kick. Before replica mode the reference was the notes of the commonest pitch that rang at least 600 ms before the next note (74 on PIRAMIT).
2. **Average** those notes from drums + bass, lined up on their waveform.
3. **Why drums + bass.** Demucs puts part of the 808's attack in the drum stem. Fitted on the bass stem alone, the voice came out with a 190 ms attack: a soft swell that is not what the track does.

**What a kick on every note does.** The drum stem also brings the kick into the average. On PIRAMIT the fit now starts the voice at 120.0 Hz, 22.3 semitones over its end, falling with a 23 ms time constant, with a 0.3 ms attack. Both sit on the search's bounds (f_start's top, attack's bottom). Before replica mode, on the 74 notes, it started at 41.25 Hz (+3.9 semitones) with a 60 ms glide and a 76 ms attack. That is the shape of a kick's sweep. Whether the kicks in the reference cause it is not separated, and the one-shot has not been judged by ear ([02](02-case-study-piramit.md#808)).

## You get what you measure

The same reference fitted with two different scores (PIRAMIT, bass-stem reference, before replica mode):

| score used for the fit | waveform NCC | harmonics 2–5 off | harmonics 6–12 off | envelope off |
|---|---|---|---|---|
| waveform NCC only | **0.969** | 24.7 dB | 7.6 dB | 1.6 dB |
| NCC + 0.1·harmonics + 0.1·envelope | 0.967 | **4.5 dB** | **6.5 dB** | **0.9 dB** |

**What happened.** Fitted on correlation alone, the optimizer set the drive to its minimum. The fundamental carries nearly all the energy below 1.5 kHz, so the harmonics hardly move the NCC; the result is a clean sine. By correlation it was the best 808 found; by ear it would have no grit at all. Adding the harmonic levels to the score brought the grit back at almost no cost in correlation.

**The general lesson.** A single similarity number rewards whatever dominates it. Score a replacement on everything the ear uses: pitch, glide, envelope, harmonics, brightness. Look at the spectrogram as well.

## Against the library

The fitted voice against the best of 865 library 808s, tried at ±12 semitones. Each run is compared with its own reference, so the two runs are not comparable cell by cell:

| | synth | best library 808 | synth, before replica mode | library, before |
|---|---|---|---|---|
| reference | 53 template notes | | 74 isolated notes | |
| waveform NCC | **0.938** | 0.894 | **0.975** | 0.947 |
| harmonics 2–5 | 6.5 dB | 6.5 dB | **7.4 dB** | 7.8 dB |
| harmonics 6–12 | 12.9 dB | **8.7 dB** | 7.7 dB | **2.3 dB** |
| envelope | **1.3 dB** | 1.4 dB | **0.9 dB** | 2.0 dB |
| log-mel distance | 19.3 dB | **11.6 dB** | 22.3 dB | **9.7 dB** |

**Reading it.**
- **The synth** has the pitch curve and the envelope; tanh drive at that level makes too few upper harmonics (6–12 off by 12.9 dB, the library 808 by 8.7).
- **The library 808** has the texture: it is a distorted 808, as the reference is. No single file stands out (margin 0.001).
- **The picture** (README figure, bottom row) shows the same: rich harmonics up to 400 Hz in the reference and the library sample, fewer in the synth.
- **Making the synth match** would need either a stronger, asymmetric drive term in the model, or the harmonics up to the 12th weighted more in the fit.

## In Live

The one-shot is rendered at the template's settled pitch, 3 s long, at −1 dBFS: PIRAMIT's is C1 + 11.8 cents, 32.93 Hz. The fit's own `f_end` is used instead only where the two agree within 2 cents, or where the template had only an unsettled tail to read (PIRAMIT's is 6.5 cents off, so it is not used). Before replica mode the one-shot sat at C1 + 33 cents (33.33 Hz), the pYIN reading that heard the 808's drop. It goes into Simpler:
- **Transpose and Detune.** Simpler plays a sample at its own pitch on C3. For a one-shot that settles at fractional MIDI r and an 808 tuned c cents, MIDI note m sounds at m + c/100 with Transpose = round(60 − r + c/100) and Detune = 100 · (60 − r + c/100 − Transpose). Live 12.4.6 keeps whole semitones of Transpose (11.99 plays as 11) and whole cents of Detune (±50), so both are whole numbers in the plan (`live.tune_808`). PIRAMIT: Transpose +36, Detune +2.
- **Heard pitch.** The one-shot falls as one exponential, faster than the 808s it copies after about 100 ms. Tuned to settle where the 808 settles, it sounds flat over the notes. Detune therefore adds `heard_cents`: over the notes at the line's most common written pitch, the median of the drop the analysis heard less the one-shot's own. Re-scored on the same plans, plan note F1@50 went 0.849 → 0.886 on the 10 suite scenarios with an 808 (bass_low-1 0.333 → 0.949). On PIRAMIT it is +1.6 cents. The tails of long notes then settle that much sharp.
- **Notes** are the written pitches at their own lengths. Each starts so that the one-shot's attack, played at the note's pitch, lands on the note's attack. One voice: each new note cuts the last.
- **Slides.** By default an 808 line has no legato notes and the Simpler no Glide: the reader's slide guesses were right 6 times in 79 ([08](08-pitfalls.md)). With slides on, three legato notes or more put the Simpler on one voice, with Glide Mode "Glide" and the song's median Glide Time. Live 12.4.6 then slides overlapping notes linearly in semitones over exactly the Glide Time. A legato chain held past the end of the 3 s one-shot is silent from there.
- **Warping off.** Pitch is changed by speed, as an 808 should be.
- **Volume set to 0 dB.** Simpler loads at −12 dB.
- **Levels.** On the suite (m2g) the plan's 808 sits 1.95 dB under a perfect kit (+0.09 dB in m1): faders matched to the bass stem's whole-track RMS now see the notes' real lengths.

The build sends Transpose and Detune, and with Glide on also Glide Mode, Glide Time and one voice. None of it has been built and recorded in Live yet: not verified ([07](07-building-in-live.md)).
