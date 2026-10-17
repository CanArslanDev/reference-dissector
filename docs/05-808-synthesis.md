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
| pitch | coarse/fine tune |
| glide | pitch envelope amount and decay |
| attack, decay | amp envelope |
| drive, asym | Saturator drive, DC offset or bias |

`synth808.one_shot` renders it at any pitch as a sample for Simpler.

## The reference

**Building it** (`fit_808`):
1. **Pick notes.** The 808 notes of the commonest pitch that ring at least 600 ms before the next note.
2. **Average** those notes from drums + bass, lined up on their waveform. PIRAMIT has 73 such C1 notes.
3. **Why drums + bass.** Demucs puts part of the 808's attack in the drum stem. Fitted on the bass stem alone, the voice came out with a 190 ms attack: a soft swell that is not what the track does.

## You get what you measure

The same reference fitted with two different scores (PIRAMIT, bass-stem reference):

| score used for the fit | waveform NCC | harmonics 2–5 off | harmonics 6–12 off | envelope off |
|---|---|---|---|---|
| waveform NCC only | **0.969** | 24.7 dB | 7.6 dB | 1.6 dB |
| NCC + 0.1·harmonics + 0.1·envelope | 0.967 | **4.5 dB** | **6.5 dB** | **0.9 dB** |

**What happened.** Fitted on correlation alone, the optimizer set the drive to its minimum. The fundamental carries nearly all the energy below 1.5 kHz, so the harmonics hardly move the NCC; the result is a clean sine. By correlation it was the best 808 found; by ear it would have no grit at all. Adding the harmonic levels to the score brought the grit back at almost no cost in correlation.

**The general lesson.** A single similarity number rewards whatever dominates it. Score a replacement on everything the ear uses: pitch, glide, envelope, harmonics, brightness. Look at the spectrogram as well.

## Against the library

The fitted voice (drums + bass reference, 49 s fit) against the best of 865 library 808s, tried at ±12 semitones:

| | synth | best library 808 |
|---|---|---|
| waveform NCC | **0.975** | 0.947 |
| harmonics 2–5 | **7.4 dB** | 7.8 dB |
| harmonics 6–12 | 7.7 dB | **2.3 dB** |
| envelope | **0.9 dB** | 2.0 dB |
| log-mel distance | 22.3 dB | **9.7 dB** |

**Reading it.**
- **The synth** has the pitch curve and the envelope; tanh drive at that level makes too few upper harmonics.
- **The library 808** has the texture: it is a distorted 808, as the reference is.
- **The picture** (README figure, bottom row) shows the same: rich harmonics up to 400 Hz in the reference and the library sample, fewer in the synth.
- **Making the synth match** would need either a stronger, asymmetric drive term in the model, or the harmonics up to the 12th weighted more in the fit.

## In Live

The one-shot is rendered at the 808's own pitch, C1 + 33 cents = 33.33 Hz (33.24 Hz in the first build, at the track tuning then measured), 3 s long, −1 dBFS. It goes into Simpler:
- **Transpose = 60 − 24 = +36.** MIDI note 24 then plays it at its own pitch, so the transcribed notes keep their real numbers.
- **Warping off.** Pitch is changed by speed, as an 808 should be.
- **Volume set to 0 dB.** Simpler loads at −12 dB.

No two of the 126 notes overlap, so each new note cuts the last, as on a mono 808. The three slides are not played yet: Simpler's glide needs overlapping notes, or the slides would need pitch automation.
