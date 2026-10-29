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

The search ranges are fixed (`synth808.BOUNDS`): f_start 25–120 Hz, glide 2–200 ms, attack 0.3–200 ms, decay 0.1–6 s, drive 0.1–8. `synth808.one_shot` renders the voice at any pitch as a sample for Simpler. The one-shot the plan plays is `synth808.tracked_shot`'s: the voice on the template's own periods ([In Live](#in-live)).

## The reference

**Building it** (`fit_808`):
1. **Pick notes.** The 808 template's notes (`bass808.json`): the long notes of its most common waveform, learnt from those that start on a kick first ([01](01-pipeline.md#808-notes-bass808transcribe)). PIRAMIT has 53 such C1 notes, all on a kick. Before replica mode the reference was the notes of the commonest pitch that rang at least 600 ms before the next note (74 on PIRAMIT).
2. **Average** those notes from drums + bass, lined up on their waveform.
3. **Why drums + bass.** Demucs puts part of the 808's attack in the drum stem. Fitted on the bass stem alone, the voice came out with a 190 ms attack: a soft swell that is not what the track does.

**The search.** Differential evolution, popsize 20, up to 150 generations, seeded: about 24,000 candidates a fit. The band filter, the Hann window and the time axis each candidate needs are made once per role or length (`butter()` took 127.7 µs a call, now 0.2 µs), with every output the same to the bit. fit-808 takes 46.2 s a suite beat at 2 jobs (m3a; 50.9 s at m2g); the fit took 44 s on PIRAMIT, one step at a time (e2a5ab4; 50 s on 2026-10-22).

**Kept fits.** `synth808.fit` keeps a fit only when given a folder. The `fit-808` command gives it the work folder's `cache/` (`--no-cache`: none), and `suite run` gives each beat's `cache/` only with `--cache`. The key is a hash of the template's samples, f1, the settings (seed, maxiter, popsize, waveform_only), numpy's, scipy's and soxr's versions, and every `.py` file of the package as the process loaded it. A fit with the same key is read back, since the seeded search would find the same parameters again; the log says so, and `synth808.json`'s `fit.cached` names the file. Read back on 6 suite beats, fit-808 took 1.0 s a beat against 44.7 s run fresh, with the same parameters and one-shot.
- The first key read three files from disk when a fit was kept. A process that imported synth808 before an edit to synth808.py filed its fit under the edited code, and a process of the edited code read back f_start 93.3 Hz where its own search finds 53.2 Hz. Now nothing is read back or kept while a file of the package differs from what the process loaded.
- Code patched in memory is not seen by the key, so a script that patches the fit (a bound, say) must not pass a folder. With the drive's bound patched to 2–8, a fit read back had a drive of 0.91, outside it; fitted again, 4.53.

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
- **The one-shot the plan plays** (since m4, the voice on the template's periods, below) reaches NCC 0.880 against the same reference, with the same harmonics and envelope (6.5, 12.9 and 1.3 dB off) and a log-mel distance of 19.3 dB. The m3 code's one-shot, the voice at the template's pitch, had 0.921. Which of them sounds closer is not checked by ear.

## In Live

The one-shot settles at the template's settled pitch, 3 s long, at −1 dBFS: PIRAMIT's is C1 + 11.8 cents, 32.93 Hz. Where the template's own notes never settle, that pitch is read on the notes that reach past the template (`settle()`, [01](01-pipeline.md#808-notes-bass808transcribe)): bass_low-1's went from 59.7 to 14.0 cents over the truth. The fit's own `f_end` is used instead only where the two agree within 2 cents, or where the template had only an unsettled tail to read (PIRAMIT's is 6.5 cents off, so it is not used). Before replica mode the one-shot sat at C1 + 33 cents (33.33 Hz), the pYIN reading that heard the 808's drop.

**On the template's own periods** (`synth808.tracked_shot`, since m4). The fitted voice falls as one exponential: on PIRAMIT 22.3 semitones with a 23 ms time constant, on bass_low-1 25.9 semitones with 19 ms. The 808s it copies fall more slowly: bass_low-1's holds about 100 cents over its settled pitch until about 200 ms. So the one-shot the plan plays keeps the voice's envelope, drive and attack, and crosses zero where the template does:
- The crossings are found back from the template's pitch curve (`bass808.json`) by least squares and followed until the drop has settled, within 5 cents of the settled pitch for 100 ms. Then the pitch eases to the settled pitch over 50 ms.
- Up to three rounds move the voice's crossings until the reader's (`scenarios.tuning`) land on the template's: the reader's low-pass and the drive's DC move them.
- Before the first crossing followed, the one-shot plays the fitted drop, scaled to end a whole cycle there; its pitch steps by that scale at that crossing (0.94 on PIRAMIT).
- An 808's own drop only falls. Where the template's curve rises 100 cents or more over its lowest point (on the suite, where a kick plays in the stem), its periods up to the rise are pooled into periods that never shorten, so the cycles and every crossing past the rise stay the template's.
- **Laid out as the template is.** The one-shot starts at the template's onset and attacks (808 band) where the template attacks after it (`bass808.json`'s `attack_ms`). A voice that attacks later starts up to 50 ms before the one-shot, its head left out and faded in over 0.5 ms; one that attacks earlier plays at 20 % of its level until a 2 ms rise puts its attack there.
- Where the curve holds fewer than three crossings, has no drop left to follow, or the voice's crossings cannot be put within 0.5 ms of the template's, the one-shot is the fitted voice, as before.
- On PIRAMIT it follows 9 of 49 periods, to 269 ms, where the drop has settled, and attacks 5.24 ms in (the template 5.44 ms after its onset). Played attack on attack, its pitch sits a median 1.5, 0.2 and 0.1 cents from the template's over 50–100, 100–150 and 150–300 ms; the m3 code's one-shot, the fitted voice, sat 65, 38 and 10 cents from it ([02](02-case-study-piramit.md#808)).
- Not judged by ear: the pitch step at the first crossing followed, the quiet head of a one-shot whose voice attacks early, and the cut head of one whose voice attacks late.

It goes into Simpler:
- **Transpose and Detune.** Simpler plays a sample at its own pitch on C3. For a one-shot that settles at fractional MIDI r and an 808 tuned c cents, MIDI note m sounds at m + c/100 with Transpose = round(60 − r + c/100) and Detune = 100 · (60 − r + c/100 − Transpose). Live 12.4.6 keeps whole semitones of Transpose (11.99 plays as 11) and whole cents of Detune (±50), so both are whole numbers in the plan (`live.tune_808`). PIRAMIT: Transpose +36, Detune +1.
- **Heard pitch.** Tuned to settle where the 808 settles, a one-shot that falls faster than the 808 sounds flat over the notes. Detune therefore adds `heard_cents` (`live.heard_808`): over the notes at the line's most common written pitch, the median of the drop the analysis heard less the one-shot's own, each note heard from where the plan starts it to its end. Against the rule it replaced, on the same plans of the 808 lane's 10 scenarios (bass_low, glides, kick808, v1, v1orig, full, producer, melodic, tempo72, tempo160; 24 beats), plan note F1@50 went 0.849 → 0.886 (bass_low-1 0.333 → 0.949). The tails of long notes then settle that much sharp.
- **One Detune for notes of every length** was m3's regression. With the fitted voice as the one-shot, the correction depended on a note's length: bass_low-1's notes of 120–600 ms played 42–52 cents flat while their tails settled 62 cents sharp, its median error 41.7 cents and plan note F1@50 0.718 (17.5 cents and 0.949 at m2g-vel, where two errors cancelled: the template 60 cents sharp, every note written a semitone high). With the one-shot on the template's periods (m4):
  - bass_low-1's correction by note length (under 120, 120–250 and 250–600 ms) is −0.1, −0.19 and −0.11 cents (htdemucs_ft +3.76, −0.08, −0.17); its plan note F1@50 is 1.0 (0.981), its median error 15.1 cents (9.3);
  - over the suite the line's correction lies within 8 cents of 0 on 48 of 48 beats (47 of 48; 37 and 43 at m3); plan note F1@50 0.921 → 0.934 (0.910 → 0.934), median error 8.3 → 4.5 cents (13.6 → 9.6);
  - on PIRAMIT the correction for the C1 notes of 120–250, 250–600 and over 600 ms went +41.4, +4.8 and +1.6 → −0.1, 0.0 and +0.8 cents; `heard_cents` is +0.8 (+1.6), so Detune +1 (+2).
- **Open: a note's own cents.** One Detune per Simpler plays every note at the line's tuning. On full the analysis writes one note a semitone high and hears 5 short notes 41 cents low; m3's one-shot, falling too fast, happened to offset both, and full's plan note F1@50 fell 0.890 → 0.851 (htdemucs_ft 0.910 → 0.890). producer's plan median error rose 6.8 → 8.2 cents (15.2 → 21.1), with its note F1@50 up, 0.770 → 0.804 (0.698 → 0.770).
- **Open: the yardstick.** The drop is read (`scenarios.tuning`) over less than one period of a 110 ms note under 30 Hz, so the reading depends on phase: 89 and 160 cents on two near-identical fitted one-shots, where the voice's own formula gives 111 and 108.
- **Notes** are the written pitches at their own lengths. Each starts so that the one-shot's attack, played at the note's pitch, lands on the note's attack; with the one-shot laid out as its template, a note starts at its onset (PIRAMIT: a median 0.2 ms after the analysis' onset, 5.3 ms earlier than with the m3 code). One voice: each new note cuts the last, and one that comes while the voice of its pitch still sounds starts 1 ms early, as Live starts it 1 ms late (m5).
- **Slides.** By default an 808 line has no legato notes and the Simpler no Glide. The rule this replaced called every unattacked pitch move a slide and was right 6 times in 79 ([08](08-pitfalls.md)); the opt-in rule (`slides=True`) made 18 legato notes, 6 of them right, on the 24 beats its thresholds were chosen on. With slides on, three legato notes or more put the Simpler on one voice, with Glide Mode "Glide" and the song's median Glide Time. Live 12.4.6 then slides overlapping notes linearly in semitones over exactly the Glide Time. A legato chain held past the end of the 3 s one-shot is silent from there.
- **Warping off.** Pitch is changed by speed, as an 808 should be.
- **Volume set to 0 dB.** Simpler loads at −12 dB. Unlike the drum pads, the 808's Simpler always plays at 0 dB; its level is the track's fader.
- **Levels.** Velocities follow Live 12.4.6's law at "Vol < Vel" 35 %, which the build sets: 115 at the line's median peak, 0.14 dB a step from there, within 70–127. On the suite the plan's 808 sits 1.87 dB under a perfect kit (m4 and m5; htdemucs_ft 2.01; 1.98 and 2.24 at m3, +0.09 dB in m1): faders matched to the bass stem's whole-track RMS see the notes' real lengths. The 808's fader follows the new one-shot's level: against m3 it moved on 9 beats (htdemucs_ft 20), by at most 0.1 dB (0.3).

The build sends Transpose and Detune, Snap and the filter off, Live's own 50 ms release, and with Glide on also Glide Mode, Glide Time and one voice. In Live (two checks, 2026-10-29, [07](07-building-in-live.md#live-behaviour-measured-here)):
- the one-shot on its template's periods played at `render.py`'s pitch within 0.11 cents, at Detune −8, −1 and +1 (the first check, on m4 plans);
- with its filter on, Simpler delayed each note 0.13–0.46 ms by its pitch and took 2.2 dB off its 32nd harmonic, so the build turns the filter off;
- a note ends in Live's release (1.683 dB a ms, silent 35.7 ms on), and one that comes while the voice of its pitch still sounds restarts it 1 ms late: both are in `render.py` now;
- on four m5 plans and PIRAMIT's, 408 notes then started within 0.11 ms of `render.py`'s, at −43.1 to −50.8 dB of residual (medians per beat);
- Glide was measured on a probe only (within 1.04 cents, p90, of `render.py`): no plan turns it on.
