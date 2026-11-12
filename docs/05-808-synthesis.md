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

The search ranges are fixed (`synth808.BOUNDS`): f_start 25–120 Hz, glide 2–200 ms, attack 0.3–200 ms, decay 0.1–6 s, drive 0.1–8. `synth808.one_shot` renders the voice at any pitch as a sample for Simpler. The one-shot the plan plays is `synth808.tracked_shot`'s: the voice on the template's own periods ([In Live](#in-live)). `fit-808 --grit` fits a grittier voice with two more numbers, off by default ([A grittier voice](#a-grittier-voice)).

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
- **Making the synth match** would need either a stronger, asymmetric drive term in the model, or the harmonics up to the 12th weighted more in the fit. Both are now in a grittier voice, behind `--grit` ([below](#a-grittier-voice)).
- **The one-shot the plan plays** (since m4, the voice on the template's periods, below) reaches NCC 0.880 against the same reference, with the same harmonics and envelope (6.5, 12.9 and 1.3 dB off) and a log-mel distance of 19.3 dB. The m3 code's one-shot, the voice at the template's pitch, had 0.921. Which of them sounds closer is not checked by ear.

## A grittier voice

`fit-808 --grit` (and `suite run --grit`; feat/808-grit, 2026-11-11). Off by default: it moves the plan's levels and some 808 numbers the wrong way (below).

```
osc(t) = sin φ + grit · Σ_{n=6..12} (6/n) · sin(nφ)
d(u)   = drive · (1 + skew · sign(u))
y(t)   = tanh(d(u) · u) − tanh(d(asym) · asym),   u = amp(t) · osc(t) + asym
```

Two more numbers, each a knob in Live:

| parameter | what it does | knob |
|---|---|---|
| skew (−0.8 to 0.8) | the drive's asymmetry: the positive half driven by drive·(1 + skew), the negative by drive·(1 − skew) | Saturator's waveshaper curve, or two drive stages |
| grit (0 to 0.1) | the upper-harmonic term: harmonics 6–12 added to the oscillator, the 6th at `grit` of the fundamental, each one after at grit·6/n | Operator's partials |

**Why these two.** On the suite the plain voice misses harmonics 6–12 mostly on seed 2's 808s, by 7–24 dB. Those 808s decay in 0.16–0.77 s, so tanh's harmonics fall with the level over the 0.15–0.6 s the harmonics are read on, while the template's stay at about −40 dB. A term added before the drive keeps its level against the fundamental as the note decays.

**How it is fitted.** On top of the plain fit (`synth808.gritted`): the plain search runs as before, then drive, asym, skew and grit are searched again (differential evolution, the same seed and size) with the pitch, the drop, the phase, the attack and the decay kept. The score is the plain one plus 0.1 × the mean distance of harmonics 6–12 again, so they weigh twice. A first try searched all ten numbers together. It reached the harmonics as well, but on v1-2 the drop moved from 120 Hz with a 17 ms glide to 39 Hz with 74 ms, and the attack from 2.5 to 101 ms. Kept separately, the fit takes 85.5 s a suite beat, against 46 s for the plain fit.

In `tracked_shot`, a gritty voice prefers the attack count whose first crossing the reader reads as it reads the template's. The upper harmonics pass the reader's low-pass in the attack. On the first run of the 808 lane (12 beats), 4 one-shots were read one crossing apart, and bass_low-2's line went 35 cents sharp (Detune −1 → +35). The plain voice keeps the old rule.

**Measured** (gr-1 and gr-1-ft against m5 and m5-ft: fit-808, plan and score over m5's other outputs, all 48 beats, `--render`, no cache):
- **Harmonics 6–12 within 3 dB of the template's:** on 45 of 48 beats (21 before), median 3.5 → 1.4 dB, mean 5.87 → 1.67; htdemucs_ft 44 of 48 (24), 3.05 → 1.4 dB. The tracked one-shot is the same: 45 (20) and 43 (23). The 3 left on htdemucs are v1orig-6 (5.7 → 6.5 dB), v1orig-2 (3.4, unchanged) and melodic-1 (3.5 → 3.3); on htdemucs_ft bass_low-2 (6.1 → 3.6), producer-1 (2.4 → 3.2), v1orig-4 (3.7 → 3.4) and v1orig-6 (6.4 → 6.8). The fit trades harmonics 6–12 against the rest of its score: on 5 beats they end 0.1–1.0 dB further off.
- **Also closer:** log-mel distance median 12.3 → 8.9 dB (11.3 → 8.8), envelope 3.1 → 2.8 dB (3.45 → 2.8), waveform NCC 0.849 → 0.857 (0.860 → 0.867). Harmonics 2–5 5.0 → 5.2 dB (5.2 → 4.9).
- **The glide shape is unchanged.** The fitted drop is the plain fit's by construction (f_start, glide and f_end are kept). The one-shot's pitch curve, as `scenarios.tuning` reads it from 50 to 300 ms, sits a median 0.1 cents from the plain one-shot's (median per beat on both models). The largest differences are 168 cents (132), at 50–70 ms on swing58-2 and swing66-2, where the reader reads the attack's first crossing differently. Against the template's own curve: 0.33 → 0.30 cents (0.23 → 0.27).
- **What it costs** (plan, means over the 22 scenarios; m5 → gr-1, m5-ft → gr-1-ft):

| | m5 | gr-1 | m5-ft | gr-1-ft |
|---|---|---|---|---|
| 808 note F1@50 | 0.9342 | 0.9342 | 0.9345 | **0.9342** |
| 808 median \|cents\| | 4.42 | 4.41 | 9.63 | 9.39 |
| 808 settled \|cents\| | 3.66 | **3.77** | 6.89 | **7.14** |
| 808 onset p90 | 7.979 ms | 7.977 | 7.0393 | **7.0395** |
| distance from a perfect kit | 1.198 dB | 1.173 | 1.152 | **1.193** |
| balance, median \|·\| | 0.512 dB | 0.491 | 0.476 | **0.551** |
| 808 against the kit | −1.87 dB | −1.62 | −2.01 | −1.75 |
| kick against the kit | −0.84 dB | **−0.89** | −0.04 | **−0.17** |
| kick balance | −0.81 dB | **−1.19** | +0.44 | +0.12 |
| snare against the kit | +0.04 dB | **+0.35** | −0.75 | −0.46 |

  Drum F1, tempo, meter and every analysis number are the same, as they should be. Bold marks the worse numbers.
  - The distance from the kit gets worse in 9 of 22 scenarios (13 on htdemucs_ft): tempo95 0.33 → 1.06 dB, swing66 1.14 → 1.88, tempo160 2.67 → 3.35, leadin 1.13 → 1.53.
  - The settled pitch moves in 5 scenarios on each model, by 0.5–1.0 cents.
  - htdemucs_ft's bass_low loses one note (F1@50 0.513 → 0.509).
- **Why the levels move.** The kick is read under the plan's own 808 ([10](10-roadmap.md#6-the-open-issues)), and the 808's fader follows the one-shot's level. A one-shot with more harmonics leaves a different residual under the kick, and the drum fader, matched by RMS, then lifts or lowers every pad. This is the cost the roadmap expected.

Not judged by ear. Live is not involved: the one-shot is still a sample, and Simpler plays it as before.

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
- **Slides.** By default an 808 line has no legato notes and the Simpler no Glide. With `--slides` the reader writes a legato note where the 808 band's pitch holds, moves straight to a whole transposition and holds again, with no restart ([01](01-pipeline.md#808-notes-bass808transcribe)): 8 of seed 1's 10 slides on each model, none wrong (in-sample), 4 of seed 2's 14 (htdemucs_ft 3), and false legato notes on 4 (htdemucs_ft 2) beats without a slide, so it stays off. The rule before it called every unattacked pitch move a slide and was right 6 times in 79 ([08](08-pitfalls.md)). Three legato notes or more put the Simpler on one voice, with Glide Mode "Glide" and the line's median Glide Time: glides-1's 162.5 ms lies a mean 46.6 ms from its truth's per-note glides (88–276 ms). Live 12.4.6 then slides overlapping notes linearly in semitones over exactly the Glide Time. A legato chain held past the end of the 3 s one-shot is silent from there; the longest suite chain reads 1.58 s of it.
- **Warping off.** Pitch is changed by speed, as an 808 should be.
- **Volume set to 0 dB.** Simpler loads at −12 dB. Unlike the drum pads, the 808's Simpler always plays at 0 dB; its level is the track's fader.
- **Levels.** Velocities follow Live 12.4.6's law at "Vol < Vel" 35 %, which the build sets: 115 at the line's median peak, 0.14 dB a step from there, within 70–127. On the suite the plan's 808 sits 1.87 dB under a perfect kit (m4 and m5; htdemucs_ft 2.01; 1.98 and 2.24 at m3, +0.09 dB in m1): faders matched to the bass stem's whole-track RMS see the notes' real lengths. The 808's fader follows the new one-shot's level: against m3 it moved on 9 beats (htdemucs_ft 20), by at most 0.1 dB (0.3).

The build sends Transpose and Detune, Snap and the filter off, Live's own 50 ms release, and with Glide on also Glide Mode, Glide Time and one voice. In Live (two checks, 2026-10-29, and an excerpt with Glide, 2026-11-05; [07](07-building-in-live.md#live-behaviour-measured-here)):
- the one-shot on its template's periods played at `render.py`'s pitch within 0.11 cents, at Detune −8, −1 and +1 (the first check, on m4 plans);
- with its filter on, Simpler delayed each note 0.13–0.46 ms by its pitch and took 2.2 dB off its 32nd harmonic, so the build turns the filter off;
- a note ends in Live's release (1.683 dB a ms, silent 35.7 ms on), and one that comes while the voice of its pitch still sounds restarts it 1 ms late: both are in `render.py` now;
- on four m5 plans and PIRAMIT's, 408 notes then started within 0.11 ms of `render.py`'s, at −43.1 to −50.8 dB of residual (medians per beat);
- Glide, on a probe (within 1.04 cents, p90, of `render.py`) and on an 8-bar excerpt of glides-1's `--slides` plan: its 4 slides within 1.21 cents (p90) of `render.py`'s, its 808 31.5 dB of residual under the render. By default no plan turns Glide on.
