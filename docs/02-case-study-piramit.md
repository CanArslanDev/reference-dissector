# 02 – Case study: Mavi – *PIRAMIT*

A Turkish trap track (album *VERITAS II*), 2:06.
- **Goal.** Take it apart and build a near replica in Ableton Live from a personal sample library and Live's own devices.
- **Starting point.** The Demucs stems and a prototype whose beat tracker gave 68.18 BPM.
- **Where the numbers come from.** The run of 2026-10-22 with replica mode: `analyze`, `melodic`, `match`, `fit-808` and `plan` with the code of 9ccc759 (the code the suite's m2g run measured), on the same stems. `results/piramit/summary.json` holds its aggregates; counts per bar and the measurements on the stems come from the same run's local files. Numbers marked *before replica mode* come from the run of 2026-10-17 (42cb7d9).
- **No truth.** Nobody has transcribed PIRAMIT by hand. Where the two runs differ, this page says which one is right only where a measurement on the stems shows it; everything else is not verified.
- **What is not here.** The song, its stems and the note-by-note transcription stay local: see [09](09-data-and-copyright.md).

## Grid: 137.00 BPM, not 136.36

**What the prototype assumed.** The beat tracker (`beat_this`) reported 68.18 BPM, half time, so the prototype assumed 136.36.

**The fitted grid.** Before replica mode, a comb fit alone: 137.002 BPM with the first beat at 31.9 ms. Now `beat_this` 1.1.0 reads the sum of the stems first:
- its beats lie on a line at 68.498 BPM (all 144 of them), and 100–180 BPM takes the octave above: 136.999 BPM;
- the comb's own best phase at that tempo is where the tracker's beats sit (their resultant 0.99; moved 0.0 ms);
- the first bar line is at 23.4 ms, moved 7.1 ms back from the onset-strength peaks onto the attacks;
- 4/4, from the tracker's downbeats (35 of 36 bars); 36 of its 37 downbeats fall on the bar line;
- 8ths straight (ratio 0.50), no 16th swing read; the beats drift 0.7 ms from a constant tempo, so no tempo map is kept.

Strong snare-band onsets per beat of the bar sit within these offsets of the grid line (on the onset-strength time base):

| | bars with an onset | median | p90 of \|offset\| |
|---|---|---|---|
| beat 1 | 60 | +0.3 ms | 10.2 ms |
| beat 2 | 60 | +1.2 ms | 8.2 ms |
| beat 3 | 58 | +1.7 ms | 4.7 ms |
| beat 4 | 62 | −0.4 ms | 11.3 ms |

Before replica mode the medians were within 0.4 ms of these and the p90s within 0.6 ms.

**At 136.36 BPM**, from the same first bar line, the beat-3 onsets sit 2.0 ms early in bar 1 and 49.5 ms early in bar 7: 8.0 ms more each bar. `grid.offsets` looks within 50 ms of each grid line, so from bar 8 on it loses them. By the end of the song the grid is 0.59 s off, more than a beat (computed from the two tempi). This page said before that the onsets "start at +29 ms and end at −49 ms"; measured again on 2026-10-22, −49 ms is where they leave that window, in bar 7, not the end of the song. An average inter-beat interval hides a drift that a fitted grid shows at once: a line through the same tracker's beats gives 68.50 BPM, not 68.18.

**Lesson.** Fit the grid; then check that the *last* bars still land on it.

## Tuning: the track at A440, the 808 at +11.8 cents

| stem | deviation from A440 (stable spectral peaks) | first version (`librosa.estimate_tuning`) |
|---|---|---|
| bass | +11 cents | +23 |
| other | −2 | +27 |
| piano | +1.5 | +28 |
| vocals | −38 | +32 |
| **808** | **+11.8** (its template's settled pitch, 32.93 Hz) | – |
| 808, before replica mode (circular mean of 126 pitches) | +33 (agreement 0.64) | – |

**Correction 1.** This section first said the whole track sat at +0.28 semitones (A4 ≈ 447 Hz). That came from `librosa.estimate_tuning`, which is off by up to 60 cents on synthetic tones. Measured with stable spectral peaks (within 1 cent on synthetic tones) and checked with pyin (+2 to +3 cents on piano and other), the melodic stems are at A440.

**Correction 2.** The 808 is detuned, but by +11.8 cents, not +33. The +33 was the circular mean of each note's pYIN median over 60–450 ms, and that window hears the 808's drop. The 808 reader now learns the song's own 808 from its long notes (53 notes, all starting on a kick) and reads its settled pitch by QIFFT over 300–600 ms, where the window's two halves agree within 3 cents. Each note is a transposition of it; the notes' settled pitches agree on +10.7 cents (0.868).

**What it explains.** The prototype's 808 notes split between C (90) and C# (82), and its key estimate wavered between C minor and C major. Notes at C + 30 cents round either way. Before replica mode, with the +33 cents removed before rounding, 107 of 126 notes were C1. Now 79 of 94 are.

**What it is not.** It could have been a whole track sped up: 1.5 % faster would also turn 135 BPM into 137. Matching every drum sample shifted by +0.26 semitones made the matches *worse*: the snare fell from 0.86 to 0.25 and the hat from 0.90 to 0.63. The drums are at their own pitch, and so are the melodic parts.

**In Live.** The 808 one-shot is rendered at the template's settled pitch, C1 + 11.8 cents (32.93 Hz). The plan plays it with Transpose +36 and Detune +2. The 2 cents are `heard_cents` (+1.6): the drop the analysis heard over the notes, less the one-shot's own. Before replica mode the one-shot sat at C1 + 33 cents (33.33 Hz) with Transpose +36, so the rebuilt 808 now settles about 19 cents lower. Melodic parts need no offset.

## Drums

`analyze` found 16 repeating sounds (12 before replica mode): 6 on the places of the beat, 10 in the residual pass. `match` named them. The ones that matter:

| sound | band, hits | where | library match | verdict |
|---|---|---|---|---|
| sound_1 | hat band, 389 detected, 54 of them under other sounds | 8ths and 16ths (steps 0, 2, 4, 6, 10, 12, 14) | a hat from a producer kit, +0.25 cents, **NCC 0.984**, margin 0.025 | same family: wins 310 of 327 hits, residual −14.9 dB (next −10.9) |
| sound_9 | hat band, 28 | step 8 (beat 3), **odd bars** | a snare from another producer kit, **−1 semitone** (−0.25 cents), NCC 0.923 | the same sound as sound_8 at another pitch |
| sound_10 | snare band, 27 | step 8, **even bars** | best NCC 0.466 | not in the library |
| sound_8 | hat band, 30 (residual pass) | bars 24–31, steps 0, 4, 6, 12, 14, 15 | sound_9's file at its own pitch (−0.25 cents), **NCC 0.986**, margin 0.027 | same family: wins 30 of 30, residual −15.7 dB (next −11.0) |
| sound_11 | hat band, 23 (residual) | steps 12–13 of odd bars from bar 35 | a copy of sound_1's file (NCC 1.0 between the two), **−2.01 semitones**, NCC 0.976 | same family |
| sound_12 | hat band, 17 (residual) | steps 12–13 of even bars from bar 36 | another hat, +0.975 semitones, NCC 0.925 | same family |
| sound_2 | low band, 93 | steps 0, 4, 12 | Core Library *Kick Doctor M*, −3, 0.848 (in the snare band), margin 0.001 | no close match (kicks all correlate) |

Before replica mode the table had sound_1 (236 hits, "same sample": 0.981, margin 0.078, 187 of 189 hits won), the −1 snare (0.953, margin 0.075, 27 of 27), the unmatched snare (0.466), a click (Core Library *E-Perc Short and Prune*, −4, 52 hits) and the kick (*Kick Doctor M*, −3, 0.910).

The other new templates are small:
- a perc every 4 bars (8 hits, a "candidate" at 0.930);
- the hat's file again, found in the snare band (7 hits);
- a snare-like sound on steps 10–11 (hat band, 8 hits, a "candidate" at 0.887);
- a low sound of 11 hits, and a low-band template of 56 hits that matches nothing above 0.672.

Four keep almost no hits: the joint fit merged a snare-band view of the kick into sound_2 and one of the hat into sound_1, and two low-band views of beat 3 keep 3 hits between them.

What the table says about the beat:
- **Two snares alternate bar by bar** on beat 3. This was invisible to a band-energy transcriber, but obvious once hits are grouped by waveform: the two groups never merge, NCC between them stays < 0.75. Bars are counted from 1, as the sections are: the identified snare plays on bars 1, 3, 5 … **Correction:** this page and the README had odd and even swapped before replica mode.
- **The same snare file at its own pitch** plays 30 times in bars 24–31, where this page said "no snares" before replica mode. The residual pass found it. A same-file twin at another pitch plays on a pad of its own, so the plan has the file twice: Transpose 0 and −1. Not checked by ear.
- **Pitched hats.** From bar 35 on, the end of most bars has a hat 2 semitones down or a hat a semitone up (40 hits). The first is the hat's own recording (a copy under another kit's name). Not checked by ear.
- **The hat under the snares.** 54 of the hat's 389 hits were found under other sounds by the joint fit: 36 under the two snares, 10 under the kick. Before replica mode, one template per moment and band could not find them.
- **The "kick" is mostly the 808's start plus a click.** 75 of its 93 hits fall within 20 ms of an 808 note's attack. The click no longer has a template of its own. Its file is now the kick template's best match in the kick band (0.81).
- **The hi-hat** is a library sample untouched: 98 % correlation after mix, master, mp3 and separation. Its verdict fell from "same sample" to "same family" only because the margin did. The search now tries every noisy sound at any pitch, and another kit's hat 3 semitones up reaches 0.958. The hits still choose the file: 310 of 327.
- **The −1 snare** took three tries to find before replica mode:
  - 0.86 for a *related* snare, when template and library were lined up on different onset definitions;
  - 0.80 for the wrong one, once the lag window was widened but transposed candidates had fallen out of the top 200;
  - 0.953 for the right one, with the pitch-invariant shortlist.

  The speckle search finds it at −1.0025 semitones: Transpose −1 and Detune 0 in whole cents. See [04](04-identifying-samples.md) and [08](08-pitfalls.md).
- **The snare's tail.** Before replica mode this page read the reference snare ringing to about 150 ms against the library's 75 ms, off a 70 dB colour scale. Measured on the new template of the snare at its own pitch (snare band, 5 ms RMS), the template and the file fall to −30 dB within 5 ms of each other (57 and 62 ms). Only under −40 dB does the reference ring longer: 75 ms against 66 ms.

## 808

- **Notes.** 94 notes by written pitch (126 before replica mode):

  | pitch | C1 | B0 | A#0 | C#2 | G1 | E1 | C2 |
  |---|---|---|---|---|---|---|---|
  | notes | 79 | 5 | 3 | 3 | 2 | 1 | 1 |
  | before replica mode | 107 | – | – | – | 3 | 1 | 1 |

  Before replica mode there were also 12 at C#1 and one each at D#1 and G0.
  - 13 notes first read as C#2: 60–90 ms each, each followed at once by a C1 note. The 10 that start on a kick are its attack split off that C1 note, which now starts there (flag `split`). 3 C#2 notes remain.
  - No note slides: legato notes are off by default. Before replica mode 3 notes slid by more than 1.5 semitones (not played).
- **Wrong: one note from bar 48 to bar 57.** The reader holds one B0 note for 15.3 s. Over that span the bass stem under 150 Hz first falls to −84 to −90 dB (RMS over 250 ms, against −7 to −27 dB where the 808 plays) for 1.0 s. Then the 808 plays alone for bars 49–51, with no kick: four rises of its envelope (the reader before replica mode wrote five notes there). Then it stays under −82 dB for 7.2 s, the breakdown.
  - The cause, measured: the reader finds four C1 notes there, but their onsets sit 27 ms before the feet of their envelope rises, past the 20 ms that `bass808.legato` allows. Their template fit is 0.887–0.893, under its 0.9. With no kick there, each reads as the note before going on, and that note takes their end.
  - Three more notes hold 0.24–1.26 s of silence. Before replica mode no note held 0.2 s of silence.
  - Not fixed.
- **Silences.** The stem's 808 rests in bars 52–56: the breakdown.
- **The template.** 53 long C1 notes, all starting on a kick, averaged into a 1.5 s template. Before replica mode, 74 C1 notes that rang 600 ms alone were the reference.
- **Pitch drop.** The template's period-by-period pitch reads 5.0 semitones over its settled pitch at 50 ms (about 44 Hz; the first reading needs two zero crossings), 1.8 at 70 ms, 0.9 at 80 ms and 0.16 at 200 ms. From 250 ms to 1.0 s it stays within 8 cents of 32.93 Hz (−7.3 to +4.2 cents; median 0.9 cents off). Before replica mode this page read 42–44 Hz falling to 33.2 Hz in about 60 ms.

**Fitted voice** (on drums + bass, so the attack is in):

| | now | before replica mode |
|---|---|---|
| reference | the 53 template notes | 74 isolated C1 notes |
| pitch | 33.05 Hz end, from 120.0 Hz (+22.3 semitones) | 32.91 Hz end, from 41.25 Hz (+3.9 semitones) |
| glide | time constant 23 ms | 60 ms |
| attack | time constant 0.3 ms | 76 ms (a soft start, under a click) |
| decay | 0.51 s | 0.46 s |
| drive | 2.88 (tanh), asym −0.028 | 1.95, nearly symmetric |
| one-shot pitch | 32.93 Hz, the template's (the fit's 33.05 Hz is 6.5 cents off it) | 33.33 Hz, the +33 cents |
| fit | 50 s | 49 s |

**Compared with the reference** (in brackets: before replica mode, against its own reference):

| | synth | best of 865 library 808s (±12 st) |
|---|---|---|
| waveform NCC (808 band) | 0.938 (0.975) | 0.894 (0.947; margin 0.001: no single 808 stands out) |
| harmonics 2–5 off by | 6.5 dB (7.4) | 6.5 dB (7.8) |
| harmonics 6–12 off by | 12.9 dB (7.7) | **8.7 dB** (2.3) |
| envelope off by | **1.3 dB** (0.9) | 1.4 dB (2.0) |
| log-mel distance | 19.3 dB (22.3) | **11.6 dB** (9.7) |

The two runs fit different references, so their numbers are not comparable cell by cell.
- Every note of the new reference starts on a kick, and the drum stem is averaged in.
- The fitted voice now starts at 120 Hz, 22 semitones over its end, and falls with a 23 ms time constant and no soft attack. Both sit on the search's bounds: the top of f_start (120 Hz) and the bottom of the attack (0.3 ms). That is the shape of a kick's sweep. Whether it comes from the kicks in the reference is not separated here.
- Both the synth and the closest library 808 correlate less with this reference than with the old one.
- Whether the one-shot now sounds like the 808 alone or like an 808 with a kick is not checked by ear. In Live the kick also plays on its own pad.

See [05](05-808-synthesis.md) for why a fit on waveform alone loses the grit entirely.

## Melodic parts

`melodic` runs Basic Pitch on the other, piano and vocal stems, each moved onto A440 first (−2, +2 and −38 cents taken out). Other and piano start at 34.65 Hz, a semitone over C1, at or under which the 808 sounds 95 % of its time ([01](01-pipeline.md#melodic)), so the 808 bleeding into them is not transcribed.
- other: 330 notes; one loop found (section B, a 1-bar period), nothing restored or dropped;
- piano: 311 notes; vocals: 509;
- chords: 141 segments.

None of this is checked against the song (no truth). On the suite's melodic and full beats (4 synthetic beats) the same step scores note F1 0.63 and chords 0.98 (major/minor; [03](03-calibration.md)).

## Sections

| | bars | what plays |
|---|---|---|
| A | 1–24 | everything |
| B | 25–28 | no hats or snares on beat 3; the snare file at its own pitch, kick and 808 |
| C | 29–32 | as B without the kick |
| A | 33–48 | |
| D | 49–56 | nothing |
| A | 57–71 | |
| E | 72–73 | outro: the 808 |

Before replica mode: A 1–24 · B 25–28 · C 29–32 · A 33–47 · D 48–51 (808 only) · E 52–56 (breakdown: drums and 808 out) · A 57–71 · F 72–73. D now takes in the 808-only bars because the 808's held note starts in bar 48, and a bar's 808 is a note that starts in it (see 808 above). The stem's 808 band agrees with the old reading: it plays through bar 51 and rests from bar 52.

The prototype's per-stem activity map (at 68 BPM bars) had the breakdown at bars 25–28 of its grid, which is 49–56 here. It also had the "other" stem dropping out earlier, and the vocals thinning at the end.

## In Live

A first draft was built in the Arrangement before the pipeline above was finished. It used the matches known at the time: the closest hat and the related snare. It held:
- an 808 track (the synthesized one-shot in Simpler, Transpose +36);
- a Drum Rack: hat, −1 snare, and the closest sound for the unmatched snare;
- a click track;
- the five stems unwarped from beat 0, with drums, bass and vocals muted;
- locators at the section starts.

The build took 4.6 s. `plan.json` from the finished pipeline differed:
- it used the identified hat and snare;
- it put the click and the kick on Drum Rack pads too: six pads, 549 notes.

`refdissect live-build` then built that plan into an empty set in 9.7 s, with the pads, notes, stems and 8 section locators checked in Live afterwards. That run found one more Live behaviour: a new audio clip's start sits at its first transient (3.6 s into the `other` stem), and the build now puts it back at 0.

Measured on the way (Live 12.4):
- Simpler loads samples at −12 dB.
- A new file in the User Library is in the browser within seconds.
- Locators at fractional beats were refused, whole beats work.

**The plan with replica mode** (not built yet, so nothing in it is verified in Live):
- 607 drum notes on 7 pads (549 on 6 before);
  - the hat, its copy at Transpose −2 / Detune −1, and the other hat at +1 / −2;
  - the snare file at 0 and at −1;
  - the unmatched snare, and the kick;
  - the click and a second kick no longer get pads;
- 94 808 notes at Transpose +36 and Detune +2, one per note, no Glide;
- three muted MIDI tracks (other, piano, vocals) with Drift as a placeholder;
- 7 locators, the 808's fader at −6.6 dB (−6.2 before) and the drums' at −12.3 dB.

See [07](07-building-in-live.md).

## Still open

- **The even-bar snare.** Nothing in the library comes close (0.466): synthesize it, or find a better library.
- **The 808 in bars 48–57** is one held note (see 808). Until the reader ends a note at silence and accepts these notes' attacks, the plan holds one B0 note there for 15 s (the 3 s one-shot plays once) instead of four C1 notes and a rest.
- **The 808 voice** now starts like a kick. Fitting it on notes without a kick, or from after the kick's decay, would tell whether the kicks in the reference cause it.
- **808 slides** are not read by default; PIRAMIT's reading has none.
- **Melody, chords and vocal chops.** They are transcribed and planned as muted MIDI tracks. The original stems still play them; choosing instruments is the next stage.
- **The ear test.** Every number here is a measurement; none of it has been confirmed by listening yet, and the replica-mode plan has not been built in Live.
