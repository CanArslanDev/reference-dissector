# 02 – Case study: Mavi – *PIRAMIT*

A Turkish trap track (album *VERITAS II*), 2:06.
- **Goal.** Take it apart and build a near replica in Ableton Live from a personal sample library and Live's own devices.
- **Starting point.** The Demucs stems and a prototype whose beat tracker gave 68.18 BPM.
- **Where the numbers come from.** The run of 2026-10-29, on the same stems and with the options of the run of 2026-10-22 (the defaults), with `--no-cache`, so every step was computed:
  - `analyze`, `melodic` and `match` with replica mode's code of c758437 (the suite's tag m3). Two more runs of that code wrote the same files, value for value but for the fit's seconds.
  - `fit-808` and `plan` again with the code of 74931ce, which gives each drum pad a level of its own and makes the 808 one-shot follow its template's pitch. Between the two, the code that `analyze`, `melodic` and `match` reach changed only by a rename and by a rule for a template of digital silence, which PIRAMIT does not reach, so their files are kept. Two runs of `fit-808` and `plan` wrote the same files, value for value but for the fit's seconds.
  - `plan` again with the code of a4f2358, which merged the fixes of the first live check: each pad plays one voice and the 808 its release as Live plays them, a note Live starts late starts 1 ms early, and the plan carries its meter. `fit-808`'s files are kept: the one-shot code changed after them (31de628) gives the same samples here. Of the run's 51 files, `plan.json`, `midi/drums.mid` and `midi/808.mid` changed.

  `results/piramit/summary.json` holds its aggregates; counts per bar and the measurements on the stems come from the same run's local files. Numbers marked *m3 code* come from c758437's `fit-808` and `plan` on the same analysis. Numbers marked *2026-10-22* come from the first run with replica mode (analysis, matches and fit with the code of 9ccc759, the plan with 4f30623's). Numbers marked *before replica mode* come from the run of 2026-10-17 (42cb7d9).
- **No truth.** Nobody has transcribed PIRAMIT by hand. Where the runs differ, this page says which one is right only where a measurement on the stems shows it; everything else is not verified.
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

**Correction 2.** The 808 is detuned, but by +11.8 cents, not +33. The +33 was the circular mean of each note's pYIN median over 60–450 ms, and that window hears the 808's drop. The 808 reader now learns the song's own 808 from its long notes (53 notes, all starting on a kick) and reads its settled pitch by QIFFT over 300–600 ms, where the window's two halves agree within 3 cents. Each note is a transposition of it; the notes' settled pitches agree on +11.5 cents (0.799). On 2026-10-22 they agreed on +10.7 (0.868): six of the eight notes the reader now adds settle 42–53 cents over C1 ([808](#808)).

**What it explains.** The prototype's 808 notes split between C (90) and C# (82), and its key estimate wavered between C minor and C major. Notes at C + 30 cents round either way. Before replica mode, with the +33 cents removed before rounding, 107 of 126 notes were C1. Now 87 of 101 are (79 of 94 on 2026-10-22).

**What it is not.** It could have been a whole track sped up: 1.5 % faster would also turn 135 BPM into 137. Matching every drum sample shifted by +0.26 semitones made the matches *worse*: the snare fell from 0.86 to 0.25 and the hat from 0.90 to 0.63. The drums are at their own pitch, and so are the melodic parts.

**In Live.** The 808 one-shot is rendered at the template's settled pitch, C1 + 11.8 cents (32.93 Hz). The plan plays it with Transpose +36 and Detune +1. The 1 cent is `heard_cents` (+0.8): the drop the analysis heard over the notes, less the one-shot's own. With the m3 code's one-shot, whose pitch fell faster than the template's, it was +1.6 and Detune +2 ([808](#808)). Before replica mode the one-shot sat at C1 + 33 cents (33.33 Hz) with Transpose +36, so the rebuilt 808 now settles about 20 cents lower. Melodic parts need no offset.

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
- a snare-like sound on steps 10–11 (hat band, 8 hits, a "candidate" at 0.886);
- a low sound of 11 hits, and a low-band template of 56 hits that matches nothing above 0.672.

Four keep almost no hits: the joint fit merged a snare-band view of the kick into sound_2 and one of the hat into sound_1, and two low-band views of beat 3 keep 4 hits between them (3 on 2026-10-22; the fourth at NCC 0.60).

The drums' analysis is otherwise 2026-10-22's: the same 16 templates, sample for sample, the same merges and the same hits per sound, none of them moved. Only the gains of 23 hits changed: 9 of the hat's, all 11 of the low sound's, and one each of the kick, the unmatched snare and the hat 2 semitones down.

What the table says about the beat:
- **Two snares alternate bar by bar** on beat 3. This was invisible to a band-energy transcriber, but obvious once hits are grouped by waveform: the two groups never merge, NCC between them stays < 0.75. Bars are counted from 1, as the sections are: the identified snare plays on bars 1, 3, 5 … **Correction:** this page and the README had odd and even swapped before replica mode.
- **The same snare file at its own pitch** plays 30 times in bars 24–31, where this page said "no snares" before replica mode. The residual pass found it. A same-file twin at another pitch plays on a pad of its own, so the plan has the file twice: Transpose 0 and −1. Not checked by ear.
- **Pitched hats.** From bar 35 on, the end of most bars has a hat 2 semitones down or a hat a semitone up (40 hits). The first is the hat's own recording (a copy under another kit's name). Not checked by ear.
- **The hat under the snares.** 54 of the hat's 389 hits were found under other sounds by the joint fit: 36 under the two snares, 10 under the kick. Before replica mode, one template per moment and band could not find them.
- **The "kick" is mostly the 808's start plus a click.** 76 of its 93 hits fall within 20 ms of an 808 note's attack (75 on 2026-10-22: one of the 808's new notes starts on a kick). The click no longer has a template of its own. Its file is now the kick template's best match in the kick band (0.81).
- **The hi-hat** is a library sample untouched: 98 % correlation after mix, master, mp3 and separation. Its verdict fell from "same sample" to "same family" only because the margin did. The search now tries every noisy sound at any pitch, and another kit's hat 3 semitones up reaches 0.958. The hits still choose the file: 310 of 327.
- **The −1 snare** took three tries to find before replica mode:
  - 0.86 for a *related* snare, when template and library were lined up on different onset definitions;
  - 0.80 for the wrong one, once the lag window was widened but transposed candidates had fallen out of the top 200;
  - 0.953 for the right one, with the pitch-invariant shortlist.

  The speckle search finds it at −1.0025 semitones: Transpose −1 and Detune 0 in whole cents. See [04](04-identifying-samples.md) and [08](08-pitfalls.md).
- **The snare's tail.** Before replica mode this page read the reference snare ringing to about 150 ms against the library's 75 ms, off a 70 dB colour scale. Measured on the new template of the snare at its own pitch (snare band, 5 ms RMS), the template and the file fall to −30 dB within 5 ms of each other (57 and 62 ms). Only under −40 dB does the reference ring longer: 75 ms against 66 ms.

## 808

- **Notes.** 101 notes by written pitch (94 on 2026-10-22, 126 before replica mode):

  | pitch | C1 | B0 | A#0 | C#2 | G1 | C2 | E1 |
  |---|---|---|---|---|---|---|---|
  | notes | 87 | 5 | 3 | 3 | 2 | 1 | – |
  | 2026-10-22 | 79 | 5 | 3 | 3 | 2 | 1 | 1 |
  | before replica mode | 107 | – | – | – | 3 | 1 | 1 |

  Before replica mode there were also 12 at C#1 and one each at D#1 and G0.
  - 13 notes first read as C#2: 60–90 ms each, each followed at once by a C1 note. The 10 that start on a kick are its attack split off that C1 note, which now starts there (flag `split`). 3 C#2 notes remain.
  - No note slides: legato notes are off by default. Before replica mode 3 notes slid by more than 1.5 semitones (not played).
- **Fixed: notes held over silence.** On 2026-10-22 the reader held one note through sections D and E. Over that span the bass stem under 150 Hz first falls to −84 to −90 dB (RMS over 250 ms, against −7 to −27 dB where the 808 plays) for 1.0 s. Then the 808 plays alone, with no kick, to the end of D; then the stem stays under −82 dB for 7.2 s, the breakdown (E). The reader found C1 notes in D, but their onsets sat 27 ms before the feet of their envelope rises, past the 20 ms that `bass808.legato` allows, and their template fit was 0.887–0.893, under its 0.9: each read as the note before going on. Two rules of the reader changed ([01](01-pipeline.md#808-notes-bass808transcribe)):
  - a note ends where the 808 falls silent (its 25 ms RMS 40 dB or more under the note's peak for 50 ms), and a note found after that is a note of its own;
  - a rise of 9 dB or more attacks a note, and such a note starts on its rise.

  Silence measured on the stem the same way for both runs: the 25 ms RMS of the bass stem's 808 band 40 dB or more under the median note peak, for 0.2 s or more inside a note.

  | | 2026-10-22 | now |
  |---|---|---|
  | notes held over silence | 4 (0.27, 0.65, 1.57 and 8.63 s of it; the last in two rests, 1.22 and 7.41 s) | 0 |
  | C1 notes in section D, where the 808 plays alone | 0 | 4 |

  - 93 of the 94 notes of 2026-10-22 are found again within 50 ms. None moved in onset, attack or written pitch. 6 end earlier, the held note among them.
  - 8 notes are new, all C1: five start on a rise of 11–21 dB inside what 2026-10-22 read as one longer note (one of them on a kick), three after a rest. Each attacks 5.2–5.45 ms after its onset, as the others do (median 5.44 ms).
  - 1 is gone: the E1 note, found in a rest, whose peak sat 49 dB under the other notes'.
- **Still wrong in section D: a fifth note.** The stem shows one more attack there: its pitch (80 ms windows) falls about 4 semitones within 120 ms, as it does at the attacks the reader finds there. But no kick plays there, and the 808 band rises only 6.5 dB over the note still ringing, where a rise needs 9. The reader keeps it in the note before; the reader before replica mode wrote a note there. Not checked by ear.
- **Read sharp: the notes of section D.** The reader settles them at +42 to +43 cents over C1. The stem's own sustain, read as the template is (QIFFT over 300–600 ms after the attack), gives +11.8 to +12.3 cents on three of them, the template's pitch; the fourth's window holds the fifth attack. Two other new notes settle at +52 to +53 cents; under 0.6 s long, they are too short for that window. All are written C1 either way, and the plan plays every 808 note at the track's one tuning.
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
| fit | 43 s in two runs (49 s with the m3 code beside another job, 50 s on 2026-10-22) | 49 s |

The fit is 2026-10-22's: the template and its 53 notes are the same, and so are the parameters, the loss and the comparison below. So was the one-shot, byte for byte, up to the m3 code; the one the plan plays now follows the template's own pitch (below).

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

**The one-shot the plan plays.** Since 74931ce, `fit-808` writes the fitted voice on the template's own periods (the period-by-period pitch of the 53 notes) as long as its drop lasts: 9 of its 49 periods, to 269 ms, where it has settled; then the fitted drop. It is laid out as the template is: it attacks 5.2 ms after its start, where the template attacks 5.4 ms after its onset, so the plan starts each note at its onset. The m3 code's one-shot was the fitted voice alone at the template's settled pitch, attacking at its start.

| | now | m3 code |
|---|---|---|
| its pitch against the template's, played attack on attack: median (max) distance over 50–100 ms | 1.5 cents (14) | 65 cents (85), sharp |
| over 100–150 ms | 0.2 cents (0.5) | 38 cents (48), flat |
| over 150–300 ms | 0.1 cents (1.4) | 10 cents (33) |
| over 300–600 ms | 1.1 cents (4.4) | 1.1 cents (4.4) |
| within 20 cents of its settled pitch from (after its onset) | 190 ms, as the template | 125 ms |
| against the fit's reference (808 band, drums + bass): waveform NCC, log-mel distance | 0.880, 19.3 dB | 0.921, 18.9 dB |
| harmonics 2–5 and 6–12, envelope, off by | 6.5, 12.9, 1.3 dB | 6.5, 12.9, 1.3 dB |

- The table starts at 50 ms: the template's curve has no reading before 40 ms (the first needs two zero crossings).
- The waveform matches the fit's reference less. The one-shot now follows the bass stem's 808, while the reference holds the kicks too; whether that is why is not separated, and nothing here is checked by ear.
- **One Detune for every length.** `heard_808`'s correction per C1 note, by the note's length (6 notes of 120–250 ms, 15 of 250–600 ms, 66 longer; none shorter): +41.4, +4.8 and +1.6 cents with the m3 code; −0.1, 0.0 and +0.8 now. The track takes their median, +0.8 cents (+1.6): Detune +1 (+2). All 87 corrections lie within 8 cents of 0 (81 with the m3 code).
- **Played as the plan plays it** (`estimate_from_plan`: each note heard as the one-shot sounds over its length), against the pitch the analysis heard, the 101 notes paired within 20 ms. By length: 2 notes under 120 ms +131 cents (+205 with the m3 code), 9 of 120–250 ms +0.3 (−40), 22 of 250–600 ms +1.3 (−2.1), 68 longer +0.4 (+0.6). 91 of 101 sound within 20 cents of the analysis (82), the median distance 1.1 cents (1.4) and the p90 19 (39). The two notes under 120 ms still read far apart; why is not checked. This compares the plan with the analysis, not with the song: nobody has transcribed it.

## Melodic parts

`melodic` runs Basic Pitch on the other, piano and vocal stems, each moved onto A440 first (−2, +2 and −38 cents taken out). Other and piano start at 34.65 Hz, a semitone over C1, at or under which the 808 sounds 95 % of its time ([01](01-pipeline.md#melodic)), so the 808 bleeding into them is not transcribed.
- other: 330 notes; one loop found (section B, a 1-bar period), nothing restored or dropped;
- piano: 311 notes; vocals: 509;
- chords: 143 segments (141 on 2026-10-22). The notes are the same; the chords take the 808's notes as their bass, and the segments that changed all lie where the 808's notes did (four places, the breakdown among them).

None of this is checked against the song (no truth). On the suite's melodic and full beats (4 synthetic beats) the same step scores note F1 0.63 and chords 0.98 (major/minor; [03](03-calibration.md)).

## Sections

| | bars | what plays |
|---|---|---|
| A | 1–24 | everything |
| B | 25–28 | no hats or snares on beat 3; the snare file at its own pitch, kick and 808 |
| C | 29–32 | as B without the kick |
| A | 33–47 | |
| D | 48–51 | the 808 alone |
| E | 52–56 | nothing: the breakdown |
| A | 57–71 | |
| D | 72–73 | outro: the 808 alone, as in D |

These are the sections read before replica mode, but for the outro's name: it plays what D plays, so it is D again (F before replica mode). On 2026-10-22 one 808 note was held from the start of D through E, and a bar's 808 is a note that starts in it, so D's later bars and E read as one section, D 49–56, and A took in D's first bar (see 808 above). The stem's 808 band agrees with this reading: it plays through D and rests in E.

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

`refdissect live-build` then built that plan into an empty set in 9.7 s, with the pads, notes, stems and 8 section locators checked in Live afterwards. That run found one more Live behaviour: a new audio clip's start sits at its first transient (a few seconds into the `other` stem), and the build now puts it back at 0.

Measured on the way (Live 12.4):
- Simpler loads samples at −12 dB.
- A new file in the User Library is in the browser within seconds.
- Locators at fractional beats were refused, whole beats work.

**The plan with replica mode** (a4f2358; built and recorded in Live 12.4.6, below):
- 630 drum notes on 10 pads (607 on 7 on 2026-10-22, 549 on 6 before replica mode);
  - the hat, its copy at Transpose −2 / Detune −1, and the other hat at +1 / −2;
  - the snare file at 0 and at −1;
  - the unmatched snare, and the kick;
  - new: the perc (8 notes, a "candidate" at 0.930), the hat's file found in the snare band (7 notes, 0.947) and the snare-like sound (8 notes, a "candidate" at 0.886);
  - the click and a second kick get no pads, as on 2026-10-22;
- 101 808 notes at Transpose +36 and Detune +1 (+2 with the m3 code), one per note, no Glide;
- three muted MIDI tracks (other, piano, vocals) with Drift as a placeholder;
- 4/4 over 292 beats, the meter the build sets in Live;
- 8 locators, the 808's fader at −13.9 dB (−13.6 on 2026-10-22) and the drums' at −12.6 dB (−12.5 at 74931ce), each pad at a Volume of its own from −12.1 to 0 dB (with the m3 code, −20.2 dB and every pad at 0 dB). Velocities follow Live 12.4.6's law at "Vol < Vel" 35 %, under which velocity 127 plays 7.9 dB over the Volume. On 2026-10-22, taken as linear levels (v/127), the same notes had the faders at −6.6 and −12.3 dB (the 808's at −6.2 dB before replica mode).

**What changed in the plan from 2026-10-22 to the m3 code:**
- **Three pads are new**, and none is gone. The new ones are sounds of the residual pass that the 12-hit floor kept out. A pad now takes, here, 5 hits (one per 16 bars), clean 0.5, and for a sound of the residual pass, hits that recur: at least half of them with another hit at their place of the bar (within 6 ms) 1–8 bars away ([12](12-replica-mode.md)). The perc recurs at 0.75 and is clean 0.875; the hat 1.0 and 1.0; the snare-like sound 0.75 and 1.0. None of the three is checked by ear.
  - 6 of the perc's 8 notes lie 0.7 ms before a note of the main hat: a layer, or the hat found again (not verified).
  - The new hat plays a copy of the hat's file (NCC 1.0 between the two files) on step 14 of 7 bars. In 5 the main hat has no note there; the other 2 double a main-hat note 0.3 ms away, at gains of 0.29 and 0.35 where its other hits have 0.71–1.12. The rule that plays a strike once looks at one pad, not at one recording on two.
  - The snare-like sound lies within 15 ms of no other pad's note.
- **The 7 earlier pads play the same notes**, one for one. The rules that play each strike once (the backbone's kicks and snares, and one sample per moment on a pad) drop nothing here. 7 of their 607 notes moved 1–3 velocity steps, with the gains above.
- **A note starts as early as its pad's sample takes to attack**: the kick 0.16 ms, the unmatched snare 0.57, the snare file 2.93 (3.11 a semitone down), the perc 1.04, the snare-like sound 0.18, the hats 0. Each earlier pad's note lies within 0.05 ms of its 2026-10-22 note less that lead.
- **Pads moved.** The backbone (a kick or snare found on the places of the beat, clean and played throughout) is now placed first: the unmatched snare took pad 38 and the snare file 40, the other way round before. The new hat took 46 by its NCC, and the other hat moved to 49.
- **Pads still sound together** within 15 ms as on 2026-10-22: the kick and the hat on 61 notes, the −1 snare and the hat on 28, the unmatched snare and the hat on 20, the kick and the snare file on 5.
- **The 808**: the 93 notes found again start where they did, at the same pitch; 6 are shorter. 22 of them play a velocity step lower: velocity 115 is the line's median peak, which rose 0.045 dB with the new notes.
- **Locators**: D (48) and E (52) where 2026-10-22 had D (49), and D (72) for the outro.

**What 74931ce changed** (`fit-808` and `plan` with 74931ce, against the m3 code, on the same analysis):
- **Each pad plays at a level of its own**, where the stems hold its sound: the median over its hits of the stems' energy over 50 ms from each attack, in its role's band, against the pad's own track at 0 dB. The loudest pad plays at 0 dB and the others under it; the drum fader matches the drums' RMS to the drum stem's.

  | pad | Volume | fader + Volume | with the m3 code |
  |---|---|---|---|
  | kick | −8.5 dB | −21.0 dB | −20.2 dB |
  | unmatched snare | −4.4 dB | −16.9 dB | −20.2 dB |
  | snare file, at 0 and at −1 | −5.3 dB | −17.8 dB | −20.2 dB |
  | hat | −6.8 dB | −19.3 dB | −20.2 dB |
  | the hat's copy at −2 | −10.3 dB | −22.8 dB | −20.2 dB |
  | the hat's file found in the snare band | −7.9 dB | −20.4 dB | −20.2 dB |
  | other hat at +1 | −10.1 dB | −22.6 dB | −20.2 dB |
  | perc | 0.0 dB | −12.5 dB | −20.2 dB |
  | snare-like sound | −12.1 dB | −24.6 dB | −20.2 dB |

- The drum fader, −12.5 dB, is the RMS match; the stop at the loudest pad (−10.3 dB) does not bind.
- **The kick** is read on the drum and bass stems together, less the plan's own 808: 14.7 dB under its pad at 0 dB, over 87 of its 93 hits (at the other 6 the 808 leaves nothing). The drum stem alone holds it 16.4 dB under, so the floor at that reading does not bind. Under the m3 code's one-shot the kick read 14.9 dB under: the new one-shot moved its Volume from −8.6 to −8.5 dB and the drum fader from −12.4 to −12.5 dB. The reading leans on that 808: played 1 dB quieter under the kick, it would put the kick's Volume at −5.6 dB; 1 dB louder, at −10.1 dB, the floor. Not verified.
- **The perc is the loudest pad**, read on hits it mostly shares: 6 of its 8 notes lie within 50 ms of another pad's (0.7 ms before the main hat's). On its 2 clear hits it reads 3.0 dB lower; the rule that reads a sound on its clear hits needs 3 of them, reading more than 6 dB under all its hits. Against the other pads it may play 3 dB loud. Not verified.
- **These levels are the stems'.** No truth says where the song holds each sound. On the suite's beats, a pad read on the drum stem sits a median 1.3 dB under where the truth holds its sound. Not checked by ear.
- **The 808's notes start at their onsets**: a median 5.3 ms earlier than with the m3 code (2.4 to 5.9 ms), now a median 0.2 ms after the analysis' onset, where the m3 code started them on their attacks. In both plans each note attacks within 0.02 ms of the analysis' attack. The pitches and velocities are the same; 28 notes end up to 2.5 ms earlier (one voice: a note ends just before the next starts). Detune +1 (see [808](#808)); the 808's fader stays at −13.9 dB.
- **Nothing else moved**: the drum notes, each pad's file, Transpose and Detune, the melodic tracks, the stems and the locators. `drums.mid` and the melodic MIDI files are the same, byte for byte; `808.mid` moved with the starts.
- **Buildable, but not built whole.** The plan's tracks pass a format check against ableton-live-mcp 2.2.0's `build_tracks` (pad Volumes as "x.y dB" at or under 0 dB, Detune in range, no Glide). In Live its build stopped at a pad (below).
- **Seconds.** `fit-808` took 45.6 and 46.1 s in the two runs (the fit itself 42.8 and 43.1 s), `plan` 1.9 and 2.0 s, with Live open, no suite running and a load of 2.5–3.6. `plan` now renders each pad and reads the stems for the levels: 0.68 GB at its peak. With the m3 code, beside another job, `fit-808` took 61.6 s and `plan` 1.4 s (0.24 GB); on a quiet machine, 45.8 s and 0.7 s.

**What a4f2358 changed** (`plan` again, on the same analysis, matches and one-shot):
- **The meter.** The plan carries its meter, 4/4 as the analysis reads it; 292 beats and the 8 locators as before.
- **Notes that Live starts late start 1 ms early.** 98 of the 630 drum notes come while their pad's hit before still plays, and Live cuts that one and starts them 1 ms late: 81 of the kick's 93, 10 of 23 on a hat pad, 6 of 30 on a snare pad, 1 of the perc's 8. Each starts 1 ms earlier, and 17 notes' lengths move with them. 62 of the 101 808 notes come while a note of their pitch still sounds in its release, and start 1 ms earlier too; 49 end 1 ms earlier with them. Pitches, velocities, Transpose +36, Detune +1 and the 808's fader are unchanged.
- **The kick** reads 14.4 dB under its pad at 0 dB, over 86 of its 93 hits (14.7 over 87): its Volume −8.5 → −8.1 dB, and the drum fader −12.5 → −12.6 dB. Every other pad's Volume and the melodic tracks are unchanged. The 0.37 dB comes from `pad_levels` reading the hits Live starts late where the plan now writes them, 1 ms early; read where Live plays them, the kick reads 14.7 dB under ([12](12-replica-mode.md#known-limits)).
- **The perc** reads 2.4 dB lower on its 2 clear hits than on its 8 (3.0 before).

**In Live 12.4.6** (2026-10-29; [07](07-building-in-live.md#live-behaviour-measured-here)):
- **The first live check** built the plan of 74931ce. Its build stopped at the snare-like sound's pad: its file's name is one character long, Live's browser search did not return it within 40 s, and `build_tracks` raised after the 808 track. The other hat's pad loaded an FX hit of the same name. The drums were built again without the stopped pad and recorded (9 pads, all 630 notes); the melodic tracks and stems were not built. Snap ended the snare file 25.4 and 26.9 ms early on its two pads, and the unmatched snare 4.7 ms; 81 of the kick's 93 hits started 1 ms late, as each cut the one before.
- **The second live check** built the plan of a4f2358 whole and recorded it. It went as 5 `build_tracks` calls: one message from that session could not carry its 1,881 notes. Its 11 Simplers held the plan's files (SHA-1), the two pads above among them, and Live held the 1,881 notes at their starts, pitches and velocities (it shortened 1 note at the 0.05-beat floor).
  - The 630 drum hits played where `render.py` plays them, within 0.04 ms; the snare file's hits played to the end of the file, within 0.02 ms of `render.py`.
  - The 101 808 notes started within 0.11 ms of `render.py`'s, at a median −48.8 dB of residual (−46.8 on the 62 restarts). The 20 notes that end in a rest fell to −40 dB 25 ms after their end, in Live and in `render.py`.
  - The 8 locators sat where planned, in 4/4; the 3 muted Drift tracks and the 5 stems were built too.
- Not checked by ear, and against `render.py`, not the song: PIRAMIT has no truth.

See [07](07-building-in-live.md).

## Still open

- **The even-bar snare.** Nothing in the library comes close (0.466): synthesize it, or find a better library.
- **The 808 in section D** has four notes where the stem shows five: one attack rises 6.5 dB over the note still ringing, under the 9 dB a rise needs, and no kick marks it (see 808). The reader also settles these notes about 30 cents over what their sustain reads; the plan does not play that.
- **One recording on two pads.** The hat's file found in the snare band doubles 2 of the main hat's notes (see In Live).
- **The 808 voice** now starts like a kick. Fitting it on notes without a kick, or from after the kick's decay, would tell whether the kicks in the reference cause it.
- **The one-shot's waveform** matches the fit's reference less than the m3 code's did (NCC 0.880 against 0.921), while its pitch now follows the template's. Which of the two sounds closer is not checked by ear.
- **The perc's level** is read on hits it mostly shares with the main hat; on its 2 clear hits it reads 2.4 dB lower (see In Live).
- **808 slides** are not read by default; PIRAMIT's reading has none.
- **Melody, chords and vocal chops.** They are transcribed and planned as muted MIDI tracks. The original stems still play them; choosing instruments is the next stage.
- **The ear test.** Every number here is a measurement; none of it has been confirmed by listening yet, the pad levels and the new one-shot included. The replica-mode plan was built and recorded in Live, and played as `render.py` plays it.
- **The kick's level and one voice.** 81 of the kick's 93 hits cut the one before in Live, while its level is read against its hits summed, as the suite's truth sums them; if the song's own sampler cut them too, the reading runs low. Not measured here.
