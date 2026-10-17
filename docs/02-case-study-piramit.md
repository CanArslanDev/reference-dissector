# 02 – Case study: Mavi – *PIRAMIT*

A Turkish trap track (album *VERITAS II*), 2:06.
- **Goal.** Take it apart and build a near replica in Ableton Live from a personal sample library and Live's own devices.
- **Starting point.** The Demucs stems and a prototype whose beat tracker gave 68.18 BPM.
- **Where the numbers come from.** `results/piramit/summary.json` unless noted.
- **What is not here.** The song, its stems and the note-by-note transcription stay local: see [09](09-data-and-copyright.md).

## Grid: 137.00 BPM, not 136.36

**What the prototype assumed.** The beat tracker (`beat_this`) reported 68.18 BPM, half time, so the prototype assumed 136.36.

**The fitted grid.** 137.002 BPM with the first beat at 31.9 ms, a sixteenth at 109.5 ms. Strong snare-band onsets per beat of the bar sit within these offsets of the grid line:

| | bars with an onset | median | p90 of \|offset\| |
|---|---|---|---|
| beat 1 | 60 | −0.1 ms | 10.8 ms |
| beat 2 | 60 | +1.1 ms | 8.6 ms |
| beat 3 | 58 | +1.5 ms | 4.6 ms |
| beat 4 | 62 | −0.5 ms | 11.8 ms |

**At 136.36 BPM**, the same onsets start at +29 ms and end at −49 ms. An average inter-beat interval hides a drift that a fitted grid shows at once.

**Lesson.** Fit the grid; then check that the *last* bars still land on it.

## Tuning: the track at A440, the 808 at +33 cents

| stem | deviation from A440 (stable spectral peaks) | first version (`librosa.estimate_tuning`) |
|---|---|---|
| bass | +11 cents | +23 |
| other | −2 | +27 |
| piano | +1.5 | +28 |
| vocals | −38 | +32 |
| **808 notes** (circular mean of 126 pitches) | **+33** (agreement 0.64) | – |

**Correction.** This section first said the whole track sat at +0.28 semitones (A4 ≈ 447 Hz). That came from `librosa.estimate_tuning`, which is off by up to 60 cents on synthetic tones. Measured with stable spectral peaks (within 1 cent on synthetic tones) and checked with pyin (+2 to +3 cents on piano and other), the melodic stems are at A440. Only the 808 is detuned: its sample sits about a third of a semitone sharp. The bass stem's +11 is the 808 mixed with cleaner bass notes.

**What it explains.** The prototype's 808 notes split between C (90) and C# (82), and its key estimate wavered between C minor and C major. Notes at C + 30 cents round either way. With the 808's own offset removed before rounding, 107 of the 126 808 notes are C1 (105 with the first version's +28 cents).

**What it is not.** It could have been a whole track sped up: 1.5 % faster would also turn 135 BPM into 137. Matching every drum sample shifted by +0.26 semitones made the matches *worse*: the snare fell from 0.86 to 0.25 and the hat from 0.90 to 0.63. The drums are at their own pitch, and so are the melodic parts.

**In Live.** The 808 one-shot is rendered at the 808's own pitch: C1 + 33 cents, 33.33 Hz (33.24 Hz in the first build). Melodic parts need no offset.

## Drums

`analyze` found 12 repeating sounds; `match` named them. The ones that matter:

| sound | band, hits | grid slots | library match | verdict |
|---|---|---|---|---|
| sound_1 | hat band, 236 detected | eighths and sixteenths, all bars | a hat from a producer kit, ±0, **NCC 0.981**, margin 0.078 | **same sample**: wins 187/189 hits, residual −14.3 dB (next −7.4) |
| sound_10 | hat band, 28 | step 8 (beat 3), **even bars** | a snare from another producer kit, **−1 semitone**, **NCC 0.953**, margin 0.075 | **same sample**: wins 27/27, residual −10.4 dB (next −6.4) |
| sound_11 | snare band, 27 | step 8, **odd bars** | best NCC 0.466 | not in the library |
| sound_3 | hat band, 52 | step 0 | Core Library *E-Perc Short and Prune*, −4 semitones, 0.865 | a click on the downbeat |
| sound_4 | low band, 133 | steps 0, 4, 12 | Core Library *Kick Doctor M*, −3, 0.910, margin 0.012 | same family (kicks all correlate) |
| sound_2, sound_12 | hat band | | the same hat as sound_1 (0.954, 0.950) | the same sound found twice |

What the table says about the beat:
- **Two snares alternate bar by bar** on beat 3. This was invisible to a band-energy transcriber, but obvious once hits are grouped by waveform: the two groups never merge, NCC between them stays < 0.75.
- **The "kick" is mostly the 808's start plus a click.** 52 of 62 kick-band hits fall within 20 ms of an 808 note onset. The click matches a short Core Library percussion sound pitched down 4 semitones.
- **The hi-hat** is a library sample untouched: 98 % correlation after mix, master, mp3 and separation.
- **The even-bar snare** is a library sample pitched down a semitone. It took three tries to find it:
  - 0.86 for a *related* snare, when template and library were lined up on different onset definitions;
  - 0.80 for the wrong one, once the lag window was widened but transposed candidates had fallen out of the top 200;
  - 0.953 for the right one, with the pitch-invariant shortlist.
  
  See [08](08-pitfalls.md).
- **The snare's tail.** The library snare ends at 75 ms; the reference snare rings to about 150 ms and has a hat 220 ms later, which is pattern bleed. Reverb or a longer decay is the next adjustment (figure in the README, middle row).

## 808

- **Notes.** 126 notes by pitch (with the tuning removed):

  | pitch | C1 | C#1 | G1 | D#1 | E1 | G0 | C2 |
  |---|---|---|---|---|---|---|---|
  | notes | 107 | 12 | 3 | 1 | 1 | 1 | 1 |

  3 notes slide by more than 1.5 semitones (these are not played yet).
- **Silences.** The 808 rests in bars 52–56: the breakdown.
- **Isolated notes.** 74 C1 notes ring at least 600 ms alone; their average is the reference for the fit.
- **Pitch drop.** The pitch falls from about 42–44 Hz to 33.2 Hz in the first ~60 ms.

**Fitted voice** (on drums + bass, so the attack is in):

| | value |
|---|---|
| pitch | 32.91 Hz end, from 41.25 Hz (+3.9 semitones) |
| glide | time constant 60 ms |
| attack | time constant 76 ms (a soft start, under a click) |
| decay | 0.46 s |
| drive | 1.95 (tanh), nearly symmetric |
| fit | 49 s (199 s before the fit was made faster) |

**Compared with the average of the isolated notes:**

| | synth | best of 865 library 808s (±12 st) |
|---|---|---|
| waveform NCC (808 band) | **0.975** | 0.947 (margin 0.002: no single 808 stands out) |
| harmonics 2–5 off by | **7.4 dB** | 7.8 dB |
| harmonics 6–12 off by | 7.7 dB | **2.3 dB** |
| envelope off by | **0.9 dB** | 2.0 dB |
| log-mel distance | 22.3 dB | **9.7 dB** |

The synth gets pitch, glide and envelope right; the library 808 has the right amount of grit. See [05](05-808-synthesis.md) for why a fit on waveform alone loses the grit entirely.

## Sections

| | bars | what plays |
|---|---|---|
| A | 1–24 | everything |
| B | 25–28 | no snares, 808 on |
| C | 29–32 | as B with fewer hats |
| A | 33–47 | |
| D | 48–51 | 808 only |
| E | 52–56 | breakdown: drums and 808 out |
| A | 57–71 | |
| F | 72–73 | outro |

The prototype's per-stem activity map (at 68 BPM bars) had the breakdown at bars 25–28 of its grid, which is 49–56 here. It also had the "other" stem dropping out earlier, and the vocals thinning at the end.

## In Live

A first draft was built in the Arrangement before the pipeline above was finished. It used the matches known at the time: the closest hat and the related snare. It held:
- an 808 track (the synthesized one-shot in Simpler, Transpose +36);
- a Drum Rack: hat, −1 snare, and the closest sound for the odd-bar snare;
- a click track;
- the five stems unwarped from beat 0, with drums, bass and vocals muted;
- locators at the section starts.

The build took 4.6 s. `plan.json` from the finished pipeline differs:
- it uses the identified hat and snare;
- it puts the click and the kick on Drum Rack pads too: six pads, 549 notes.

`refdissect live-build` then built that plan into an empty set in 9.7 s, with the pads, notes, stems and 8 section locators checked in Live afterwards. That run found one more Live behaviour: a new audio clip's start sits at its first transient (3.6 s into the `other` stem), and the build now puts it back at 0.

Measured on the way (Live 12.4):
- Simpler loads samples at −12 dB.
- A new file in the User Library is in the browser within seconds.
- Locators at fractional beats were refused, whole beats work.

See [07](07-building-in-live.md).

## Still open

- **Odd-bar snare.** Nothing in the library comes close (0.47): synthesize it, or find a better library.
- **808 slides** are detected but not played; Simpler's glide needs overlapping notes.
- **Melody, chords and vocal chops** play from the original stems. Transcribing them (Basic Pitch is in ableton-live-mcp as `audio_to_midi`) and choosing instruments is the next stage.
- **The ear test.** Every number here is a measurement; none of it has been confirmed by listening yet. The draft is in Live for exactly that.
