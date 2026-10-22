# 08 – Pitfalls (all of them were hit here)

## Time

- **Half time and double time.** A beat tracker reported 68.18 BPM for a 137 BPM track. Both fit, because every other grid line carries a hit. Decide explicitly with `--bpm-range`, then check that the snare sits on the beat the genre puts it on (trap: beat 3).
- **An average tempo hides drift.** 136.36 BPM was 0.47 % off: PIRAMIT's snares sat 2 ms off its grid in bar 1, 50 ms off in bar 7 and 0.59 s off by the end (computed). `grid.offsets` looks within 50 ms of each grid line, so it lost them after bar 7, and this page said "49 ms off by the end". Fit the grid, then check that the *last* bars still land on it, and count them: at 136.999 BPM `grid.offsets` finds beat 3 in 58 bars; at 136.36 it finds the snares in bars 1–7 only.
- **A tracker's median interval is quantized.** beat_this's beats come on 20 ms frames, so their median interval is a whole number of frames: 0.88 s (68.18 BPM) on PIRAMIT. A line through the same beats gives 68.498.
- **Seconds versus beats.** MIDI notes go into Live in beats. Convert with the fitted tempo (beat = s × BPM / 60), and place the stems unwarped at beat 0 so both share a clock. A MIDI file's own tempo is not applied by Live's import: ableton-live-mcp's `import_midi_file` reads beats.
- **The grid phase is not the attack.** Onset-strength peaks come 4.8–13.2 ms after the attacks (82 test beats; 7.1 ms on PIRAMIT's drums). That is fine for finding a tempo; it is not fine for lining up waveforms or placing notes. The grid is moved onto the attacks, and every hit and note is timed where its sound starts.

## Pitch

- **An 808 not at A440.** PIRAMIT's 808 settles at +11.8 cents while the rest of the track is at A440. Read about 30 cents over C, with its drop, every 808 note came out "C or C#". Measure the 808's own tuning and remove it before writing notes.
- **A pitch tracker on an 808 measures its drop.** An 808's pitch falls for 0.1–0.6 s after every attack, and a median over the note reads the drop. Read the note as a transposition of the 808 itself: PIRAMIT's 808 read +33 cents by a pYIN median over 60–450 ms, +11.8 by QIFFT of its template's sustain. pYIN also reads 14–27 cents sharp under 50 Hz by itself.
- **An 808 may not have settled by 300 ms.** Check that the two halves of the sustain window agree: one suite 808 read 35 cents sharp over 300–600 ms and 1.2 cents flat over 600–750 ms.
- **A short 808 never settles.** bass_low's 110–220 ms notes never reach their settled pitch. Read from a fitted drop, it came out 44–60 cents sharp and every note of one beat was written a semitone high. On htdemucs_ft, read from a 217 ms tail, it was 2.4 semitones off (note F1@50 0.34).
- **`librosa.estimate_tuning` is not reliable.** Up to 60 cents off on synthetic tones; it read PIRAMIT as +0.28 semitones throughout. Check any tuning method on tones of known pitch first.
- **Sped up, or tuned?** Same pitch shift, different cause. Test it: shift every drum sample by the offset and see whether matches improve. On PIRAMIT they collapsed (0.86 → 0.25), so the drums were not shifted.
- **Transposed samples decorrelate completely.** −1 semitone: 0.95 at −1, 0.15 at 0, 0.13 at ±0.25. Never assume 0, and do not shortlist candidates by their score at 0.
- **Whole semitones are not a pitch.** The true file's NCC halves within 1–5 cents. A snare 0.70 semitones down was matched at −1 and played 30 cents off. Search the cents (the speckle) and play them (Detune).

## Lining up

- **The sign of a lag.** `ncc(t, c)` returns d when t is c delayed by d; align with `shift(c, +d)`, or read the window at `s − d`. Using the opposite sign doubled every misalignment, and the averaged templates blurred. There is a test for it (`tests/test_xcorr.py`).
- **Different onset definitions on the two sides.** Library files were trimmed at their first rise above −30 dB, templates at an onset-strength peak. Samples with a quiet lead-in ended up 12–21 ms apart. The true snare scored 0.15 instead of 0.85. Align templates on the attack and search ±30 ms.
- **The same fix in the wrong place.** Moving library files to the attack point too made things worse (27/30 → 14/30), because bleed changes where a template's attack seems to be. One side robustly placed plus a wide lag search worked; both sides re-placed did not.
- **An attack read in the wrong band.** In the full band a snare's attack sat on the kick or 808 before it: all six calibration snares were placed 13–24 ms early. Read a sound's start in its own role's band.

## Windows

- **Patterns live inside windows.** Everything that always follows a hit is in its average: the kick two sixteenths after the snare, the hat on every eighth. A 300 ms window made the following kick decide the snare's detections (352 ms early) and its matches. Use the attack (40–80 ms) to detect, a little more (80–200 ms) to identify.
- **NCC ignores level.** The quiet tail of a low hit matches its own template; a 55 Hz kick was found again 183 ms after every hit, and a tonal snare's template matched its own ring 101 times for 32 snares. Drop hits no louder than the template's own ring at that lag, where nothing starts.
- **A low tone correlates with itself a period later.** Below 200 Hz, keep hits at least 100 ms apart.

## Grouping and naming

- **The same sound split into groups.** Overlaps differ from hit to hit (hat or no hat on a kick), so one sample lands in several groups. Merge groups whose averages match.
- **Different sounds merged as "duplicates".** A snare that always lands on a hat shares every moment with the hat and looks like it in the hat band. Require the same waveform in both bands before calling two templates one sound.
- **Naming by energy fails.** Bright snares and dull hats overlap; an 808 under a snare makes it look like a kick. Name sounds after matching (the library's file names vote), and keep the discovery band only as a hint.
- **A role from the wrong band.** The kick band does not tell recordings apart: an open hat's top match there was an fx file at 0.957, and its attack was then read on the next kick, 220 ms later. Let a close match name the role only in the snare and hat bands.
- **One template, two sounds.** Hand identities out from the strongest match down, so a hat heard under a kick is not reported as the kick.
- **Guessing slides costs more than it gains.** A rule that called every unattacked 808 pitch move a slide was right 6 times in 79. Merging those notes into the note before scored better on every 808 number.
- **A kick inside a held 808 note** reads as a short low note followed by the held note found again. Read it as the held note going on.
- **A note without an attack is not always the note before.** Merging unattacked 808 notes bridged a 1.0 s and a 7.2 s silence on PIRAMIT. An 808 alone after a rest had its onsets read 27 ms before its envelope's feet, and no kick to anchor them, so four notes became one held for 15.3 s. Check a held note against the stem's level.

## Libraries

- **Kits copy each other.** The same file appears under several names, and at other pitches. Treat > 95 % correlation between two entries as one recording, or the margin to the "next" sound is the sound itself. Compare them as they were found: a kit's copy 0.22 semitones down tied the original at 0.84. Different kicks brought to one pitch agree where their tone is: a wrong kick 2 semitones down hid the true one as its "copy" (0.952 over the full band, 0.83 in the snare band).
- **A wider search finds closer neighbours.** Searched at any pitch, PIRAMIT's hat still matched its file at 0.984, but another kit's hat 3 semitones up reached 0.958. The margin fell from 0.078 to 0.025 and the verdict to "same family".
- **Names are not roles.** "808 Snare" is a snare, "Kick 808 Tone" may be a 808; role rules are ordered and only approximate.
- **Generated files match themselves.** Leave `AbletonMCP Generated` (and Live's backups) out of the inventory.

## Measuring the measurer

- **One number rewards what dominates it.** Correlation in the 808 band is the fundamental. A fit on it alone drove the distortion to zero (harmonics 25 dB off) at NCC 0.969. Score on every dimension you care about.
- **Pictures hide level differences.** On a 70 dB scale, a library 808 with harmonics 12–17 dB too strong looked like a match, and a snare's tail looked 75 ms longer than its file's where the levels differ by 9 ms at −40 dB.
- **Votes always have a winner.** Per-hit votes rank the top three candidates only: PIRAMIT's unmatched snare wins 26 of 27 votes for a file at NCC 0.47. The null test (−1.1 dB) is what says no.
- **Score where the sound starts.** The truth time of a hit is where its file starts, a detection is where the attack is. Removing one constant offset per sound hid that the v1 plans put the calibration snares 21–24 ms early (F1 1.00; strict F1 at 20 ms 0). Time both sides at the attack in the sound's own band, and remove nothing.
- **A mean over sounds is not a mean over hits.** 37 pitched copies of 1–68 hits each count as much as a main hat of about 170 hits. Report both.

## Merging

- **Measure the merged pipeline, not each part.** The joint fit was built on the old templates (one per 16th slot). On the attack-based ones, residual templates and variants of a sound were held under the main templates and found their hits a second time. On 22 beats the false hits went from 32 to 60 a beat; over the suite, open hats fell from 0.98 to 0.90.
- **A rule keyed on where a template was found.** match kept whole semitones for templates found in the low band, which used to be kicks. The new discovery finds snares there too, and drum_pitch-2's detuned snare came out 30 cents off.
- **A shared pad hides a pitch.** A template found twice, as the same file at another pitch, shared its twin's pad and played at the twin's pitch, 300 cents off.
- **A reused step reads stale inputs.** Re-running analyze while reusing melodic.json left the chords built on the old 808 notes. The suite marks such steps `reused_stale`.
- **Chords follow the sections, which follow the drums.** A change in the drum hits moved a section boundary on full-2 and broke its 8-bar loop: chords majmin 0.999 → 0.961, with the same notes.

## Live

- **Simpler loads at −12 dB.** Set 0 dB, or every level estimate is 12 dB off. "0.0 dB" as text lands on −0.005 dB.
- **Simpler's root is C3.** Transpose +36 for an 808 whose notes are written at C1 (MIDI 24).
- **Transpose and Detune keep whole numbers.** 11.99 semitones play as 11; a Detune of 11.5 plays as 11, the text "11.5 ct" as 12 (Live 12.4.6). Write whole semitones and whole cents.
- **Velocity is not a linear level.** dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over the Volume: velocity 127 plays 7.9 dB (35 %) or 10.2 dB (45 %) over it, and the amount depends on how the Simpler was made. The plan still assumes v/127.
- **Snap cuts the end.** On by default, it stops every sample 10–30 ms early.
- **The browser keeps an old file.** Loaded by a name Live has loaded before, a pad plays the old content after the file is overwritten; a file loaded by path is read again.
- **New files reach the browser after a few seconds;** retry, do not fail.
- **Locators at fractional beats were refused,** so round them.
- **A new audio clip starts at its first transient.** Live's warp analysis moves the start, 21 ms to 3.6 s in here, and it stays moved with warping off. Set `loop_start` back to 0, or the reference plays early against the notes.
- **The Trial cannot save.** Keep the plan, rebuild in seconds.
