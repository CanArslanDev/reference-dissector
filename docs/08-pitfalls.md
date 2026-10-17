# 08 – Pitfalls (all of them were hit here)

## Time

- **Half time and double time.** A beat tracker reported 68.18 BPM for a 137 BPM track. Both fit, because every other grid line carries a hit. Decide explicitly with `--bpm-range`, then check that the snare sits on the beat the genre puts it on (trap: beat 3).
- **An average tempo hides drift.** 136.36 BPM was 0.47 % off: within ±2 ms at the start, 49 ms off by the end. Fit the grid and look at the *last* bars (`grid.offsets`).
- **Seconds versus beats.** MIDI notes go into Live in beats. Convert with the fitted tempo (beat = s × BPM / 60), and place the stems unwarped at beat 0 so both share a clock. A MIDI file's own tempo is not applied by Live's import: ableton-live-mcp's `import_midi_file` reads beats.
- **The grid phase is not the attack.** Onset-strength peaks come 2–8 ms after the attack. That is fine for a grid; it is not fine for lining up waveforms.

## Pitch

- **An 808 not at A440.** PIRAMIT's 808 sits at +33 cents while the rest of the track is at A440; it made every 808 note "C or C#". Measure the 808's own offset and remove it before rounding.
- **`librosa.estimate_tuning` is not reliable.** Up to 60 cents off on synthetic tones; it read PIRAMIT as +0.28 semitones throughout. Check any tuning method on tones of known pitch first.
- **Sped up, or tuned?** Same pitch shift, different cause. Test it: shift every drum sample by the offset and see whether matches improve. On PIRAMIT they collapsed (0.86 → 0.25), so the drums were not shifted.
- **Transposed samples decorrelate completely.** −1 semitone: 0.95 at −1, 0.15 at 0, 0.13 at ±0.25. Never assume 0, and do not shortlist candidates by their score at 0.

## Lining up

- **The sign of a lag.** `ncc(t, c)` returns d when t is c delayed by d; align with `shift(c, +d)`, or read the window at `s − d`. Using the opposite sign doubled every misalignment, and the averaged templates blurred. There is a test for it (`tests/test_xcorr.py`).
- **Different onset definitions on the two sides.** Library files were trimmed at their first rise above −30 dB, templates at an onset-strength peak. Samples with a quiet lead-in ended up 12–21 ms apart. The true snare scored 0.15 instead of 0.85. Align templates on the attack and search ±30 ms.
- **The same fix in the wrong place.** Moving library files to the attack point too made things worse (27/30 → 14/30), because bleed changes where a template's attack seems to be. One side robustly placed plus a wide lag search worked; both sides re-placed did not.

## Windows

- **Patterns live inside windows.** Everything that always follows a hit is in its average: the kick two sixteenths after the snare, the hat on every eighth. A 300 ms window made the following kick decide the snare's detections (352 ms early) and its matches. Use the attack (40–80 ms) to detect, a little more (80–200 ms) to identify.
- **NCC ignores level.** The quiet tail of a low hit matches its own template; a 55 Hz kick was found again 183 ms after every hit. Drop hits far below the template's usual level, and suppress tails behind louder hits.
- **A low tone correlates with itself a period later.** Below 200 Hz, keep hits at least 100 ms apart.

## Grouping and naming

- **The same sound split into groups.** Overlaps differ from hit to hit (hat or no hat on a kick), so one sample lands in several groups. Merge groups whose averages match.
- **Different sounds merged as "duplicates".** A snare that always lands on a hat shares every moment with the hat and looks like it in the hat band. Require the same waveform in both bands before calling two templates one sound.
- **Naming by energy fails.** Bright snares and dull hats overlap; an 808 under a snare makes it look like a kick. Name sounds after matching (the library's file names vote), and keep the discovery band only as a hint.
- **One template, two sounds.** Hand identities out from the strongest match down, so a hat heard under a kick is not reported as the kick.

## Libraries

- **Kits copy each other.** The same file appears under several names. Treat > 95 % correlation between two entries as one recording, or the margin to the "next" sound is the sound itself.
- **Names are not roles.** "808 Snare" is a snare, "Kick 808 Tone" may be a 808; role rules are ordered and only approximate.
- **Generated files match themselves.** Leave `AbletonMCP Generated` (and Live's backups) out of the inventory.

## Measuring the measurer

- **One number rewards what dominates it.** Correlation in the 808 band is the fundamental. A fit on it alone drove the distortion to zero (harmonics 25 dB off) at NCC 0.969. Score on every dimension you care about.
- **Pictures hide level differences.** On a 70 dB scale, a library 808 with harmonics 12–17 dB too strong looked like a match.
- **Score against the right truth.** The truth time of a hit is where the file starts, a detection is where the attack is. Remove a constant offset before computing F1, or a perfect transcription scores 0.

## Live

- **Simpler loads at −12 dB.** Set 0 dB, or every level estimate is 12 dB off.
- **Simpler's root is C3.** Transpose +36 for an 808 whose notes are written at C1 (MIDI 24).
- **New files reach the browser after a few seconds;** retry, do not fail.
- **Locators at fractional beats were refused,** so round them.
- **A new audio clip starts at its first transient.** Live's warp analysis moves the start, 21 ms to 3.6 s in here, and it stays moved with warping off. Set `loop_start` back to 0, or the reference plays early against the notes.
- **The Trial cannot save.** Keep the plan, rebuild in seconds.
