# 01 – The pipeline, step by step

Every step reads files and writes JSON (and, locally, audio). The order:

```
separate ─▶ analyze ─▶ match ─▶ fit-808 ─▶ plan ─▶ live-build
             │          │         │          │
     analysis.json  matches.json synth808.json plan.json, midi/
     templates/                  808 (synth).wav
```

All audio is processed as mono float32 at 44.1 kHz (`audio.read`).

## separate

Demucs (`htdemucs`, or `htdemucs_6s` for guitar/piano stems) runs in its own uv environment (`workers/separate_worker.py`), the same way [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp)'s `separate_stems` tool does: torch never loads into the main process, and the memory is gone when the job ends.
- On an M-series Mac, a 30 s beat separates in about 5 s on MPS once cached; a 129 s song with `htdemucs_6s` takes about 15 s.
- The guitar stem of the 6-stem model came out nearly empty (−80 dB) on Turkish trap.

The rest of the pipeline only needs `drums.wav` and `bass.wav`. `other`, `piano` and `vocals` are used for tuning and as reference tracks in Live.

## analyze

### Grid (`grid.fit_grid`)

Produced music sits on a fixed grid, so the grid is fitted rather than tracked:
- **Score.** For each tempo T and phase φ, the score is the mean onset strength (librosa, hop 128 = 2.9 ms) at φ + k·60/T.
- **Search.** Tempos from `--bpm-range` (default 100–180) in 0.05 BPM steps, then ±0.1 BPM around the three best in 0.002 BPM steps, with phase in half-frame steps.
- **Speed.** About 2 s for a 2-minute song.
- **Half and double time** score almost alike, since every other grid line carries a hit too. The range decides. For trap, 100–180 counts the hats' grid, not the half-time feel. The snare then falls on beat 3, which is the check the reference prompt asked for.
- **Check.** `grid.offsets` reports, per beat of the bar, how far the strongest snare-band onset sits from the grid line (median and p90). A right tempo stays within a few ms over the whole song; a wrong one drifts. On PIRAMIT, 137.00 BPM gives p90 4.6–11.8 ms, and 136.36 BPM drifts from +29 ms to −49 ms.
- **Phase.** The phase marks onset-strength peaks, which come a few ms after the attack (about 2–8 ms). Detections carry their own times, so this offset does not matter downstream.

### Tuning (`grid.tuning`)

Stable spectral peaks on each pitched stem (bass, other, piano, vocals, guitar), over 60 s: every frame's peaks (16384-point FFT, parabolic interpolation) that stay within ~5 cents for five frames, their offsets from equal temperament averaged on the circle and weighted by level. The consensus is the median.

The first version used `librosa.estimate_tuning`. On synthetic tones it was off by up to 60 cents (it histograms every bin, sidelobes and noise included), and it put all of PIRAMIT at +0.28 semitones. The peak method stays within 1 cent of synthetic tones and puts PIRAMIT's melodic stems at A440 (other −2, piano +1.5 cents; pyin agrees, +2 to +3).

The 808 has a tuning of its own (`detect.pitch_offset`, the circular mean of its notes' pitches): PIRAMIT's notes sit at +33 cents (agreement 0.64). An 808 sample is often detuned by itself. Its notes are rounded with that offset, and `fit-808` renders the one-shot at it. Without removing it, the notes measured "C or C#" one by one.

### Drum templates (`detect.discover`)

1. **Bands.**
   - `low` (< 200 Hz) finds kicks under hats and snares.
   - `snare` (120 Hz–14 kHz).
   - `hat` (> 3.5 kHz).
2. **Onsets on the grid.** In each band: onset-strength peaks within 12 ms of a sixteenth-note grid line.
3. **Group per slot.** Per grid slot (16 per bar), hits are grouped greedily by waveform (`hits.group`). Each hit joins the group whose running average it matches best (NCC of the first 40–80 ms) if that reaches 0.6. Groups with at least 8 hits are kept.
4. **Merge** groups within a band whose averages match (NCC ≥ 0.8 over 100 ms).
5. **Drop duplicates across bands.** A candidate whose detections mostly coincide (≥ 60 % within 15 ms) with a larger candidate's, *and* whose waveform matches it (NCC ≥ 0.6) in *both* candidates' bands, is the same sound found twice. Coinciding alone is not enough: a snare that always lands on a hat shares every moment with the hat and looks like it in the hat band.
6. **Template.** Each survivor's hits are lined up and averaged (`hits.onset_template`) with the attack placed exactly 5 ms into the window.
   - The attack is the first point where the 1 ms envelope reaches 30 % of the peak.
   - This matters for matching: hit times from onset-strength peaks sit several ms off the attack, and a template 12 ms off matched its true sample at NCC 0.15 instead of 0.85.
7. **Names and roles.** Templates are named `sound_1`, `sound_2`, … by number of detections. The role is only a guess here (`kick`/`snare`/`hat`, from the energy the hits add over the 60 ms before them). Matching decides it later.

### Hits (`detect.detect`, `layers.explain`)

**Matched filter.** Each template's first 40–80 ms (the attack, see below) slides over the whole drum stem in its band (`xcorr.sliding_ncc`). Peaks above 0.6 (0.65 for hats) are candidate hits, 8 ms apart at least, so a flam's grace note 10–30 ms before its hit and a 64th roll (27.6 ms apart at 136 BPM) stay apart. A tone's candidates are 35 ms apart (100 ms in the low band). The stem is padded: a hit at 0 s is found too, and one whose attack the stem cuts off plays from 0 s.

**Every moment at once.** Keeping one template per moment and band lost nearly every hat under a snare (416 of 432 on 22 suite beats). Instead, each candidate's neighbourhood (±150 ms) is fit as a non-negative sum of all the candidates in it (NNLS) on the kick, snare and hat bands, each band weighted by 1/RMS of the stem in it so that each counts alike. Each hit gets its own gain, the hit's level.
- **One sound found twice.** Two candidates whose placed templates correlate at 0.6 or more are the same sound (a duplicate template, a kick ringing one period on): the one that explains more of the stem keeps it. Templates of one kind that lose two thirds of the fewer's hits to each other are merged, so a hat found in the hat band and in the snare band plays once; the merged template's pitched copies go with it. At one half, a clap on every other snare, its template and the snare's each holding half the other, lost half its hits to the snare and was merged into it: 0 of 32 claps. Of the 15 merges on the suite, 13 lose 0.67–1.0.
- **Held sounds.** A template that holds another (the snare template holds the hat on every snare) has it subtracted first; otherwise the snare takes the hat's place at every hit. The held template's coefficient is 0.3–2.0, read both over all bands and in its own band (the smaller is subtracted), it takes 1.5–30 % of the holder's energy over its span, and it starts up to 30 ms before or after the holder. On the 48 suite beats: 50 holds, coefficients 0.32–1.25, shares 1.8–30 %, snare templates holding a hat at −19 to +29 ms. Over all bands only, a hat's average, which carries the body of the kicks some hats sit on, made a kick template hold a hat it never plays (full-2: 0.61 over all bands, 0.06 in the hat band); the purified kick then held a negative hat, and a hat was found under the kicks that have none (24 such hats on 3 beats, now 0).
- **Composites.** A template the others explain to 80 % is left out, and its hits go to them.
- **Keeping a hit.** Its gain reaches 25 % of the template's median gain, and the fit loses more without it than the template takes at shifted, empty windows (their 95th percentile).

**Where the kept hits leave something out.** A held template is looked for at its offset under every hit of the template that holds it. Every noisy template (not a tone, not an echo of another) is looked for where its matched filter peaked at NCC 0.3 up to its threshold, and at the onsets of its band. Such a hit needs half the template's median gain (a quarter where what the other hits leave matches the template at its threshold), and the null. It is marked `source: "layer"`, with `under`, the template it was under. For these hits the null is a backstop: of 1,724 probe hits over the gain floors on the 48 beats, it alone rejected 44, all of sounds that no truth sound was assigned to. Two further floors were measured and left out. An NCC of 0.2 with what the others leave, in the template's own band, would remove 23 true probe hits for 13 false ones and 35 of unassigned sounds. A median gain of at least a quarter of the template's own level lost a sound that plays 26–30 dB under the few loud hits its template was made from: 0.75–0.80 → 0.12–0.14 of its hits found (synthetic).

**Tones.** A template with half its energy in the kick band (bands weighted as in the fit), or a quarter there that rings (NCC 0.5 with itself 8–35 ms later), is a tone. Its matched filter has a broad maximum, so it gets no pitched copies and is not looked for under other hits; a kick template's low ring explains a little of almost anything, so it is never held. A kick may ring little and still be a tone (leadin-2: 59 % in the kick band, ringing 0.29); probed as a noisy sound, it gave 65 layer hits, 40 of them false (as a tone, 15, all true). A hit of a tone (or of a low-band template) within 250 ms after a louder one is that one's ringing when it is less than half as loud, or below what the template's own decay holds at that delay (0.70 at 23 ms for a kick on v1-1, which was found again 15–23 ms after every strike). A tone found in the hat band with less than half its energy in the kick band is a hat whose average carries the body of the kicks under some of its hits: it is held, spaced and kept as a hat (not probed).

**Pitched copies.** A noisy template (its copy one semitone off matches it at NCC < 0.5: hats, snares) is also matched 1–5 semitones up and down, transposed as Simpler plays it. Its hits carry `semitones` and are scored as sounds of their own.

A quieter hit at most 35 ms before another of the same sound (no third within 70 ms) is a flam's grace note (`flam_of`).

On the 22 suite beats of the layers, rolls, flams, dynamics, v1, v1orig, producer, full and kick808 scenarios (htdemucs stems), against one template per moment and band (tags m1 → l1b-fix):
- hats within 20 ms of a snare or clap: 16 → 324 of 432 found (of a snare: 16 → 292 of 358);
- drums, strict F1 at 20 ms (mean over the scenarios): 0.421 → 0.584 at analysis, 0.282 → 0.440 in the plan; at 50 ms 0.503 → 0.665 and 0.321 → 0.531. Over hits (every truth and estimated hit counting alike): 0.613 → 0.781 and 0.602 → 0.730;
- hats 0.501 → 0.796. That mean counts each of 37 pitched-copy sounds (1–68 hits, 193 in all; 0 → 80 found) as much as a main hat of about 170 hits; the plain hats alone: 0.674 → 0.837;
- 64th hat rolls: 19 → 50 of 64 hits; pitched roll hits: 23 → 68 of 84;
- false hits per beat, one-to-one at 20 ms (a sound's second hit on one moment and every hit of a sound that no truth sound was assigned to count): 102 → 32 at analysis, 65 → 28 in the plan. Hits more than 20 ms from any truth hit of their sound: 59 → 30 at analysis;
- one strike found twice (two kick hits under 35 ms apart, at most one truth kick within 60 ms): 1 → 5;
- the plan's level, median |level against a perfect kit|: 7.16 → 5.80 dB.

Regressions on these beats:
- full-1's main hat (169 hits): F1@20 0.310 → 0 at analysis. A perc template on its hats became a composite of a snare-band template, which was merged into a hat-band one that matching calls a kick; that template's attack is read in the kick band, 143 ms in. Its pitched copies move full's hat timing p90 from 0.3 to 14.2 ms, and full-1 loses 9 of its 45 swung hits (70 → 63 of 92 in all). In the plan its hat improves (0.555 → 0.755).
- Kicks at 20 ms: still 0 on v1 and flams, whose kick templates attack 20–50 ms off (template building). At 50 ms: v1-1 0.723 → 0.991, dynamics-1 0.737 → 0.983. Before the ringing rule, a kick found in the snare band gave a second hit per strike (68 such pairs on these beats), and those low-NCC hits kept the template under the plan's pad rule (half its hits at NCC 0.8 or more: 0.39–0.47). With a pad, 6 of the 48 beats get 54–66 kick notes each that are right at 50 ms and false at 20 ms: the plan's false hits per beat are 28 against 20 in the lane's previous run (l1b-after3), and its kick F1@50 0.585 against 0.421.
- The plan's level on full (6.63 → 12.26 dB) and producer (1.66 → 4.62 dB): pitched-copy sounds now matched bring the hat's level error into the median.

On the other 26 suite beats (13 scenarios): drums strict F1@20 0.438 → 0.492 at analysis, 0.345 → 0.365 in the plan; layer hits 317 true at 20 ms, 31 false, 137 of sounds no truth sound was assigned to. 95 hits of one of those (melodic-1, 80 found under other hits) come from a template whose median gain is 0.088 of its own level; they play on a hat pad and raise the plan's hat F1@20 there 0.306 → 0.418 (precision 1.0 → 0.53). Kicks at 20 ms regress on meter34-2 (0.737 → 0.286), rare-1 (0.167 → 0; at 50 ms 0.518 → 0.927, one hit per strike) and melodic-1 (0.317 → 0, that template no longer assigned to the kick). In the plan, bass_low (0.277 → 0.157), glides (0.514 → 0.491) and leadin (0.386 → 0.369) fall: the plan places notes at a template's window time.

It costs about 4 s per 30 s beat: the suite's analyze step takes 10.0 s per beat against 6.2 s with one template per moment (m1) on the 22 beats, 8.7 s against 5.4 s on the other 26. Not found:
- a sound that always plays with another has no template of its own (the layers scenario's clap on every snare);
- a hat 14 dB under a clap's top end (v1orig-4);
- a hat under a snare 9 dB or more below its usual level (synthetic: 24 of 24 at −6 dB, 0 of 24 at −9 dB);
- a clap on every other snare at a constant offset when its template and the snare's each hold half the other: its 16 solo hits, not the 16 on the snare;
- a tone ringing longer than 250 ms (a 50 Hz kick with a 250 ms decay is found again 340 ms on, once per strike);
- closed hats in a long open hat's decay (0.69 of them at a 350 ms decay, synthetic); one that always sits there puts its decay into the closed hat's template, which then finds one closed hat too many per beat.

**Why the attack only.** Patterns repeat, so whatever always follows a hit is in its average too: the kick two sixteenths after a snare, for example. With a 300 ms window, that louder kick decided the matched filter: the snare's "hits" landed 352 ms early.

`peak_ncc_histogram` shows each template's peaks per 0.1 of NCC. A clean template shows a gap: PIRAMIT's even-bar snare has 27 peaks at ≥ 0.9, one in 0.8–0.9 and nothing between 0.3 and 0.8.

### 808 notes (`detect.notes_808`)

1. **Onsets** in the bass stem below 1.5 kHz.
2. **Pitch.** pYIN pitch over 60–450 ms after each onset (or until the next one); the median is the note.
3. **Rounding.** The track's tuning is removed before rounding to a MIDI note.
4. **End.** Where the 25 ms envelope falls 35 dB below the note's peak, or at the next note.
5. **Slide.** The pitch at the end minus the pitch at the start; `|slide| ≥ 1.5` semitones marks a slide.

`detect.isolated` picks the notes of the commonest pitch that ring at least 600 ms alone. These are what the 808 sounds like without the next note cutting it off; the fit uses them.

### Sections (`structure.sections`)

1. **Per-bar signature.** Every bar gets the set of sounds heard in it: templates with at least 16 detections, plus the 808.
2. **Smoothing.** Each sound's presence is smoothed by a 3-bar majority, so one extra perc does not start a section.
3. **Runs.** Runs of equal signatures shorter than 2 bars join the neighbour with the closest signature.
4. **Names.** Equal signatures get the same letter, so a returning section reads A … A.

## match

`library.Library` holds the first 0.6 s of every one-shot (1 s for 808s):
- **Trimmed** to 2 ms before it first rises above −30 dB of its peak, and peak-normalized.
- **Spectrum.** Each entry also gets a semitone-band spectrum: log energy per semitone from 40 Hz, z-scored, over the first 100 ms (400 ms for 808s).
- **Size.** 8,854 one-shots are about 470 MB in memory as float16.

For each template, in each of the kick, snare and hat bands (150 ms for snare, 80 ms for hat, 200 ms for kick):

1. **Waveform pass.** NCC against every entry at 0 semitones, vectorized through the FFT with the band filter applied to the spectra, over ±30 ms of lag. 1–3 s for 8,854 entries.
2. **Pitch-invariant shortlist.** The template's semitone spectrum is compared with every entry's at every transposition at once: transposing by k semitones shifts the spectrum by k bands. The best 200 entries, with their best transposition and its neighbours, are tried as waveforms.
   - Without this, a snare pitched down a semitone never reached the top 200 at 0 semitones (NCC 0.15 there) and was never tried at −1, where it matches at 0.95.
3. **Distinct.** Copies of a higher-ranked sound under another name are removed: entries correlating > 0.95 with each other. Kits copy each other's sounds.

**Handing out identities.** All (template, band) pairs are sorted by their best NCC and handed out strongest first:
- A sample already given to another template goes to this one too only if both match it at ≥ 0.9. That marks it `same_as`: the same sound found twice.
- Otherwise the template takes its best match in another band.
- This keeps a hat that plays on every kick from being reported as the kick.

**Evidence for the top three:**
- **Per-hit votes.** Which candidate matches each individual hit best, over its first 70 ms. A real match wins nearly every hit, not just the average.
- **Null test.** What is left of the template after subtracting the candidate, aligned at the best gain, in dB.

**Role.** The majority role of the top five library names if the best NCC is ≥ 0.75; otherwise the guess.

**Verdicts** (`library.verdict`), from the calibration:

| Role | Verdict | Condition |
|---|---|---|
| snare, clap, hat, perc | same sample | NCC ≥ 0.95 and margin ≥ 0.03 |
| | candidate: maybe the same sample, processed (check by ear) | margin ≥ 0.05, ≥ 80 % of hits won, NCC ≥ 0.8 |
| | same family | NCC ≥ 0.85 |
| | no close match | otherwise |
| kick, 808 | same sample (likely) | NCC ≥ 0.97 and margin ≥ 0.05 |
| | same family | NCC ≥ 0.9 |
| | no close match | otherwise |

Kicks and 808s are mostly decaying sines, so every one correlates highly with every other at the same pitch. Their margins stayed below 0.08 even when right (see [03](03-calibration.md)).

`--detune X` also tries every sound shifted by X semitones. The idea was a track sped up by its tuning offset; on PIRAMIT it made the drums match worse, which showed the drums were not shifted.

## fit-808

1. **Reference.** The isolated root notes of the 808 are averaged from drums + bass. Demucs puts part of the 808's attack in the drum stem; the bass stem alone made the fit choose a 190 ms attack.
2. **Fit.** The voice in [05](05-808-synthesis.md) is fitted by differential evolution (scipy; popsize 20, up to 150 generations, about 2–3 minutes). The score is waveform NCC + 0.1 × harmonic error (2nd–12th, /20 dB) + 0.1 × envelope error (/20 dB).
3. **One-shot.** A 3 s sample at the track's pitch, at −1 dBFS with a 10 ms fade, is written for Simpler.
4. **Comparison.** With `--library-808`, the best of the library's 808s (±12 semitones) is compared on the same measures.

## plan

`live.make_plan` turns the analysis into what Live gets.

**Drum pads:**
- **Which sounds.** A template gets a Drum Rack pad when it was detected at least 12 times and at least half its hits matched at NCC ≥ 0.8. It also needs either a library match ≥ 0.75, or to be a kick or snare: the backbone is played with the closest sound, however far.
- **Pads** follow GM (kick 36, snare 38, hats 42/44/46, …). A `same_as` template shares its twin's pad. A pitched copy (hits with `semitones`) gets the next free pad from 54 up: the same sample, transposed that much further. On the rolls, full and producer beats (6 beats) the plan's drums F1 at 20 ms is 0.452 with these pads, 0.157 with the copies played unpitched (runs l1b-after3, l1b-nopads3) and 0.145 for one template per moment (m1); 0.455 now. Over hits it is 0.726, 0.670 and 0.602 (0.732 now): the 37 pitched-copy sounds hold 193 of the 1,878 truth hits, so the pads are worth 87 of them (88 of the 193 found now). The mean over sounds counts each copy, of 1–68 hits, as much as a main hat.
- **Velocities** come from the hit gains: 127 at the 90th percentile.

**Notes:**
- Drum notes last up to 1 beat and end before the next note of their pitch.
- 808 notes keep their real MIDI numbers, and Simpler's Transpose is set to 60 − root (+36 for C1), so MIDI 24 plays the sample's own pitch.
- Times are converted to beats from the start of the audio: beat = seconds × BPM / 60. The reference stems start at beat 0 unwarped, so the notes line up with them by construction.

**Levels.**
- Our drums and 808 are rendered offline at unity gain (Simpler at 0 dB, velocity as linear level).
- Each is set to the RMS of the stem it replaces.
- Everything, the stems included, is 4 dB down for headroom.

**Locators** go at section starts, on whole beats.

`midi/drums.mid` and `midi/808.mid` (480 PPQ, the plan's tempo) are written too, for `import_midi_file` or any DAW.

## live-build

See [07](07-building-in-live.md). It refuses unless every track in the open set is empty. Empty tracks are reused and the rest created, so nothing is overwritten.

## calibrate

See [03](03-calibration.md).
