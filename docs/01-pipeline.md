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

### Hits (`detect.detect`)

**Matched filter.** Each template's first 40–80 ms (the attack, see below) slides over the whole drum stem in its band (`xcorr.sliding_ncc`). Peaks above 0.6 (0.65 for hats) are hits; the least-squares gain at the peak is the hit's level.

Three cleanups:
- **Quiet tails.** Hits below 25 % of the template's median gain are dropped. NCC ignores level, so a ringing tail matches its own template too.
- **Low band.** Hits are at least 100 ms apart, and a hit less than half as loud within 250 ms after a louder one is its tail.
- **Overlaps within a band.** A moment claimed by two templates goes to the better match.

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
- **Pads** follow GM (kick 36, snare 38, hats 42/44/46, …). A `same_as` template shares its twin's pad.
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
