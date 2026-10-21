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
2. **Where onsets start.** Each onset-strength peak of a band is moved back to where its sound starts (`hits.rise`): from the quietest point in the 30 ms before it, the first point where the 1 ms envelope reaches 30 % of its peak. The low band also reads the snare band and looks back 45 ms: on htdemucs drums a kick's body comes 26 ms after its click, the rest of it went to the bass stem. What still rings from the hit before is left out.
3. **Places in the beat.** The starts are clustered by where they fall in the beat (±12 ms), densest first. Straight or swung 16ths, triplets and 32nds are places of their own wherever the song puts them, in any meter. Per place, hits are grouped greedily by waveform (`hits.group`, NCC ≥ 0.6). A group needs 3 hits, or one per 4 bars in a longer song. With 8 hits per 16th slot within ±12 ms, an 8-bar beat at 95 BPM got no template at all, and a rim only on swung 16ths (62 %) none.
4. **Merge** groups within a band whose averages match (NCC ≥ 0.8 over 100 ms).
5. **Hits that start something.** Each candidate's template slides over the stem. One whose hits sit on an onset less than 30 % of the time, or rise in the hat band less than 5 % of the time, rings rather than starts: an 808 matched once a period in the snare band, or a kick's body after its click.
6. **One template per sound.** Candidates are taken in order of clean hits (NCC ≥ 0.8; squared, over all their hits). A candidate is the same sound as a kept one that finds most of its hits when their gains move together (r ≥ 0.6 for the same guessed role, 0.8 otherwise), or when, in the band of its role, the kept template has its shape at its level and either the two find each other's hits or they guess the same role and the kept one holds its level where it was found too.
   - A kick under some of the hats stays a kick: the hat template holds it 6–11 dB down in the low band.
   - A crash on every kick stays a crash: the kick does not hold it in the hat band.
   - A view that finds 3 hits or more that nobody else does (a kick in the hat band where an 808 hid its low band) is kept as `same_sound_as`, and the hits it shares are counted once (`count_once`).
   - Of two candidates of different roles locked at one lag of 15–60 ms, the quieter goes: on two beats something 42 ms before every snare that was no hit.
7. **Residual pass.** Onsets that no template of their band explains (no hit within 10 ms) are grouped wherever they sit, 3 hits or more, and go through 5 and 6: rare sounds, crashes, fills.
8. **Template.** Each survivor's hits are lined up with the sound's start 5 ms into the window (`hits.aligned_template`), read in the band of its guessed role: the snare band leaves out the kick or 808 before a snare. They are averaged, or their median taken in the hat band (on 30 calibration beats the median made hat templates 6.5 dB cleaner against their true files). analysis.json gives each template's `attack_s`, `same_sound_as` and which pass `found` it, and each hit its `t_attack`.
   - In the full band the attack sat on the kick or 808 before the snare: all six calibration snares were placed 13–24 ms early. On 32 suite beats (mean over 14 scenarios of each sound's median error) the plan's snares moved from 11.6 ms early to 1.4 ms late, its kicks from 9.6 to 2.4 ms late, and the analysis's kicks from 22.6 to 0.9 ms late.
9. **Names and roles.** Templates are named `sound_1`, `sound_2`, … by number of detections. The role is only a guess here (`kick`/`snare`/`hat`): the median over the hits of the share of energy each adds below 150 Hz and above 5 kHz over the 60 ms before it. Summed instead, the kicks under some hats decided: 33 of the 56 templates that played a hat on 32 suite beats read as kicks (8 of 84 now; 71 % of all guesses right, 57 % before). Matching decides it later.

On those 32 beats (14 scenarios, htdemucs stems), against the code before (16th slots ±12 ms, 8 hits a slot, the cross-band duplicate test, the full-band attack), strict F1@20 per role, mean over scenarios: kick 0.26 → 0.87, snare 0.85 → 0.96, hat 0.45 → 0.75, open hat 0.23 → 0.97, clap 0.61 → 0.67, crash 0 → 0.47 in the analysis; kick 0.17 → 0.80, snare 0.58 → 0.79, hat 0.50 → 0.72 in the plan (`results/calibration/suite-l1a-after2.json`). 88 % of the true sounds get a template (59 % before), at 1.2 templates each (1.4 before); the residual pass also keeps 2.2 templates a beat that play no true hit (on the v1 beats, transients of the melodic loop that htdemucs left in the drum stem).

### Hits (`detect.detect`)

**Matched filter.** Each template's first 40–80 ms (the attack, see below) slides over the whole drum stem in its band (`xcorr.sliding_ncc`). Peaks above 0.6 (0.65 for hats) are hits; the least-squares gain at the peak is the hit's level. Each hit carries `t`, 5 ms after the window starts, and `t_attack`, where the sound starts: the template's `attack_s` after it.

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
