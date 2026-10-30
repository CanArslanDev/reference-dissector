# 11 – A new beat in a song's style: what went wrong, and the checks that came of it

The second study (HAYIR DEMELİYDİM) did not rebuild the song. It wrote a new beat "by the same producer, on the same album": the same sounds, effects and groove, but new notes. Two complete beats were built in Live, mixed and measured, and the user rejected both by ear. This page records why, what each step cost, and what changed in the code because of it. As in [09](09-data-and-copyright.md), no audio, stems or transcriptions of the song are published; the numbers below are measurements.

## What the ear caught and the numbers did not

| Round | Verdict | What the measures said | What was wrong |
|---|---|---|---|
| v1 | "the drums hit unrelated to everything; the melody has nothing to do with the drums" | style 0.849 (the song's own halves: 0.850), section loudness within ~1 dB | no ducking under the kick; hats and percussion as loud as kick and clap; one-shots cut short by their notes; invented fills |
| v2 | "the intro does not pull you in like the song; the groove and B are flat" | style 0.856, ducking curve within ~1 dB of the song's | the parts themselves: written to fit transcription statistics that turned out to be noise |

### 1. Ducking, balance and one-shot length ([`groove`](../src/refdissect/groove.py))

- **Ducking.** Around every kick, the song's 808 falls ~20 dB within 20 ms and ~45 dB by 200 ms, back by ~500 ms; its chords fall ~13 dB and are back by 270 ms. v1 did not move. `kick_curve` measures this from any stem or recording, given the kick times. A Live Compressor keyed from the kick track (ratio 20, attack 1–5 ms, release 60–200 ms) was then chosen by recording a grid of settings and comparing curves (`curve_distance`); the chords came within 1.2 dB of the song's curve.
- **Balance.** Matched as one group, the drums came out with hats and percussion level with kick and clap; in the song they sit 19 and 31 dB under them. Each role's median hit peak, measured in its own band (`hit_levels`, `role_offsets`), gives the fader moves (`balance_moves`).
- **One-shot length.** With one-shot mode off, Simpler ends the sound with the note: one-16th kick notes cut a 280 ms kick at 130 ms (`cut_one_shots` lists such notes).

### 2. Effects before sounds ([`space`](../src/refdissect/space.py))

Every chord candidate lost by ear while it was compared dry ("the song's chords come from far away"). The space has to be measured first:
- L/R correlation per octave (`band_correlation`) bounds the reverb's share;
- the tail after a stop (`tail`) gives the drop and the reverb time;
- a section that sounds like a second, thinner pad can be the same chords through a band-pass: `fit_bandpass` finds the high- and low-pass between two sections' spectra. Here it was 900 Hz to 3 kHz, within 3.1 dB.

### 3. Statistics have to survive the same pipeline ([`rhythm`](../src/refdissect/rhythm.py))

The new parts were written to match the song's layer statistics from Basic Pitch: notes per bar, triplet share, how many distinct bars there are. After v2 the new beat went through the same separation and transcription, and its plain, straight block chords scored a triplet share of 0.31 (the song: 0.38) and 56 distinct bars (the song: 52). On reverb-heavy layers those numbers measure the transcription, not the music. Only coarse ones separated the two: notes sounding at once (3.4 against 1.5) and median note length.

**Rule:** run any statistic on your own render through the same path before matching it.

Onsets read in the layer's register at 48 points per bar (`onset_profile`) were reliable where notes were not. They showed the intro's bass and riff hitting together, with the harmony moving inside the bar. Neither v1 nor v2 did that.

### 4. Choose parts by ear, in short sketches, before building

Sounds were chosen by ear in two-part files (song → beep → candidate, [`listening.ab_file`](../src/refdissect/listening.py)), and that worked: drums, 808, riff and chords were approved that way. Parts were never auditioned. Both full builds carried notes nobody had heard. The order now is:
1. Measure the structure (rhythm, harmonic rhythm, ducking).
2. Write several 8-bar sketches per section.
3. Let the listener pick.
4. Only then build and mix the whole beat.

## What it cost

About 7.7 hours of active work (228 minutes of tool time, the rest writing and deciding):

| Step | Time | Why | Now |
|---|---|---|---|
| Mix loop | ~135 min, 60 % of tool time | Every fader change recorded 13 tracks × 100 bars in real time (~3 min), then again to check: 14 rounds | Record stems once; decide gains on them ([`mixsim`](../src/refdissect/mixsim.py)); record again only after a non-linear change. ableton-live-mcp's `capture_stems` now records the section asked for (it played from wherever the playhead was). |
| Probing 1,856 Live presets | ~35 min | Every preset loaded and recorded; browser ids went stale at preset 300 and the run restarted | CLAP shortlist first (30 presets: 1 min). Browser ids no longer go stale (Remote Script 2.0.1). |
| 808 fit | ~10 min | ~24,000 loss evaluations, half of each in a 25 ms moving mean | 2× faster with the same result. |
| Rebuilding the set | ~4 min each time | The Trial cannot save; Live restarted twice | – |

## The commands

- `refdissect compare REF_STEMS REBUILD_STEMS`: every measure on both stem sets (separate the rebuild with the same model first), flagged where the difference is larger than between the reference's own halves. On the v2 beat it put the 808's and the chords' kick curves 4.0 and 4.2 dB from the song's. Kicks come from the separated drums, which can hold other attacks; there, Demucs put the 808's entry (two 16ths after each kick) into the drums, so `--kick-gap 0.3` was needed.
- `refdissect listening-round REF CANDIDATES... --question ...`: one numbered A/B file per candidate and a manifest.
- `refdissect space STEM --spans a-b c-d`: width per octave, and the band-pass from the first span to the second.
- `refdissect mix-gains STEMS --targets targets.json`: the gain per recorded stem that reaches a target level.

## Smaller findings

- Basic Pitch returns no notes at all when its minimum frequency is under its lowest note, A0 (27.5 Hz): 25 Hz gave 0 notes on a bass stem, 30 Hz gave 232. ableton-live-mcp now raises the floor.
- Basic Pitch's posteriorgram frames drift 0.8 s by the end of a song if indexed as t × 22050 / 256 ([`basicpitch`](../src/refdissect/basicpitch.py)). Read on Basic Pitch's own frame times, its onset posterior at its own note starts on this song was 0.45–0.47; read the naive way, 0.29 falling to 0.21. Those frame times are still up to 18 ms off inside each block of 142 frames, so notes are now timed block by block.
- Recordings clip at 0 dBFS without a warning (Simpler at 0 dB is +12 dB over its default); `capture_stems` now reports clipped stems.
- `grid.tuning` was wrong by up to 60 cents (see [01](01-pipeline.md#tuning-gridtuning)); the PIRAMIT case study is corrected.

## What the song taught replica mode

The `melodic` step and the drum discovery of replica mode ([01](01-pipeline.md)) were checked on this song's stems too. There is no truth for it, so these are counts, not scores:
- **A stored tuning can be stale.** This song's `analysis.json` predated the stable-peak estimator: +21 to +35 cents where its stems measure −2 to −5. Taking those values out moved in-tune stems off A440, and the other stem kept 165 of its 520 notes. `melodic` now measures each stem's tuning itself.
- **The 808's range as a floor, by time, not by notes.** Notes under the 808's range are left out of the stems it bleeds into. The highest 808 pitch heard twice put that floor at E3: 266 of the song's 273 808 notes are at or under MIDI 43, but two sit at MIDI 51. The pitch under which the 808 spends 95 % of its time puts the floor at G#2, and the guitar keeps 157 notes instead of 128 (159 with no floor).
- **A vocal keeps its low notes.** 458 of its 768 notes lay under the first rule's floor; the floor now applies to other, piano and guitar only.
- **Rare drum templates grow with a song's length.** With 3 hits enough anywhere, the residual pass made 7 templates on this song; with one hit per 16 bars, 5.
