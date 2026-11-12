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

### 5. Sketches drawn from the plan ([`candidates`](../src/refdissect/candidates.py))

Step 2 is now a command: `refdissect candidates WORK --out DIR --n 6 --seed 1 --section A`. It reads the plan's statistics and draws new 8-bar beats from them. It never copies the plan's notes. The statistics come from the bars of the named section (every bar without `--section`):

| What | Read from | Drawn as |
|---|---|---|
| tempo, meter | the plan | the same |
| swing | `analysis.json`'s `grid.swing`, else the plan's off-16th hats with the rolls left out | each 16th moved as the swing puts it |
| kick, snare, hat, perc placement | the share of bars with a hit on each 16th, per role; how much a bar differs from the one two bars before | a two-bar motif drawn from those shares, each later bar the one two bars before with each 16th drawn again at that rate |
| hat rolls | the share of beats holding hats closer than 3/4 of a 16th; their division (16th triplets, 32nds, 32nd triplets, 64ths) and length | a roll in a beat at that rate, replacing the hats it covers, its velocity rising |
| 808 placement | how often it starts on a kick's 16th, and on each 16th without one | the same chances, as a two-bar motif |
| 808 pitch | how often it repeats its pitch or plays the chord's root, its pitches, its register (5th to 95th percentile) | in that order of chances, in the register |
| 808 length, slides | the median share of the gap to the next note it holds; the share of notes the note before is held over (a slide), their intervals, the Glide Time | the same; a candidate with a slide gets one voice with Glide, as `plan` does |
| chords | `melodic.json`'s chord segments: the labels that cover 80 % of the time (at most 6), the changes between them, the length of one in half bars | a progression on half bars, each chord following the reference's changes from the one before, voiced from where the melodic stems play |
| velocities, pads | each role's velocities and pads, the pad played most on each 16th | drawn from those |

- **Not a copy.** Each bar's kick and snare 16ths, 808 (16th, pitch) and chord changes are compared with every reference bar of the section (F1). A candidate whose mean over its bars is over `--max-copy` (0.75) is drawn again, up to 50 times; `candidates.json` keeps the score, how many were drawn and which one was kept. A statistic at 0 or 1 still plays where the reference does: in the tests a snare on every bar's 3 always comes back. No candidate is a bar-for-bar copy.
- **Reproducible.** Candidate k is drawn by a generator seeded with (seed, k): the same seed gives the same notes, plans, MIDI files and audio samples, and candidate 3 is the same whether 3 or 10 are drawn. `candidates.json` carries a digest of each candidate's notes. Two runs give float WAVs whose samples are equal, though their bytes are not: libsndfile writes a timestamp into each one's PEAK chunk.
- **Rendered as Live would play the plan.** Each candidate is a plan with the reference plan's pads, 808 one-shot and settings, and faders. `render.py` plays its drums and 808, and `live-build` can build it. The chords play on a plain additive pad, as a sketch of the harmony, not as a choice of sound.
- **Ducked like the reference.** Under each kick the 808 and the chords follow the reference's own kick curve (`kick_curve` re the layer's median, at its points from 20 ms to 400 ms), cut at 0 dB and at −24 dB: a stand-in for a Compressor keyed from the kick. Without reference stems, a duck of 12 dB (808) and 10 dB (chords) over 200 and 270 ms. The chords sit as far over or under the drums (LUFS) as the reference's other stem does over its drums.
- **Measured, then heard.** Each candidate's stems (drums, the 808 as bass, the chords as other) are set to the reference's drum loudness and measured against the same bars of the reference (`compare`): how far the 808's and the chords' kick curves are from the reference's, each drum role's level re the loudest against the reference's, and how many measures differ by more than the reference's own halves do. The mixes go into one listening round, numbered as the candidates are.
- **Clean against separated.** The candidates' stems are clean and the reference's are Demucs stems, so the levels and the octave balance include the separation's own differences. `--separate MODEL` puts each candidate's mix through Demucs first. A difference within the reference's spread is no difference. A curve distance is not a verdict: it ranks candidates against each other.

On PIRAMIT's plan (section A, 54 bars, seed 1; the reference stems by htdemucs, the candidates clean) the four candidates drew a copy score of 0.50 to 0.72: one was drawn 4 times to get under 0.75. The 808 does not duck in PIRAMIT (its curve rises at the kick, where it starts), so the fit left it alone. The chords' curve falls 7–8 dB at 20–50 ms and is back by 100 ms. The 808's curve came 1.5–8.8 dB from the reference's and the chords' 0.9–1.5 dB. The kick sat 4.3–5.6 dB further under the snare than in the reference, the hat within 1.3 dB. 45 to 51 of 78 measures differed by more than the reference's own halves do, as expected of clean stems against separated ones. Two runs gave the same digests, scores and samples. Nobody has listened to these candidates yet.

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
- `refdissect candidates WORK --out DIR --n N --seed S`: N new 8-bar beats drawn from the plan's statistics (section 5), each as a plan, MIDI files and a render, measured against the reference's same bars and put into one listening round. Options: `--section`, `--bars`, `--reference-bar`, `--stems`, `--question`, `--max-copy`, `--no-duck`, `--no-compare`, `--kick-gap`, `--separate MODEL`, `--lufs`.
- `refdissect check-rebuild WORK --reference MIX`: the plan checked against the song the same way on both sides (below).

## Checking a rebuild the same way ([`rebuildcheck`](../src/refdissect/rebuildcheck.py), [`mastering`](../src/refdissect/mastering.py))

`compare` needs a rebuild's stems made the way the song's were. `check-rebuild` makes them from a plan, without Live:
1. **Render.** `render.py` plays the plan (the pads and the 808 as Live plays them, at their faders), with the song's own stems the plan does not mute at their fader, as Live plays them beside it.
2. **Master like the reference.** Three things are measured on the reference mix and nothing else: integrated loudness (LUFS), spectral tilt (the slope of its third-octave levels, 50 Hz–16 kHz, in dB per octave) and its sample peak. The rebuild is tilted to that slope, then gained into a look-ahead limiter at that peak until it reaches that loudness. Only the tilt, not a full EQ match: a match would copy the song's balance between sounds onto the rebuild and hide what is wrong with it. `--mp3 256` adds the mp3 round trip the suite's mixes had.
3. **Separate with the same Demucs**: the model, seed and shifts are read from the reference stems' `separation.json`.
4. **Measure.** Every `compare` measure, "before" (the rebuild's clean tracks at the master's gain against the song's separated stems: the comparison without this command) and "after" (separated against separated), each against the song's own spread. Null tests per stem, before and after: the residual in dB after the least-squares gain, on the waveform and on STFT magnitudes (46 ms frames). With `--truth` (a suite beat's `events.json`), null tests per sound as well: each truth drum sound against the pads assigned to it, and the truth 808 against the 808 track.

```bash
uv run refdissect check-rebuild work/song/refdissect --reference song.mp3 --mp3 256    # -> work/song/refdissect/check/
uv run refdissect check-rebuild "$S/v1-1/runs/m5" --truth "$S/v1-1/events.json" --mp3 256 --out check-v1-1
```

`check/` holds the song's stems played back and re-separated: keep it local ([09](09-data-and-copyright.md)).

### What it measured on the suite (tag sy-check, 2026-11-11)

The 11 beats of the dev subset, m5's plans on htdemucs and m5-ft's on htdemucs_ft, against each beat's truth. As a control, the truth's own premix (the beat before its clipper) went through the same mastering, mp3 and separation: a perfect rebuild.

**Mastering.** Every master reached the reference's loudness within 0.04 LU after the mp3. The tilts applied were −1.02 to +0.94 dB/octave (the references: −3.43 on average); the limiter took up to 1.84 dB (htdemucs_ft plans: 2.27; the control 4.86, the clipper's density).

**Per sound, against the truth.** The magnitude null agrees with the suite's per-sound scores; the waveform null does not.

| | htdemucs (m5) | htdemucs_ft (m5-ft) |
|---|---|---|
| Spearman ρ, magnitude null vs per-sound F1@20, 61 truth sounds (a sound no pad plays nulls at 0 dB: 14 and 10) | −0.90 | −0.86 |
| the same, the sounds a pad plays | −0.77 (47) | −0.75 (51) |
| Spearman ρ, waveform null vs F1@20, the sounds a pad plays | +0.11 | +0.17 |
| median null of a played sound: waveform / magnitude (dB) | −0.06 / −10.7 | −0.16 / −12.4 |
| Spearman ρ, per-sound \|balance\| (magnitude gain) vs the suite's \|balance_db\| | 0.76 | 0.66 |
| Spearman ρ, 808 magnitude null vs the plan's 808 note F1@50, per beat | −0.68 | −0.56 |

- The waveform null is too strict for drums: a hat 0.2 ms off (the plan's median hat error) is out of phase above a few kHz, so hats and snares null at about 0 dB whatever their F1. Only kicks null on the waveform (−0.02 to −15.3 dB).

**Per stem, separated against separated.** Measures beyond the song's own spread, over the ~78 measures both sides have:

| | before | after | control (a perfect rebuild) |
|---|---|---|---|
| htdemucs | 38.1 | 44.0 | 22.4 |
| htdemucs_ft | 38.2 | 41.4 | 23.5 |

- Separating both sides takes most of the gap in the drums and the 808 out of the measures. Median |difference| on htdemucs: drums LUFS 3.9 → 0.8 dB (control 0.3), bass LUFS 2.8 → 0.3, the bass's octave balance 5.9 → 3.05 dB (0.6), the hat's level re the kick 4.7 → 1.8 dB (0.8), the snare's 2.3 → 1.1 (0.3). On htdemucs_ft the bass moves alike (1.2 → 0.2 dB, 4.4 → 2.3), the drum roles do not (hat 1.3 → 1.7, snare 0.4 → 1.7).
- More measures are flagged after than before because the song's own other stem, which the plan plays back, comes out of a second separation different: it nulls at −7.7 dB against the song's other stem (htdemucs_ft −11.2), the control's at −13.1 (−14.8). That stem holds the separation's bleed (drums and 808), which a second pass puts elsewhere. Before, it was the song's stem itself and could not differ.
- A perfect rebuild still differs on 22–23 measures: the limiter is not the scenarios' clipper, and a synthetic loop's halves are so alike that its spread is near 0 on many measures. Read "after" against that floor, not against 0.
- The separated drums null at −4.8 dB on average (htdemucs_ft −6.9; control −14.5 and −15.8), the separated bass at −10.8 dB (−12.1; control −25.8 and −25.5).

**Agreement with the suite, per beat** (11 beats, Spearman ρ against the plan's scores):

| | htdemucs | htdemucs_ft |
|---|---|---|
| separated drums, magnitude null, vs drums F1@20 | −0.06 | −0.30 |
| the truth's drum group against the rendered drums (clean), vs drums F1@20 | −0.47 | −0.84 |
| separated bass, magnitude null, vs 808 note F1@50 | −0.53 | −0.54 |
| measures beyond the spread after, vs strict F1@20 | −0.35 | −0.30 |

- Against a truth, per sound, the new measure ranks sounds as the suite does. Separated against separated, the only form a song allows, it ranks beats on the 808 about as the suite does, but hardly on the drums: tempo160-1 (drums F1 0.73) and layers-1 (0.47) null at −0.3 dB, full-1 (0.35) at −7.2 dB. A drum stem's null mixes what was played wrong with the kit's sound, its level and what the separation does to both; the suite's F1 sees only the hits.

**Existing outputs.** `compare` gives the same rows and spread as before on full-1's htdemucs and htdemucs_ft stems (the same data; its rows come out in a different order from run to run, as before), and `render_plan` writes the same samples and `render.json` for full-1's m5 plan.

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
