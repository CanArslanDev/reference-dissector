# 01 – The pipeline, step by step

Every step reads files and writes JSON (and, locally, audio). The order:

```
separate ─▶ analyze ─▶ melodic ─▶ match ─▶ fit-808 ─▶ plan ─▶ live-build
             │           │          │         │          │
     analysis.json  melodic.json matches.json synth808.json plan.json, midi/
     bass808.json   basicpitch/              808 (synth).wav
     templates/
```

All audio is processed as mono float32 at 44.1 kHz (`audio.read`). `refdissect suite` runs the same steps on 48 synthetic beats with every answer kept and scores the analysis and the plan against them ([03](03-calibration.md)). Suite numbers on this page are strict (no offset removed, a hit counts within 20 ms of where its sound starts) and are means over scenarios on htdemucs stems; "m2g" is the run of the merged replica-mode code (`results/calibration/suite-m2g.json`).

## separate

Demucs (`htdemucs`, or `htdemucs_6s` for guitar/piano stems) runs in its own uv environment (`workers/separate_worker.py`), the same way [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp)'s `separate_stems` tool does: torch never loads into the main process, and the memory is gone when the job ends.
- On an M-series Mac, a 30 s beat separates in about 5 s on MPS once cached; a 129 s song with `htdemucs_6s` takes about 15 s. The suite, which separates several files per model load, takes 2.1 s a beat with htdemucs and 8.5 s with htdemucs_ft.
- The guitar stem of the 6-stem model came out nearly empty (−80 dB) on Turkish trap.
- `--seed` seeds Demucs' random shift: the same seed, the same stems. The suite separates every beat with seed 0 and one shift, several files per model load.
- **htdemucs stays the default.** On the merged code htdemucs_ft gives better drums (plan F1@20 0.612 → 0.648, paired over the 48 beats +0.039 [+0.017, +0.062]) but worse 808 notes (note F1@50 0.936 → 0.924; bass_low 0.949 → 0.338) and snares (0.961 → 0.891), m2g against m2g-ft. The default changes only for a model better on both drums and 808 notes.

The drum and 808 steps need `drums.wav` and `bass.wav`. The beat tracker reads the sum of all stems, `melodic` reads `other`, `piano`, `guitar` and `vocals`, and all of them are tuned and go to Live as reference tracks.

## analyze

### Grid (`grid.estimate`, `beats.track`)

Produced music sits on a fixed grid, so the grid is fitted rather than tracked. A comb finds it:
- **Score.** For each tempo T and phase φ, the score is the mean onset strength (librosa, hop 128 = 2.9 ms) at φ + k·60/T.
- **Search.** Tempos in 0.05 BPM steps, then ±0.1 BPM around the three best in 0.002 BPM steps, with phase in half-frame steps. About 2 s for a 2-minute song.

The comb cannot know the octave, the bar line or the meter. A beat tracker gives those: `beat_this` 1.1.0 (MIT code and weights), in its own uv environment like Demucs, on the sum of the stems. On an Apple M4 it took 1.1 s for PIRAMIT's 126 s. Its downbeats were right more often on mixes than on drum buses (F 0.94 against 0.82). From its beats and downbeats `grid.estimate` takes:
1. **The tempo**: a line through the beats, not their median interval. beat_this's beats come on 20 ms frames, so on PIRAMIT the median gave 68.18 BPM and the line 68.498.
2. **The octave.** The tracker's level and its relatives (×2, ×½, ×3/2, ×2/3) are each refined by the comb within ±3 %. `--bpm-range` (default 100–180) picks the octave in range; for trap that counts the hats' grid, and the snare falls on beat 3. ×3/2 is taken only for dotted quarters: the tracker's bars hold two beats (or `--meter` is x/8) and the onsets split its beat in three, as in 6/8. `--bpm` forces a tempo.
3. **The phase.** The comb's, within 30 ms of the tracker's beats, then moved from the onset-strength peaks onto the attacks: they came 4.8–13.2 ms after them on 82 test beats, 7.1 ms on PIRAMIT. Then it goes to the 16th of the bar that most downbeats vote for, so `phase` is the first bar line. Fewer than two thirds of the votes mark the bar line unsure; `--meter` then decides.
4. **The meter**: 3 or 4 beats a bar where 60 % of the tracker's bars agree, else 4/4; `--meter` sets it.
5. **Swing**: where the hats' off-beat 16ths and 8ths attack (0.5 straight, 0.66 a triplet swing). Onsets at one lag after each kick (a kick sample's own second transient) are left out.
6. **Drift**: how far the tracker's beats move from the constant grid. Beyond 10 ms the moved grid is kept as `tempo_map`; the grid itself stays constant. Two tempi (a beat switch) are found too: the grid follows the longer part.

On 86 test beats (72–160 BPM; 4/4, 3/4 and 6/8; swing up to 66 %), the comb alone in 100–180 read 24 of 80 constant-tempo beats outside 6/8 at ×4/3, ×3/2 or ×2/3; with the tracker, none. The first bar line was right on 81 of 84, and the 3 wrong ones were flagged. On the suite, beats F 0.715 → 0.962 and downbeats F 0.637 → 0.939 (base → m2g). Still wrong there:
- tempo72's bar line, a beat off on both beats (downbeat F 0);
- meter68, read in 4/4 at ×2/3: 6/8 was never detected;
- the seed-1 swing beats, read as straight 8ths.

Without a tracker (`--no-tracker`, or when it cannot run) the grid is the comb in `--bpm-range` alone, as before replica mode.

- **Check.** `grid.offsets` reports, per beat of the bar, how far the strongest snare-band onset sits from the grid line (median and p90). A right tempo stays within a few ms over the whole song; a wrong one drifts. On PIRAMIT, 136.999 BPM gives p90 4.7–11.3 ms, beat 3 in 58 bars. At 136.36 BPM, from the same bar line, beat 3's onsets sit 2.0 ms early in bar 1 and 49.5 ms early in bar 7 (8.0 ms a bar). `offsets` looks within 50 ms of each line, so after bar 7 it finds only other onsets the drifting line passes (22 more bars). By the end the grid is 0.59 s off (computed from the two tempi).
- **Two time bases.** The grid's phase is on the attacks; `offsets` and the onset-strength peaks read it moved back by the phase correction (`grid.onset_grid`).

### Tuning (`grid.tuning`)

Stable spectral peaks on each pitched stem (bass, other, piano, vocals, guitar), over 60 s: every frame's peaks (16384-point FFT, parabolic interpolation) that stay within ~5 cents for five frames, their offsets from equal temperament averaged on the circle and weighted by level. The consensus is the median.

The first version used `librosa.estimate_tuning`. On synthetic tones it was off by up to 60 cents (it histograms every bin, sidelobes and noise included), and it put all of PIRAMIT at +0.28 semitones. The peak method stays within 1 cent of synthetic tones and puts PIRAMIT's melodic stems at A440 (other −2, piano +1.5 cents; pyin agrees, +2 to +3).

The 808 has a tuning of its own: its template's (see 808 notes below), the settled pitch of its long notes over the nearest MIDI note (`analysis.json` tuning `808`). An 808 sample is often detuned by itself. PIRAMIT's is C1 +11.8 cents: QIFFT of the 300–600 ms sustain of 53 notes, all starting on a kick. Each note's written pitch is its settled pitch less that tuning, and `fit-808` renders the one-shot at it. The prototype's notes, read about 30 cents over C, came out "C or C#" one by one: a pitch tracker hears the drop ([02](02-case-study-piramit.md)).

Before replica mode the tuning was the circular mean of the notes' pYIN pitches (`detect.pitch_offset`), and it said +33 cents (agreement 0.64): it heard the drop. The same mean over the new notes' settled pitches is kept as `808_agreement` (PIRAMIT: 0.868).

### Drum templates (`detect.discover`)

1. **Bands.**
   - `low` (< 200 Hz) finds kicks under hats and snares.
   - `snare` (120 Hz–14 kHz).
   - `hat` (> 3.5 kHz).
2. **Where onsets start.** Each onset-strength peak of a band is moved back to where its sound starts (`hits.rise`): from the quietest point in the 30 ms before it, the first point where the 1 ms envelope reaches 30 % of its peak. The low band also reads the snare band and looks back 45 ms: on htdemucs drums a kick's body comes 26 ms after its click, the rest of it went to the bass stem. What still rings from the hit before is left out.
3. **Places in the beat.** The starts are clustered by where they fall in the beat (±12 ms), densest first. Straight or swung 16ths, triplets and 32nds are places of their own wherever the song puts them, in any meter. Per place, hits are grouped greedily by waveform (`hits.group`, NCC ≥ 0.6). A group needs 3 hits, or one per 4 bars in a longer song. With 8 hits per 16th slot within ±12 ms, an 8-bar beat at 95 BPM got no template at all, and a rim only on swung 16ths (62 %) none.
4. **Merge** groups within a band whose averages match (NCC ≥ 0.8 over 100 ms).
5. **Hits that start something.** Each candidate's template slides over the stem. One rings rather than starts when its hits sit on an onset (within 10 ms) less than 30 % of the time, or when they have no attack in the hat band (under 5 %) and something starts 15–45 ms before 30 % of them or more: an 808 matched once a period in the snare band, or a kick's body after its click.
   - A clean candidate (NCC ≥ 0.8 on 80 % of its hits) with nothing starting just before it may sit 15 ms from its onsets. Its template starts where its role's band climbs from the floor, an onset where its band reaches 30 % of its peak: on leadin-1 the snare's template started 13 ms before its hat-band onsets, and the snare was lost.
   - A kick without a click has no top either, but nothing starts just before it. Between 8th hats, or with no hats at all, it was lost (F1 0 on synthetic beats).
6. **One template per sound.** Candidates are taken in order of clean hits (NCC ≥ 0.8; squared, over all their hits). A candidate is the same sound as a kept one that finds most of its hits when their gains move together (r ≥ 0.6 for the same guessed role, 0.8 otherwise), or when, in the band of its role, the kept template has its shape at its level and either the two find each other's hits or they guess the same role and the kept one holds its level where it was found too.
   - A kick under some of the hats stays a kick: the hat template holds it 6–11 dB down in the low band.
   - A crash on every kick stays a crash: the kick does not hold it in the hat band.
   - A view that finds 3 hits or more that nobody else does (a kick in the hat band where an 808 hid its low band) is kept as `same_sound_as`, and the hits it shares are counted once (`count_once`). Hits it finds again within 50 ms of its holder's are not new ones. A candidate with fewer new hits is left out: kept as a sound of its own, it played its holder's hits a second time.
   - Of two candidates of different roles locked at one lag of 15–60 ms, the quieter goes: on two beats something 42 ms before every snare that was no hit.
7. **Residual pass.** Onsets that no template of their band explains (no hit within 10 ms) are grouped wherever they sit, 3 hits or more (one per 16 bars in a longer song), and go through 5 and 6: rare sounds, crashes, fills.
   - With 3 hits in any song, the residual templates grew with its length. One per 16 bars took PIRAMIT from 15 to 10, HAYIR DEMELIYDIM from 7 to 5, and a v1 beat tiled 6 times (192 s) from 23 to 11 (its extra templates 14 → 9).
   - What recurs in every 16 bars stays: a producer beat tiled 6 times kept its 19. On those two tiled beats 2 and 4 extra templates still get a drum pad, as kicks or snares (`live.make_plan`; library NCC 0.53–0.83).
8. **Template.** Each survivor's hits are lined up with the sound's start 5 ms into the window (`hits.aligned_template`), read in the band of its guessed role: the snare band leaves out the kick or 808 before a snare. They are averaged, or their median taken in the hat band (on 30 calibration beats the median made hat templates 6.5 dB cleaner against their true files). analysis.json gives each template's `attack_s`, `same_sound_as` and which pass `found` it, and each hit its `t_attack`.
   - In the full band the attack sat on the kick or 808 before the snare: all six calibration snares were placed 13–24 ms early. On 32 suite beats (mean over 14 scenarios of each sound's median error) the plan's snares moved from 11.6 ms early to 1.4 ms late, its kicks from 9.6 to 2.4 ms late, and the analysis's kicks from 22.6 to 0.9 ms late.
   - A template can start on its sound's body (`detect.starts_before`). Where at least half of its hits start 3 ms or more earlier in the hat, snare or kick band, and a fifth more of them than at its own start, its `attack_s` moves there; its window stays where the matched filter finds the hits, and the plan's notes go to `t_attack`. On 48 suite beats this moved 10 templates 3–22 ms, 8 of them nearer the truth: dynamics-2's kick from 16.7 ms late to 1.6 ms early.
9. **Names and roles.** Templates are named `sound_1`, `sound_2`, … by the number of hits their matched filter finds here, before the joint fit. The role is only a guess here (`kick`/`snare`/`hat`): the median over the hits of the share of energy each adds below 150 Hz and above 5 kHz over the 60 ms before it. Summed instead, the kicks under some hats decided: 33 of the 56 templates that played a hat on 32 suite beats read as kicks (8 of 84 now; 71 % of all guesses right, 57 % before). Matching decides it later.

On those 32 beats (14 scenarios, htdemucs stems), against the code before (16th slots ±12 ms, 8 hits a slot, the cross-band duplicate test, the full-band attack), strict F1@20 per role, mean over scenarios: kick 0.26 → 0.87, snare 0.85 → 0.96, hat 0.45 → 0.75, open hat 0.23 → 0.97, clap 0.61 → 0.67, crash 0 → 0.47 in the analysis; kick 0.17 → 0.80, snare 0.58 → 0.79, hat 0.50 → 0.72 in the plan (`results/calibration/suite-l1a-after2.json`). 88 % of the true sounds get a template (59 % before), at 1.2 templates each (1.4 before); the residual pass also keeps 2.2 templates a beat that play no true hit (on the v1 beats, transients of the melodic loop that htdemucs left in the drum stem).

After the review fixes (each hit once, hits that start something, templates that start where their hits do, the residual pass's minimum), on all 48 suite beats (22 scenarios), against the code before them, strict F1@20, mean over scenarios (`results/calibration/suite-l1a-pre.json`, `suite-l1a-fix.json`):

| | Analysis, before | Analysis, after | Plan, before | Plan, after |
|---|---|---|---|---|
| drums | 0.816 | 0.830 | 0.538 | 0.547 |
| kick | 0.833 | 0.871 | 0.791 | 0.809 |
| snare | 0.926 | 0.949 | 0.811 | 0.834 |
| tom | 0.37 | 0.48 | 0.37 | 0 |

- Kicks are found 0.04 ms late (median), not 0.89; p90 9.2 ms, not 11.4. The plan's kicks: p90 9.1 → 9.0 ms.
- leadin's snare: 0.50 → 1.00. full's kick: 0.64 → 0.80 (0.72 before the lane): its snare-band view had played each kick again 37 ms late.
- Plan levels against a perfect kit: 3.90 → 3.62 dB.
- The plan's tom is lost: rare-1's tom template no longer counts its own ring (17 hits → 10, NCC ≥ 0.8 on 53 % → 90 %), and the plan gives a pad only from 12 hits.
- melodic-1's kick stays 14.7 ms late (18.6 ms in the plan): its click shows in the hat band, but at no one place on most of its hits.
- Extra templates: 2.14 → 2.21 a beat in the analysis, 0.68 in the plan both times.

### Hits (`detect.detect`, `layers.explain`)

**Matched filter.** Each template's first 40–80 ms (the attack, see below) slides over the whole drum stem in its band (`xcorr.sliding_ncc`). Peaks above 0.6 (0.65 for hats) are candidate hits, 8 ms apart at least, so a flam's grace note 10–30 ms before its hit and a 64th roll (27.6 ms apart at 136 BPM) stay apart. A tone's candidates are 35 ms apart (100 ms in the low band). The stem is padded: a hit at 0 s is found too, and one whose attack the stem cuts off plays from 0 s. Each hit carries `t`, 5 ms after the window starts, and `t_attack`, where the sound starts: the template's `attack_s` after the window start (for a pitched copy, sooner by as much as Simpler plays it faster). A hit without `t_attack` is timed by its template's `attack_s` too (`evaluate.estimate_from_analysis`). Timed by `audio.attack_time` on the averaged template instead, the kicks of 32 suite beats came 15.7 ms late (drums F1@20 0.810 → 0.717).

**Every moment at once.** Keeping one template per moment and band lost nearly every hat under a snare (416 of 432 on 22 suite beats). Instead, each candidate's neighbourhood (±150 ms) is fit as a non-negative sum of all the candidates in it (NNLS) on the kick, snare and hat bands, each band weighted by 1/RMS of the stem in it so that each counts alike. Each hit gets its own gain, the hit's level.
- **One sound found twice.** Two candidates whose placed templates correlate at 0.6 or more are the same sound (a duplicate template, a kick ringing one period on): the one that explains more of the stem keeps it. Templates of one kind that lose two thirds of the fewer's hits to each other are merged, so a hat found in the hat band and in the snare band plays once; the merged template's pitched copies go with it. At one half, a clap on every other snare, its template and the snare's each holding half the other, lost half its hits to the snare and was merged into it: 0 of 32 claps. Of the 15 merges on the suite, 13 lose 0.67–1.0.
- **Held sounds.** A template that holds another (the snare template holds the hat on every snare) has it subtracted first; otherwise the snare takes the hat's place at every hit. The held template's coefficient is 0.3–2.0, read both over all bands and in its own band (the smaller is subtracted), it takes 1.5–30 % of the holder's energy over its span, and it starts up to 30 ms before or after the holder. On the 48 suite beats: 50 holds, coefficients 0.32–1.25, shares 1.8–30 %, snare templates holding a hat at −19 to +29 ms. Over all bands only, a hat's average, which carries the body of the kicks some hats sit on, made a kick template hold a hat it never plays (full-2: 0.61 over all bands, 0.06 in the hat band); the purified kick then held a negative hat, and a hat was found under the kicks that have none (24 such hats on 3 beats, now 0).
  - **Where it sits.** Of where the held template fits best in its own band and over all bands, the one nearer the holder's start. Over all bands, a snare's body put the hat 21 ms after the hat it plays with (tempo95-1: 23.8 ms, against 2.6 ms in the hat band), and those hats came 20.5 ms late. In the hat band alone, a snare's own top 28 ms on beat the hat under it (producer-1: 28.7 ms, against 2.4 ms over all bands), and 41 hats under snares were lost.
  - **A variant is no layer.** A template of the holder's fit band, held within 2 ms and averaged from fewer hits, is a variant of the holder's sound. On tempo160-1 the main hat held a second hat-band template at 0.4 ms, which then got 37 hits under the hats, 33 of them hats the main template already had.
- **Residual templates** (discover()'s second pass) keep their matched-filter hits. They are not held, not looked for at their band's onsets, and get no pitched copies: they were found where no template of their band plays. Held, probed or copied, one found another template's sound a second time: on v1-1, 40 of the 50 hits one got under other templates were the main hat's own hits, and 9 were on no hit.
- **Composites.** A template the others explain to 80 % is left out, and its hits go to them.
- **Keeping a hit.** Its gain reaches 25 % of the template's median gain, and the fit loses more without it than the template takes at shifted, empty windows (their 95th percentile).

**Where the kept hits leave something out.** A held template is looked for at its offset under every hit of the template that holds it. Every noisy template (not a tone, not an echo of another) is looked for where its matched filter peaked at NCC 0.3 up to its threshold, and at the onsets of its band. Such a hit needs half the template's median gain (a quarter where what the other hits leave matches the template at its threshold), and the null. It is marked `source: "layer"`, with `under`, the template it was under. For these hits the null is a backstop: of 1,724 probe hits over the gain floors on the 48 beats, it alone rejected 44, all of sounds that no truth sound was assigned to. Two further floors were measured and left out. An NCC of 0.2 with what the others leave, in the template's own band, would remove 23 true probe hits for 13 false ones and 35 of unassigned sounds. A median gain of at least a quarter of the template's own level lost a sound that plays 26–30 dB under the few loud hits its template was made from: 0.75–0.80 → 0.12–0.14 of its hits found (synthetic).

**Tones.** A template with half its energy in the kick band (bands weighted as in the fit), or a quarter there that rings (NCC 0.5 with itself 8–35 ms later), is a tone. Its matched filter has a broad maximum, so it gets no pitched copies and is not looked for under other hits; a kick template's low ring explains a little of almost anything, so it is never held. A kick may ring little and still be a tone (leadin-2: 59 % in the kick band, ringing 0.29); probed as a noisy sound, it gave 65 layer hits, 40 of them false (as a tone, 15, all true). A hit of a tone (or of a low-band template) within 250 ms after a louder one is that one's ringing when it is less than half as loud, or below what the template's own decay holds at that delay (0.70 at 23 ms for a kick on v1-1, which was found again 15–23 ms after every strike). A tone found in the hat band with less than half its energy in the kick band is a hat whose average carries the body of the kicks under some of its hits: it is held, spaced and kept as a hat (not probed).

**Pitched copies.** A noisy template (its copy one semitone off matches it at NCC < 0.5: hats, snares) is also matched 1–5 semitones up and down, transposed as Simpler plays it. Its hits carry `semitones` and are scored as sounds of their own.

A quieter hit at most 35 ms before another of the same sound (no third within 70 ms) is a flam's grace note (`flam_of`).

**Each hit once** (`count_once`, after the fit). A view's hits within 50 ms of its main's are the main's: a view finds its main's hits again at one lag (36–39 ms on 4 suite beats; within 15 ms, full-1's view of the kick kept 43 hits 37 ms late). A template's own ring is no hit either: within 250 ms of an earlier hit, no louder than that hit's ring would be there (the template's level at that lag, never rising again), where nothing starts in its band. A tonal snare's template had matched its own ring 101 times for 32 snares.

**On discover()'s templates** (all 48 suite beats; m2 is the merged code before the residual, variant and where-it-sits rules, m2g after them):
- drums F1@20 0.847 → 0.880 at analysis, 0.600 → 0.612 in the plan;
- open hats 0.901 → 0.979, snares 0.933 → 0.961, hats 0.807 → 0.859; hat timing p90 6.8 → 3.3 ms;
- claps 0.580 → 0.500: the layers scenario's clap had been found only by a residual template held under the snare;
- on the 22 beats listed below: false hits per beat 59.6 → 35.6, hats within 20 ms of a snare or clap 381 → 376 of 432, F1 over hits 0.841 → 0.879. On the earlier templates (l1b-fix) these were 31.5, 324 and 0.781.

Still behind each part alone:
- Two templates of one sound found in two bands can hand each other their moments by hit count. On melodic-1 the low-band kick became a composite of a snare-band residual view of it: kick F1@20 0.929 → 0.748.
- A merge can move a snare to its less clean template. On layers-2 the clean low-band snare template went into the snare-band one (clean 0.44, under the plan's 0.5), so no pad plays the snare (plan snare 1.0 → 0).

The numbers below were measured on the templates of the earlier discover() (16th slots, 8 hits a slot, the full-band attack), with no `count_once`.

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

On the other 26 suite beats (13 scenarios): drums strict F1@20 0.438 → 0.492 at analysis, 0.345 → 0.365 in the plan; layer hits 317 true at 20 ms, 31 false, 137 of sounds no truth sound was assigned to. 95 hits of one of those (melodic-1, 80 found under other hits) come from a template whose median gain is 0.088 of its own level; they play on a hat pad and raise the plan's hat F1@20 there 0.306 → 0.418 (precision 1.0 → 0.53). Kicks at 20 ms regress on meter34-2 (0.737 → 0.286), rare-1 (0.167 → 0; at 50 ms 0.518 → 0.927, one hit per strike) and melodic-1 (0.317 → 0, that template no longer assigned to the kick). In the plan, bass_low (0.277 → 0.157), glides (0.514 → 0.491) and leadin (0.386 → 0.369) fell: the plan then placed notes at a template's window time, not at `t_attack`.

It costs about 4 s per 30 s beat: the suite's analyze step takes 10.0 s per beat against 6.2 s with one template per moment (m1) on the 22 beats, 8.7 s against 5.4 s on the other 26. Not found:
- a sound that always plays with another has no template of its own (the layers scenario's clap on every snare);
- a hat 14 dB under a clap's top end (v1orig-4);
- a hat under a snare 9 dB or more below its usual level (synthetic: 24 of 24 at −6 dB, 0 of 24 at −9 dB);
- a clap on every other snare at a constant offset when its template and the snare's each hold half the other: its 16 solo hits, not the 16 on the snare;
- a tone ringing longer than 250 ms (a 50 Hz kick with a 250 ms decay is found again 340 ms on, once per strike);
- closed hats in a long open hat's decay (0.69 of them at a 350 ms decay, synthetic); one that always sits there puts its decay into the closed hat's template, which then finds one closed hat too many per beat.

**Why the attack only.** Patterns repeat, so whatever always follows a hit is in its average too: the kick two sixteenths after a snare, for example. With a 300 ms window, that louder kick decided the matched filter: the snare's "hits" landed 352 ms early.

`peak_ncc_histogram` shows each template's peaks per 0.1 of NCC. A clean template shows a gap: PIRAMIT's −1 snare (odd bars) has 27 peaks at ≥ 0.9, one in 0.8–0.9 and none between 0.4 and 0.8 (267 more in 0.3–0.4). Before replica mode this page said "even-bar" and "nothing between 0.3 and 0.8"; the old template's own histogram had 269 peaks in 0.3–0.4.

### 808 notes (`bass808.transcribe`)

An 808 line is one sample retriggered at different transpositions and cut by the next note. `bass808.transcribe` reads it that way and writes `bass808.json`. `analysis.json`'s `notes_808` is its projection (t, t_attack, end, pitch, midi_float, midi_settled, cents, slide, peak, glide, flags).
1. **Onsets.** Re-attacks of the bass stem's 808 band (a notch of at least 4 dB recovering 6 dB within 30 ms, or a rise of 9 dB), and attacks of the drum stem's low band. Where the re-attacks are too few to learn a template from, the drum attacks are onsets too: on synthetic lines at 160–170 BPM the re-attacks found only 6–10 of 28–32 notes, as a driven 33 Hz tone dips only 0.5–2.3 dB at a cut.
2. **Template.** The long notes of the most common waveform, averaged for as long as each still matches the others. It is learnt from three sets of onsets (on a drum attack; on a drum attack or a notch of 8 dB; all), twice each. The one whose notes explain the most of the band wins, and within 1.5 % of it the one whose notes start on drum attacks.
3. **Settled pitch.** QIFFT over the first sustain window whose two halves agree within 3 cents, searched from 300–600 ms on in 150 ms steps. With no note that long, the asymptote of an exponential fitted to the template's own falling pitch. v1orig-1's 808 is still 88 cents over its settled pitch at 0.3 s: read over 300–600 ms its template was 35.1 cents sharp, over 600–750 ms 1.2 cents flat.
4. **Candidates.** The template's first 80 ms at every semitone within an octave either way, slid over the stem. At each peak the three best semitones, each refined in cents and in start (±20 ms) to what explains the most of the next 300 ms.
5. **Segmentation.** Dynamic programming over the candidates picks the template copies that explain the 808 band best, with one penalty per note. Each copy runs from its start until it is cut or the next note starts. A note is split where the 808 starts again inside it: two 16th notes at one pitch at 160 BPM had read as one note a semitone sharp.
6. **Pitch.** Each note's transposition to the cent: NCC over up to 300 ms, every 2 cents, then a parabola. The written note (`pitch`) is the settled pitch less the 808's own tuning. The pitch heard (`midi_float`) is the settled pitch plus the template's own drop over the note's read window, the truth's rule.
7. **Notes found where nothing attacks** are the note before going on: no envelope foot within 20 ms, no drum attack, a template fit under 0.9 and no restart of the drop.
   - A short attacked note whose follower settles back at the pitch before is a kick leaking into a held note (kick808-2: precision at 50 ms 0.66 → 0.78).
   - A note under 80 ms on a kick, at a transposition past the search's whole semitones, is the kick's attack split off the next note. PIRAMIT has 10 (no truth).
   - Legato notes are made only on request (`transcribe(slides=True)`). On the 24 beats of the 808 scenarios, the earlier rule made 79 legato notes for the truth's 24, and 6 of them were right.
   - The rule also merges across a rest. On PIRAMIT four notes of an 808 playing alone after a rest went into the note before, which then held 15.3 s: their onsets sit 27 ms before their envelope feet, and their fit is 0.887–0.893 ([02](02-case-study-piramit.md)).

On the suite (m2g, before replica mode in brackets):
- note F1@50 (50 ms, 50 cents): 0.936 at analysis (0.278), 0.926 in the plan (0.229); onset F1@20 0.935;
- median pitch error: 6.0 cents (55.3) at analysis, 7.7 (69.8) in the plan;
- the written note is right for 0.947 of the notes. bass_low's is right for 0.487: its 110–220 ms notes never settle, and its template reads 44–60 cents sharp from the fitted drop;
- no legato note is written; the truth has 24 on 5 beats.

The reader takes 9–50 s on a suite beat and 159 s on PIRAMIT's 126 s. analyze now takes 38.7 s a suite beat at 2 jobs, against 5.7 s in m1, and 202 s on PIRAMIT (1.83 GB at its peak). By the 808 lane's measure the reader is about 30 s of that; the joint fit and discovery take the rest (not separated). The pYIN reader it replaced (librosa onsets, a pYIN median over 60–450 ms) dropped every note under 140 ms and read pitches 14–27 cents sharp under 50 Hz.

### Sections (`structure.sections`)

1. **Per-bar signature.** Every bar gets the set of sounds heard in it: templates with at least 16 detections, plus the 808 where one of its notes starts.
2. **Smoothing.** Each sound's presence is smoothed by a 3-bar majority, so one extra perc does not start a section.
3. **Runs.** Runs of equal signatures shorter than 2 bars join the neighbour with the closest signature.
4. **Names.** Equal signatures get the same letter, so a returning section reads A … A.

## melodic

`refdissect melodic` (after analyze) transcribes the tonal stems (other, piano, guitar, vocals) with Basic Pitch 0.4.0. It runs once per stem in its own uv environment (`workers/basicpitch_worker.py`) and writes `melodic.json` and `basicpitch/` (notes and posteriorgrams per stem).
- **Tuning.** Each stem is measured again (as analyze does) and moved onto A440 before Basic Pitch hears it; its notes keep that tuning as cents.
- **Presets.** Two decodings of one run (`basicpitch.PRESETS`): `melodic` (onset 0.6, frame 0.3, 35 ms, melodia off) for the notes that go to Live, and `chord` (melodia on) for the chords. They were chosen on 63 synthetic renders: pads, 16th and 32nd plucks at 140 and 75 BPM, a lead with glides and bends, full mixes; clean, under a distorted 808, with a loop as bleed, in heavy reverb. By macro onset F1 the melodic preset scores 0.705 (held-out seed 0.663), against 0.583 (0.529) for ableton-live-mcp's `audio_to_midi` defaults (0.5 / 0.3 / 127.7 ms / melodia); 55 of 63 renders were better. 127.7 ms drops every note under 139 ms, 16ths at 140 BPM included.
- **The 808's range.** Other, piano and guitar start a semitone above the pitch under which the 808 spends 95 % of its time, so the 808 bleeding into them is not transcribed; a vocal keeps its low notes. On HAYIR DEMELIYDIM the first rule (the highest 808 pitch heard twice) put that floor at E3; this one puts it at G#2, and the guitar keeps 157 notes instead of 128.
- **Velocity** comes from the level in the note's own CQT bin, not Basic Pitch's amplitude, which is a confidence. Rank correlation with the true velocity on synthetic renders: plucks 0.75 (amplitude 0.26), a lead 0.58 (0.38), pads 0.12 (0.06).
- **Timing.** Basic Pitch's frames come in blocks of 142. Read as t × 22050 / 256 they drift 0.8 s by the end of a 3-minute song, so notes are timed block by block (`basicpitch.block_times`). On 865 synthetic pluck notes the onset error's p90 distance from its median fell from 8.9 to 6.2 ms. Notes go into beats from the start of the audio, like the drum notes; `--snap-ms` snaps them to the 16th grid.
- **Loops.** Where a section's notes repeat (every 1, 2, 4 or 8 bars), the repeats vote: a note most repeats hold is put back where one missed it, if the stem sounds at its pitch there. `--drop-below` also drops notes few repeats hold (off by default: it cost F1 on synthetic loops).
- **Chords.** The chord stems' note posteriorgrams are summed into pitch classes per 8th note, matched against 14 chord qualities, smoothed by Viterbi with a change penalty and averaged over the repeats of the loops the notes show. The 808 stays out of the pitch classes but names the bass: the 808 note sounding in a span gives the slash bass, and without one the lowest chord tone does. Chords with the same tones go to that bass, else to min7 and hdim7 over maj6 and min6. On 36 synthetic renders: majmin 0.953 and sevenths 0.904, against 0.732 and 0.201 for one triad per bar from chroma. Without the 808 as bass, majmin_inv fell from 0.804 to 0.715.

On the suite's melodic and full beats (4 beats, m2g): note F1 (50 ms, 50 cents) 0.630, onset F1 0.693; chords majmin 0.977, sevenths 0.929, root 0.926. majmin was 0.888 in m1, when the chords read the 808 notes of the reader before replica mode. In the plan each tonal stem's notes become a muted MIDI track ([07](07-building-in-live.md)).

## match

`library.Library` holds the first 0.6 s of every one-shot (1 s for 808s):
- **Trimmed** to 2 ms before it first rises above −30 dB of its peak, and peak-normalized.
- **Spectrum.** Each entry also gets a semitone-band spectrum: log energy per semitone from 40 Hz, z-scored, over the first 100 ms (400 ms for 808s).
- **Speckle index** (`library_speckle.npz`, written by `index-library`, or `index-speckle` for a library indexed before it): the fine structure of each noisy one-shot's spectrum in the snare and hat bands, on a 1-cent grid. It covers 6,718 of the 8,854 one-shots (every role but kick, 808 and melodic/loop), takes 102 MB and builds in 4.6–6.0 s. It is used only when its version, settings and entries match the library. See [04](04-identifying-samples.md).
- **Size.** 8,854 one-shots are about 470 MB in memory as float16. Loading them with the speckle index peaks at 690 MB; PIRAMIT's whole match step peaked at 1.51 GB.

For each template, in each of the kick, snare and hat bands (150 ms for snare, 80 ms for hat, 200 ms for kick):

1. **Waveform pass.** NCC against every entry at 0 semitones, vectorized through the FFT with the band filter applied to the spectra, over ±30 ms of lag. 1–3 s for 8,854 entries.
2. **Pitch-invariant shortlist.** The template's semitone spectrum is compared with every entry's at every transposition at once: transposing by k semitones shifts the spectrum by k bands. The best 200 entries, with their best transposition and its neighbours, are tried as waveforms.
   - Without this, a snare pitched down a semitone never reached the top 200 at 0 semitones (NCC 0.15 there) and was never tried at −1, where it matches at 0.95.
3. **Snare and hat bands: at any pitch.** The speckle index proposes the 20 sounds whose fine structure matches best over ±12 semitones, with their transposition; the whole-semitone search (1 and 2, −3 to +3) proposes its own 20. NCC tries each at its speckle transposition ±1.5 cents, in quarter cents. `semitones` is fractional; `transpose` and `cents` split it into Simpler's Transpose and Detune (`detune_cents` is the same number).
   - With 57 noisy suite sounds each played 0, +32.5, −47.5, −100 and +240 cents off (templates of the true hits), the right file came first 191 times in 285, against 65 by whole semitones. When right, its transposition was within 1.5 cents (median 0).
   - **The kick band keeps whole semitones** (−4 to +4): other kicks at fractional transpositions fit a kick better than its own file. A template found in the low band is matched at any pitch in the snare and hat bands like any other. discover() finds snares in the low band too, and held to whole semitones in every band, drum_pitch-2's snare (−0.70) came out at −1, 30 cents off.
4. **Distinct.** Copies of a higher-ranked sound under another name are removed: entries correlating > 0.95 with each other (±2 ms), each played at the transposition it was found at. Kits copy each other's sounds, at the same pitch or resampled. Two entries found at different pitches are one recording only when they also agree above 0.95 in the snare band: two different kicks brought to one pitch agree where their tone is, not above it (v1orig-6: 0.952 over the full band, 0.83 in the snare band).

**Handing out identities.** All (template, band) pairs are sorted by their best NCC and handed out strongest first:
- A sample already given to another template goes to this one too only if both match it at ≥ 0.9. That marks it `same_as`: the same sound found twice, perhaps at another pitch.
- Otherwise the template takes its best match in another band.
- This keeps a hat that plays on every kick from being reported as the kick.

**Evidence for the top three:**
- **Per-hit votes.** Which candidate matches each individual hit best, over its first 70 ms. A real match wins nearly every hit, not just the average.
- **Null test.** What is left of the template after subtracting the candidate, aligned at the best gain, in dB.

**Role.** The top match's family, when the template is matched in the snare or hat band at NCC ≥ 0.95 and that family is a drum one (kick, snare, hat, perc). Otherwise the majority of the top five's names, if three agree and the top match is ≥ 0.75. Otherwise the guess. In a 20-beat suite run, 7 hat templates matched to their own hat at 0.976–0.997 had been called kick (5) or perc (2): the five names split, and the guess heard the kick under the hats. The kick band does not tell recordings apart (an open hat's top match there was an fx file at 0.957).

**Kick sources** (`--kick-sources`, off by default). A template given the kick band is also read from drums + bass and from the mix, with the 808 taken out by fit-808's one-shot, and the source whose best match is closest wins. It needs `synth808.json`, so fit-808 must run before match. With fit-808's synth one-shot it found 13 of 20 kicks at the true times, against 12 from the drum stem alone (17 with the true 808 file taken out: an oracle). In a 20-beat suite run both of its switches put an 808 file first.

**Layers.** A template matched in the snare or hat band is also fitted as its top match plus a second file (non-negative gains, `pair_fit`). The best partner is a layer when it takes 3 dB or more off the residual, is no other template's sound, does not correlate with the top match (≥ 0.5), is at most 10 dB louder, and is not heard more often away from the template's hits than at them. With 12 snares carrying a clap at a fixed offset and 12 plain ones (hits at the true times), both sounds were found in 10, and no plain snare got a layer. The suite's clap moves 0–8 ms from hit to hit and is found in neither of its 2 beats. Layers go to `matches.json`; the plan does not play them yet. The kick band is not searched: any sine-like file takes energy off a kick.

On the 8 scenarios of the identification lane (20 beats), templates whose top match is their truth sound's file: 49 of 78 in m1, 75 of 95 now (m2g). The match step takes 49 s a beat at 2 jobs (24 s in m1) and 115 s on PIRAMIT.

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

1. **Reference.** The 808 template's notes (`bass808.json`: the long notes of its most common waveform) are averaged from drums + bass. Demucs puts part of the 808's attack in the drum stem; the bass stem alone made the fit choose a 190 ms attack. An analysis written before `bass808.json` existed falls back to the isolated notes of the commonest pitch (at least 600 ms alone).
2. **Fit.** The voice in [05](05-808-synthesis.md) is fitted by differential evolution (scipy; popsize 20, up to 150 generations; 50 s on PIRAMIT, 51 s a suite beat at 2 jobs). The score is waveform NCC + 0.1 × harmonic error (2nd–12th, /20 dB) + 0.1 × envelope error (/20 dB).
3. **One-shot.** A 3 s sample at −1 dBFS with a 10 ms fade, written for Simpler. Its pitch is the template's settled pitch as `bass808` measured it. The fit's `f_end` replaces it where the two agree within 2 cents, or where the template only had an unsettled tail to read. `synth808.json` keeps both (`measured_hz`, `fit_vs_measured_cents`) and names the one used (`pitch_hz`, `pitch_from`: the measuring method, `fit` or `agree`). On PIRAMIT the template's 32.93 Hz is used; the fit's `f_end` is 6.5 cents over it.
4. **Comparison.** With `--library-808`, the best of the library's 808s (±12 semitones) is compared on the same measures.

## plan

`live.make_plan` turns the analysis into what Live gets.

**Drum pads:**
- **Which sounds.** A template gets a Drum Rack pad when it was detected at least 12 times and at least half its own matched-filter hits reached NCC ≥ 0.8. It also needs either a library match ≥ 0.75, or to be a kick or snare detected 16 times or more, 70 % of them that clean: the backbone is played with the closest sound, however far. The 12-hit gate leaves sounds with few hits unplayed. Open hats (4–8 hits) score 0.979 at analysis and 0.043 in the plan, and tempo72's and tempo95's snares (8 hits) score 0 in the plan (m2g).
- **Pitch.** A pad plays its match at the pitch it was found at: Transpose in whole semitones and Detune in whole cents (`pad_pitch`), as Live 12.4.6 keeps them. drum_pitch's snare (−0.70 semitones) is Transpose −1, Detune +30. With Transpose alone it played 30 cents off, and its hat (+0.25) 25. Over the suite no pad now plays its sound's file off pitch (m2g). Matches written before replica mode carry whole semitones and get Detune 0.
- **Pads** follow GM (kick 36, snare 38, hats 42/44/46, …). A pitched copy (hits with `semitones`) gets the next free pad from 54 up: the same sample, transposed that much further, with the base pad's Detune. On the rolls, full and producer beats (6 beats) the plan's drums F1 at 20 ms is 0.452 with these pads, 0.157 with the copies played unpitched (runs l1b-after3, l1b-nopads3) and 0.145 for one template per moment (m1); 0.455 now. Over hits it is 0.726, 0.670 and 0.602 (0.732 now): the 37 pitched-copy sounds hold 193 of the 1,878 truth hits, so the pads are worth 87 of them (88 of the 193 found now). The mean over sounds counts each copy, of 1–68 hits, as much as a main hat.
- **Twins.** A `same_as` template shares its twin's pad. The same file at another pitch plays on a pad of its own, transposed so far. producer-2's hat 3 semitones down had played at the hat's pitch, and its own pitched copies 300 cents off; re-planned, producer-2's drums F1@20 went 0.579 → 0.665.
- **Velocities** come from the hit gains: 127 at the 90th percentile, 35 at least.
- **Layers** found by `match` are not played yet.

**Notes:**
- Drum notes start where their hits attack (`t_attack`), last up to 1 beat and end before the next note of their pitch. The pad sample's own attack is not taken off: the plan's medians are kick +2.1 ms, snare +1.1, hat +0.6 (m2g).
- **808 notes** (`live.plan_808`) keep their MIDI numbers (the written pitch) and their own lengths.
  - Transpose and Detune put the one-shot's settled pitch on the written note plus the 808's tuning, in whole semitones and whole cents (`live.tune_808`).
  - Detune also takes `heard_cents`: the one-shot falls faster than the 808s it copies, so tuned to settle where the 808 settles it sounds flat over the notes. `heard_cents` is the median, over the notes at the line's most common pitch, of the drop the analysis heard less the one-shot's own. Re-scored on the same plans of the 10 scenarios with an 808, plan note F1@50 went 0.849 → 0.886 against the rule it replaced (bass_low-1 0.333 → 0.949).
  - Each note starts so that the one-shot's attack, played at the note's pitch, lands on the note's attack.
  - One voice: a note ends 0.001 beat before the next starts. Only a line with three legato notes or more is put on Glide (voices 1, Glide Mode "Glide", the median Glide Time), its legato notes held 10 ms over; the reader writes none by default.
- **Melodic notes**: one track per tonal stem of `melodic.json`, with Drift as a placeholder instrument, built muted.
- Times are converted to beats from the start of the audio: beat = seconds × BPM / 60. The reference stems start at beat 0 unwarped, so the notes line up with them by construction.

**Levels.**
- Our drums and 808 are rendered offline at unity gain (Simpler at 0 dB, velocity as linear level): each pad at its Transpose and Detune, the 808 as `render.render_plan` plays it (Transpose, Detune and Glide, cut at each note's end).
- Each is set to the RMS of the stem it replaces.
- Everything, the stems included, is 4 dB down for headroom.
- On the suite (m2g) the plan's levels sit a median 3.67 dB from a perfect kit under its faders (7.18 dB before replica mode). The 808 sits 1.95 dB under it (+0.09 dB in m1): faders matched by whole-track RMS now see the notes' real lengths.
- Live's velocity curve is not linear (measured on 2026-10-19, see [07](07-building-in-live.md)). Neither the levels nor the plan's render apply it yet.

**Locators** go at section starts, counted from the grid's first bar line in its meter; the build rounds them to whole beats.

**Not in the plan:** the meter, and where bar 1 is. Live counts bar 1 from 0 s, so a 3/4 beat is built in 4/4 (the plan's meter is right on 0.909 of the scenarios: meter34 and meter68 are not), and full's first bar line, at 0.4 s, lands 400 ms off.

`midi/drums.mid`, `midi/808.mid` and a file per melodic track (480 PPQ, the plan's tempo) are written too, for `import_midi_file` or any DAW.

On the suite (m2g), strict F1@20 over drum hits and 808 onsets is 0.886 at analysis and 0.667 in the plan; drums alone 0.880 and 0.612; 808 notes F1@50 0.936 and 0.926.

## live-build

See [07](07-building-in-live.md). It refuses unless every track in the open set is empty, then builds every track in one `build_tracks` call; empty tracks already in the set are left as they are. No replica-mode plan has been built in Live yet.

## calibrate and suite

See [03](03-calibration.md). `calibrate` scores six v1 beats with one offset removed per sound. `suite` builds and runs calibration v2: 48 beats in 22 scenarios, scored strictly at analysis and at plan level against every hit, note and chord of their truth (`evaluate`). `--steps` runs a subset of the steps and `--reuse TAG` copies the others' outputs from an earlier run; a reused step whose inputs the run rewrote is marked `reused_stale`.
