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
9. **Names and roles.** Templates are named `sound_1`, `sound_2`, … by number of detections. The role is only a guess here (`kick`/`snare`/`hat`): the median over the hits of the share of energy each adds below 150 Hz and above 5 kHz over the 60 ms before it. Summed instead, the kicks under some hats decided: 33 of the 56 templates that played a hat on 32 suite beats read as kicks (8 of 84 now; 71 % of all guesses right, 57 % before). Matching decides it later.

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
- **Composites.** A template the others explain to 80 % is left out, and its hits go to them.
- **Keeping a hit.** Its gain reaches 25 % of the template's median gain, and the fit loses more without it than the template takes at shifted, empty windows (their 95th percentile).

**Where the kept hits leave something out.** A held template is looked for at its offset under every hit of the template that holds it. Every noisy template (not a tone, not an echo of another) is looked for where its matched filter peaked at NCC 0.3 up to its threshold, and at the onsets of its band. Such a hit needs half the template's median gain (a quarter where what the other hits leave matches the template at its threshold), and the null. It is marked `source: "layer"`, with `under`, the template it was under. For these hits the null is a backstop: of 1,724 probe hits over the gain floors on the 48 beats, it alone rejected 44, all of sounds that no truth sound was assigned to. Two further floors were measured and left out. An NCC of 0.2 with what the others leave, in the template's own band, would remove 23 true probe hits for 13 false ones and 35 of unassigned sounds. A median gain of at least a quarter of the template's own level lost a sound that plays 26–30 dB under the few loud hits its template was made from: 0.75–0.80 → 0.12–0.14 of its hits found (synthetic).

**Tones.** A template with half its energy in the kick band (bands weighted as in the fit), or a quarter there that rings (NCC 0.5 with itself 8–35 ms later), is a tone. Its matched filter has a broad maximum, so it gets no pitched copies and is not looked for under other hits; a kick template's low ring explains a little of almost anything, so it is never held. A kick may ring little and still be a tone (leadin-2: 59 % in the kick band, ringing 0.29); probed as a noisy sound, it gave 65 layer hits, 40 of them false (as a tone, 15, all true). A hit of a tone (or of a low-band template) within 250 ms after a louder one is that one's ringing when it is less than half as loud, or below what the template's own decay holds at that delay (0.70 at 23 ms for a kick on v1-1, which was found again 15–23 ms after every strike). A tone found in the hat band with less than half its energy in the kick band is a hat whose average carries the body of the kicks under some of its hits: it is held, spaced and kept as a hat (not probed).

**Pitched copies.** A noisy template (its copy one semitone off matches it at NCC < 0.5: hats, snares) is also matched 1–5 semitones up and down, transposed as Simpler plays it. Its hits carry `semitones` and are scored as sounds of their own.

A quieter hit at most 35 ms before another of the same sound (no third within 70 ms) is a flam's grace note (`flam_of`).

**Each hit once** (`count_once`, after the fit). A view's hits within 50 ms of its main's are the main's: a view finds its main's hits again at one lag (36–39 ms on 4 suite beats; within 15 ms, full-1's view of the kick kept 43 hits 37 ms late). A template's own ring is no hit either: within 250 ms of an earlier hit, no louder than that hit's ring would be there (the template's level at that lag, never rising again), where nothing starts in its band. A tonal snare's template had matched its own ring 101 times for 32 snares.

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
