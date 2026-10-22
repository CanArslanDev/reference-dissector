# 10 – Roadmap

In rough order of value. Each item names how it will be measured. The calibration-v2 suite ([03](03-calibration.md)) is the answer key for most of them.

Numbers are means over its 22 scenarios on htdemucs stems, "analysis / plan" as in [12](12-replica-mode.md), unless a line says otherwise. "Now" is tag m2g (9ccc759); "before" is base-htdemucs (471fd70), the code before replica mode.

## Done in replica mode

- **An answer key with every answer kept.** Calibration v2 has 48 beats in 22 scenarios (`suite build`). Scores are strict, at analysis and plan level (`evaluate`, `render`).
  - `suite run` reruns any subset of steps on another tag's outputs (`--steps`, `--reuse`), and marks the reused steps whose inputs the run rewrote.
  - Demucs runs are seeded: the same stems every time on one device.
- **The grid** (beat_this, tag m1): beats F 0.715 → 0.962, downbeats F 0.637 → 0.939, meter right 0.91 → 0.95.
- **Drum transcription for kicks and hats** (item 1 before). It was done in place, without a separation or ADT model: templates where sounds start, one per sound, then every moment fit with all the templates at once. That is the old third candidate, templates as fixed bases, on waveforms.
  - Drums F1@20: 0.397 → 0.880 at analysis, 0.283 → 0.612 in the plan.
  - Kick 0.247 → 0.872, hat 0.508 → 0.859, open hat 0.326 → 0.979 (analysis).
  - Kick timing: median +16.6 → 0.0 ms, p90 20.9 → 6.1 ms.
- **Layered and simultaneous hits** (item 6 before). On the 22 beats of the layers, rolls, flams, dynamics, v1, v1orig, producer, full and kick808 scenarios:
  - hats within 20 ms of a snare or clap found: 16 (m1) → 376 of 432;
  - 64th-roll hits: 65 → 165 of 200, from the layer lane's run (l1b-fix) to now.
  - Pitched copies play on pads of their own. Library layers are searched (`matches.json`) but not played.
- **The 808's notes**, read as transpositions of the song's own 808:
  - note F1 (50 ms, 50 cents): 0.278 → 0.936 at analysis, 0.229 → 0.926 in the plan;
  - median pitch error: 55 → 6 cents at analysis;
  - the written note is right for 0.947 of the notes.
- **Sounds at any pitch.** The snare and hat bands are matched to a quarter cent (the speckle index), and pads play their match's Transpose and Detune. No pad that plays its sound's file is off pitch now; before Detune per pad, drum_pitch's snare played 30 cents off.
- **Melodies and chords, started** (item 5 before). Basic Pitch runs per tonal stem: note F1 0.630; chords majmin 0.977 (4 beats). The plan carries the notes as muted MIDI tracks.
- **Levels.** The plan's median distance from a perfect kit: 7.18 → 3.67 dB. Since m2g-vel the plan's velocities follow Live 12.4.6's law at "Vol < Vel" 35 %, which the build sets on every Simpler (3.67 dB as before; not verified in Live).

## Targets

The targets are set for a replica: no listening test backs them yet.

| metric | target | now | gap from now | on htdemucs_ft (m2g-ft) |
|---|---|---|---|---|
| kick, F1@20 | ≥ 0.90 | 0.872 / 0.845 | 0.028 / 0.055 | 0.927 / 0.919 |
| hat, F1@20 (pitched copies included) | ≥ 0.90 | 0.859 / 0.845 | 0.041 / 0.055 | 0.913 / 0.916 |
| 808 note accuracy: note F1 (50 ms, 50 cents) | ≥ 0.95 | 0.936 / 0.926 | 0.014 / 0.024 | 0.924 / 0.910 |
| 808 written note right | ≥ 0.95 | 0.947 | 0.003 | 0.937 |
| onset timing, p90 of the error per sound | ≤ 10 ms | kick 6.1 / 7.9, snare 3.2 / 3.0, hat 3.3 / 4.2, 808 8.0 / 8.0 ms | met on the means. Per scenario, 15 role means are over 10 ms at analysis and 19 in the plan | kick 3.4 / 3.4, snare 3.8 / 2.9, hat 1.7 / 2.5, 808 7.0 / 7.0 ms; 9 and 6 role means over 10 ms |
| chord accuracy, majmin | ≥ 0.85 | 0.977 (4 synthetic beats) | met; not verified on songs | 0.979 |

The timing misses, at analysis level:
- kick: tempo95, glides, drum_pitch, melodic (27.1 ms);
- snare: flams (24.0 ms), producer (21.1 ms);
- hat: tempo160, bass_low (32.0 ms), melodic;
- 808: tempo95, meter68, swing58, glides, dynamics, producer (23.6 ms).

## Next

### 1. The plan and the build, with Live's measured behaviour

Live 12.4.6 was measured on 2026-10-19; the results are recorded in ableton-live-mcp's CLAUDE.md on its replica-mode branch. Against that:

| Live 12.4.6 | the plan and the build now | next |
|---|---|---|
| Transpose keeps whole semitones (a fraction is truncated toward zero). Detune keeps whole cents from −50 to +50. | `pad_pitch` and `tune_808` round both. Pads and the 808 get Detune. | Check them in Live (item 2). |
| A velocity sets a Simpler's level: dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over its Volume. "Vol < Vel" is 35 % on a Simpler made with `insert_device` and 45 % on a pad loaded through the browser. | Done offline (m2g-vel): the build sends "Vol < Vel" 35 % to every pad and to the 808, the velocities invert the law, and `render.py` and `estimate_levels` play it. Plan levels 3.67 dB from a perfect kit; with v/127 velocities, Live would have played m2g's plans 9.30 dB off (computed from the law). | Check it in Live (item 2). |
| Snap, on by default, ends a sample 10–30 ms early. | Not set. | Snap off on every Simpler. |
| A choke group cuts the sounding pad with a linear 1 ms fade (groups 0–16). | No choke groups. `render.py` fades a cut over 5 ms. | Choke groups for the hats (rolls, producer and full choke their open hats), and a 1 ms fade in `render.py`. |
| Glide slides linearly in semitones over exactly the Glide Time, with one voice and overlapping notes. Notes 25 ms apart jump. | A legato note's predecessor is held 10 ms over it, and voices are set to 1 after `build_tracks`. Slides are off by default. | Check it in Live. Slides that score better than merging the unattacked notes. |
| `sample_path` loads the exact file and reads it back. A pad built that way starts in Classic mode with its filter on. | Pads load through Live's browser by file name (ableton-live-mcp 2.2.0). | `sample_path`, with `playback_mode` and the filter set explicitly. |
| `signature` sets the meter before anything is built. | The plan holds no meter, and Live's bar 1 sits at 0 s. | The meter, and the song's first bar line on bar 1. |

The plan's own gaps:
- **The pad gate.** A template gets a pad from 12 hits on. Open hats (4–8 hits) score 0.043 in the plan against 0.979 in the analysis, and tempo72's and tempo95's snares (8 hits) score 0. Metric: plan F1@20 per role. Target: the analysis's score; the gap for open hats is 0.936.
- **Note timing.** Drum notes go to `t_attack` without the pad sample's own attack taken out: plan medians kick +2.1 ms, snare +1.1 ms, hat +0.6 ms.
- **The 808's level.** It sits 1.97 dB under a perfect kit (m2g-vel), because its fader matches the bass stem's whole-track RMS. Match faders the way levels are scored.
- **Library layers.** Play `matches.json`'s layer on a second pad with the same notes, at its gain and lag.

**Measured by:** plan-level scores (drums F1@20 0.612 against 0.880 at analysis; kick and hat, gaps above), and the levels: 3.67 dB from a perfect kit now. Target for the levels: under 1 dB, an estimate.

### 2. The build, checked in Live

No replica-mode plan has been built in Live and recorded yet.
- Build m2g plans of a few suite beats into an empty Live 12.4.6 set, with item 1's settings.
- Record them (`capture_stems`, `render_and_analyze`) and score the recording as the plan is scored.
- Check pad Detune, the 808's Transpose, Detune, Glide and voices, the meter, bar 1 and the levels.

**Measured by:** the recording's scores against the offline render's: strict F1@20, levels (dB), pad and 808 pitch (cents). Target: within 0.01 F1, 0.5 dB and 1 cent, an estimate. In the Live battery, played pitches followed the stored values within 0.002 cents and notes started within 0.04 ms. Gap: not measured.

### 3. The closed loop

1. Record the build.
2. Measure it against the reference: the suite's scores where a truth exists; on a song, `compare`'s measures against the reference's own spread ([11](11-style-beat-lessons.md)) and a listening report per sound ([06](06-listening-for-a-model.md)).
3. Turn the differences into edits (level, transpose and detune, decay, EQ, drive, reverb, another sample) and apply them through ableton-live-mcp.
4. Rebuild, and repeat while the distance falls.

**Measured by:** the distance per iteration: levels against a perfect kit, cents, timing p90, and `compare`'s measures beyond the reference's spread. Target: every iteration lowers it, and it ends within the reference's own spread (an estimate). Gap: not started.

### 4. Speed

Now (2 jobs, Apple M4): 142 s a beat after separation, 56 minutes for the 48 beats.
- analyze takes 38.7 s (5.7 s in m1). The 808 reader is about 30 s of it, by its lane's measure.
- match takes 49.1 s (24.4 s in m1).
- fit-808 takes 50.9 s. In m1 it was 56.2 of 90.2 s a beat (62 %); now it is 36 %.

A probe on 4 m1 beats (v1-1, kick808-1, glides-1, bass_low-2) re-ran `synth808.fit`:
- differential evolution with popsize 10 instead of 20 took 26.0–35.5 s against 57.0–66.2 s (1.9–2.2× faster), at equal mean loss (0.2673 against 0.2686);
- 60 generations instead of 150 was 2.2–2.7× faster, at 0.4–1.8 % more loss.

Not run on the 48 beats yet.

**Next:** a 48-beat run with popsize 10, a fit cache keyed on its inputs, and the 808 reader's and match's costs.
**Measured by:** seconds per beat, the fit's loss, and the plan's 808 scores. Target: fit-808 under 30 s a beat, with the mean loss within +1 % and no scenario more than 5 % worse (an estimate).

### 5. The open issues from the merge

Measured on m2g; [12](12-replica-mode.md) has the details.

| issue | metric now | target |
|---|---|---|
| melodic-1's kick became a composite of a snare-band residual view of itself (templates are taken by hit count) | kick F1@20 on melodic-1: 0.748 (0.929 in the discovery lane) | 0.90 |
| Merges, composites and `distinct()` choose between templates by energy and hit count, not alignment: layers-2's snare has no pad | plan snare F1@20 0.813 (0.834 in the discovery lane) | 0.90 |
| Hats held by a snare's top end at 15–24 ms (melodic-1, leadin-1, tempo160-1, producer); bass_low's hats almost missing | hat timing p90 3.3 ms (2.55 ms in the lane); bass_low hat F1@20 0.006 | ≤ 10 ms in every scenario; 0.90 |
| A sound that always plays with another has no template: the layers scenario's clap | clap F1@20 0 on layers, 0.500 over the suite | – |
| Pitched copies of residual templates are off | re-planned on 20 beats: rolls-2 +0.054, producer-2 +0.051, rolls-1 −0.043 | – |
| Residual templates on melodic-loop transients left in the drum stem | false hits 35.6 a beat on the layer lane's 22 beats (31.5 in that lane); 1.96 sounds a beat that play no truth sound (0.71 with a pad) | – |
| Chords follow the sections, which follow the drum hits | full-2 majmin 0.999 → 0.961 in an intermediate run, with the same notes | ≥ 0.85 |
| bass_low's 808 never settles: its template reads 44–60 cents sharp | written note right 0.487 on bass_low | ≥ 0.95 |
| The 808's slides (item 4 before). Legato notes are off by default, so no plan plays a slide; the plan and the build carry Glide Mode, Glide Time and one voice for a line with three legato notes or more, not verified in Live. The rule this replaced was right 6 times in 79; the opt-in rule (`slides=True`) made 18 legato notes, 6 of them right, on the beats its thresholds were chosen on | legato recall 0 (the truth: 24 on 5 beats) | slides that beat merging on note F1 |
| The 3 s one-shot leaves long legato chains' tails silent | not scored | – |
| Library layers are not played; kick sources are off by default | 75 of 95 templates with the right file (identify lane's 8 scenarios) | – |
| The plan holds no meter; tempo72's bar line is wrong; meter68 is read in 4/4 at x2/3; the seed-1 swing beats read straight 8ths | plan meter right 0.909, plan downbeat F 0.892; tempo72 downbeat F 0 | – |
| htdemucs_ft has better drums but worse 808 notes on short notes (bass_low) and on producer | plan drums +0.039 [+0.017, +0.062]; 808 note F1@50 0.936 → 0.924 | the default changes when a model is better on both drums and 808 notes |

Songs have no truth to score against: every number here comes from synthetic beats.

## Later

These are from before replica mode, and still open.

1. **Symmetric comparison.** *Started:* `refdissect compare` measures two stem sets the same way against the reference's own spread ([11](11-style-beat-lessons.md)).
   - Still to do: master the rebuild like the reference, and per-sound null tests.
   - Render the rebuilt parts from Live (ableton-live-mcp's `render_and_analyze` / `capture_stems`) and master them like the reference. Then separate them with the same Demucs and compare stems with stems: the separation's artifacts are then on both sides.
   - **Measured by:** per-sound null tests and descriptor differences before/after.
2. **A grittier 808.** Extend the voice with asymmetric drive and a weighted upper-harmonic term.
   - **Measured by:** harmonics 6–12 within 3 dB, glide shape.
3. **Melodic instruments.** The plan's melodic tracks are muted and play a placeholder (Drift). Choose instruments by CLAP (`find_similar_sounds`), or synthesize them.
   - **Measured by:** note F1 against hand-checked bars; chroma distance.
4. **Style mode and profiles** (the original brief):
   - pattern statistics per artist or album (hat roll rate, kick placement, 808 glide rate, chord vocabulary, section lengths) from several songs;
   - new beats generated from those statistics, not from one song.
5. **Packaging.**
   - Expose `analyze` / `match` / `plan` as jobs in ableton-live-mcp (`analyze_reference`, `build_from_blueprint`), so an AI client can run "make a beat like this song" end to end.
   - Run the packaged `live-build` against Live in CI-like checks.
6. **Style beats** ([11](11-style-beat-lessons.md)).
   - Done: `listening-round` (numbered A/B files and a manifest), `compare`, `space` and `mix-gains` commands.
   - Still to do: generating the 8-bar candidates themselves from a plan.
   - **Measured by:** the listener's picks, and ducking and balance against the reference.

**Not planned:** publishing transcriptions or audio of songs, for the reasons in [09](09-data-and-copyright.md).
