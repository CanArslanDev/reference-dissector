# 10 – Roadmap

In rough order of value. Each item names how it will be measured. The calibration-v2 suite ([03](03-calibration.md)) is the answer key for most of them.

Numbers are means over its 22 scenarios on htdemucs stems, "analysis / plan" as in [12](12-replica-mode.md), unless a line says otherwise. "Now" is tag m5 (a4f2358): replica mode with the fixes of 2026-10-29, the last of them from the first live check. m4 (e7ea748) is the same before those: one voice per pad, the 808's release, the plan's meter, and pads loaded by their content. m3 (6d96bc5) is m4 before its last two, a level per drum pad and the 808 one-shot on its template's periods. "Before" is base-htdemucs (471fd70), the code before replica mode. The fixes were measured against m2g-vel: the merged code of 9ccc759, its plans under Live's velocity law.

## Done in replica mode

- **An answer key with every answer kept.** Calibration v2 has 48 beats in 22 scenarios (`suite build`). Scores are strict, at analysis and plan level (`evaluate`, `render`).
  - `suite run` reruns any subset of steps on another tag's outputs (`--steps`, `--reuse`), and marks the reused steps whose inputs the run rewrote.
  - Demucs runs are seeded: the same stems every time on one device.
- **The grid** (beat_this, tag m1): beats F 0.715 → 0.962, downbeats F 0.637 → 0.939, meter right 0.91 → 0.95.
- **Drum transcription for kicks and hats** (item 1 before). It was done in place, without a separation or ADT model: templates where sounds start, one per sound, then every moment fit with all the templates at once. That is the old third candidate, templates as fixed bases, on waveforms.
  - Drums F1@20: 0.397 → 0.883 at analysis, 0.283 → 0.868 in the plan.
  - Kick 0.247 → 0.882, hat 0.508 → 0.860, open hat 0.326 → 0.980 (analysis).
  - Kick timing: median +16.6 → +0.1 ms, p90 20.9 → 5.3 ms.
  - Each view of a sound found in another band is fit right after its main (2026-10-29): melodic-1's kick F1@20 0.748 → 0.929, the suite's kick p90 6.13 → 5.28 ms.
- **Layered and simultaneous hits** (item 6 before). On the 22 beats of the layers, rolls, flams, dynamics, v1, v1orig, producer, full and kick808 scenarios:
  - hats within 20 ms of a snare or clap found: 16 (m1) → 376 of 432 (m2g);
  - 64th-roll hits: 65 → 165 of 200, from the layer lane's run (l1b-fix) to m2g.
  - Pitched copies play on pads of their own. Library layers are searched (`matches.json`) but not played.
- **The 808's notes**, read as transpositions of the song's own 808:
  - note F1 (50 ms, 50 cents): 0.278 → 0.936 at analysis, 0.229 → 0.934 in the plan;
  - median pitch error: 55 → 6 cents at analysis, 70 → 4.5 cents in the plan;
  - the written note is right for 0.970 of the notes (0.947 at m2g-vel). Where a template's own notes never settle, its settled pitch is read on the notes that reach past it: bass_low 0.487 → 0.987.
  - A note ends where the 808 falls silent, and a 9 dB rise attacks one. On PIRAMIT no note is held over 0.2 s or more of silence (4 before), and the 808 alone and the breakdown are two sections again.
- **Sounds at any pitch.** The snare and hat bands are matched to a quarter cent (the speckle index), and pads play their match's Transpose and Detune. No pad that plays its sound's file is off pitch now; before Detune per pad, drum_pitch's snare played 30 cents off.
- **Files a pad can play.** A library file that peaks under −60 dBFS is never a match, and a band whose five best sounds tie within 0.002 NCC names none of them. No pad plays a near-silent file (1 of 227 at m3a), and full-1's and melodic-1's kick pads play the truth's kick file, not a synth-bass or a tied file.
- **Drum pads on evidence** (the 12-hit pad gate before): 3 hits or one per 16 bars, a clean template, a close match or a place in the backbone; a sound of the residual pass must recur bar to bar. The backbone is planned first, and each strike is played once. m2g-vel → m3:
  - plan drums F1@20 0.612 → 0.868; open hats 0.043 → 0.981; tempo72's and tempo95's snares 0 → 1.0; crash, tom and rim 0 → 0.5;
  - false hits 23.0 → 9.6 a beat; sounds with a pad that play no truth sound 0.71 → 0.05 a beat. That last is a suite figure: with recurrence waived, as a loop on the beat's tempo would face, 0.37 (an estimate).
- **Drum note timing.** A drum note starts as much before its hit's attack as its pad's sample takes to attack, so the plan times each sound where the analysis does: plan kick median +2.06 → −0.34 ms and p90 7.88 → 5.58 ms, hat p90 4.24 → 2.86 ms.
- **Melodies and chords, started** (item 5 before). Basic Pitch runs per tonal stem: note F1 0.630; chords majmin 0.977 (4 beats). The plan carries the notes as muted MIDI tracks.
- **Levels.** The plan's median distance from a perfect kit: 7.18 → 3.67 dB at m2g-vel, whose velocities follow Live 12.4.6's law at "Vol < Vel" 35 %, which the build sets on every Simpler (in Live the law held from velocity 35 to 127). The pad gate made it worse again (4.09 dB at m3), a level per pad brought it to 1.25 dB (m4, below), and Live's playback in the estimate to 1.20 dB (m5, below).
- **A level per drum pad** (m4; item 1 before). Each pad plays where the stems hold its sound: a kick on drums + bass less the plan's own 808, every other sound on the drum stem, the loudest pad at 0 dB, the drum fader stopped where the loudest pad meets its stem.
  - m3 → m4: 4.09 → 1.25 dB from a perfect kit (htdemucs_ft 3.70 → 1.23), balance 3.03 → 0.55 dB (2.84 → 0.50); no scenario of either model further off on either.
  - Against the kit, hats +6.23 → −1.10 dB and open hats +11.57 → −0.09 dB (htdemucs_ft +6.47 → −0.69, +10.54 → −0.83).
  - In Live a pad's Volume is held a display step under the text sent and played within 0.003 dB of what is held (the first live check, below).
- **The 808's plan pitch** (m4; item 2 before). The one-shot follows its template's own periods until the drop has settled, laid out as the template is.
  - m3 → m4, plan: 808 note F1@50 0.921 → 0.934 (htdemucs_ft 0.910 → 0.934), median error 8.3 → 4.5 cents (13.6 → 9.6), settled within 50 cents 0.925 → 0.970 (0.935 → 0.958).
  - bass_low-1, m3's regression: note F1@50 0.718 → 1.0 (0.449 → 0.981); its Detune correction by note length within 0.2 cents of 0 (htdemucs_ft: 3.8 for notes under 120 ms). The line's correction lies within 8 cents of 0 on 48 of 48 beats (47; 37 and 43 at m3).
  - In Live the one-shot played at `render.py`'s pitch within 0.11 cents (the first live check), and its notes within 0.11 ms of `render.py`'s (the second). Not judged by ear.
- **The build in Live, checked** (2026-10-29; item 3 before; [07](07-building-in-live.md#live-behaviour-measured-here)).
  - The first live check built eight m4 plans, two probes and PIRAMIT's plan with the build's calls (74931ce) and recorded them. As the plan assumes: every note; Transpose and Detune within 0.1 cent on pads and 0.11 cents on the 808; pad Volumes within a display step (0.05 dB at most); the velocity law from 35 to 127; Glide on a probe within 1.04 cents (p90). Not as assumed: two pads played other files of their names, and PIRAMIT's build stopped at a name the browser's search did not return; Snap ended pads 4.7–27 ms early; a pad plays one voice, and a hit that cuts its sounding hit starts 1 ms late (391 hits); the 808's filter delayed its notes 0.13–0.46 ms by pitch; its release lasts 35.7 ms, and a note after one of its pitch restarts 1 ms late; a 3/4 plan was built in 4/4.
  - Fixed (m5): each pad's file goes into the User Library under a name of its content and is read back; Snap and the 808's filter are off; one voice per pad and the 808's release and restart are in `render.py` and the estimates, and the notes Live starts late start 1 ms early; the plan carries its meter; a failed or refused build says why.
  - The second live check built four m5 plans and PIRAMIT's (60b485f) and recorded them: 30 of 30 Simplers held their plan's files; 1,320 drum hits within 0.04 ms of `render.py`'s; 408 808 notes within 0.11 ms, at −43.1 to −50.8 dB of residual (medians per beat); drum F1@20 per role equal to `render.py`'s; the level scores within −0.25 to +0.12 dB (distance from a perfect kit) and −0.12 to +0.40 dB (balance) of `render.py`'s, all of it Simpler's top-end loss.
- **The plan's meter** (m5). The plan carries the grid's meter and the build sets it: the plan's meter right 0.909 → 0.955, downbeat F 0.892 → 0.924, meter34 built in 3/4.
- **Speed.** A suite beat takes 112.6 s after separation at 2 jobs, against 141.9 s at m2g (m3a; not a controlled timing): match 49.1 → 28.0 s, analyze 38.7 → 34.4 s, fit-808 50.9 → 46.2 s. The speed changes left every output the same on all 48 beats (and on the 8 htdemucs_ft beats run).
- **Kept results.** The 808 reader's results and the 808 fits are kept by the `analyze` and `fit-808` commands unless `--no-cache`, and by `suite run` only with `--cache`, under a key of the inputs, the settings, the versions and the package's code as the process loaded it. Read back, analyze takes 8.1 s and fit-808 1.0 s a suite beat; a run records what it read back and leaves it out of its seconds.

## Targets

The targets are set for a replica: no listening test backs them yet.

| metric | target | now | gap from now | on htdemucs_ft (m5-ft) |
|---|---|---|---|---|
| kick, F1@20 | ≥ 0.90 | 0.882 / 0.871 | 0.018 / 0.029 | 0.935 / 0.925 |
| hat, F1@20 (pitched copies included) | ≥ 0.90 | 0.860 / 0.861 | 0.040 / 0.039 | 0.913 / 0.919 |
| 808 note accuracy: note F1 (50 ms, 50 cents) | ≥ 0.95 | 0.936 / 0.934 | 0.014 / 0.016 | 0.924 / 0.934 |
| 808 written note right | ≥ 0.95 | 0.970 | met | 0.959 |
| onset timing, p90 of the error per sound | ≤ 10 ms | kick 5.3 / 5.6, snare 3.2 / 3.2, hat 3.3 / 2.9, 808 8.0 / 8.0 ms | met on the means. Per scenario, 14 role means are over 10 ms at analysis and 13 in the plan | kick 3.3 / 2.2, snare 3.8 / 3.3, hat 1.7 / 1.4, 808 7.0 / 7.0 ms; 9 and 7 role means over 10 ms |
| chord accuracy, majmin | ≥ 0.85 | 0.977 (4 synthetic beats) | met; not verified on songs | 0.979 |
| plan levels, median distance from a perfect kit | under 1 dB (an estimate) | 1.20 dB (1.25 at m4, 4.09 at m3, 3.67 at m2g-vel) | 0.20 dB | 1.15 dB (1.23; 3.70; 3.21) |
| the recording in Live against the offline render | within 0.01 F1, 0.5 dB and 1 cent (an estimate) | drum F1@20 equal per role; level scores −0.25 to +0.12 dB (distance from the kit), −0.12 to +0.40 dB (balance); 808 notes within 0.11 ms; the 808's pitch within 0.11 cents on the first live check (m4 plans), not read on the second | met on the four m5 beats built, the pitch on the first check's m4 beats; per pad, a bright hat 0.97 dB under | not built |

The timing misses, at analysis level:
- kick: tempo95 (11.4 ms), drum_pitch (11.8 ms), melodic (17.1 ms; 27.1 at m2g);
- snare: flams (24.0 ms), producer (21.1 ms);
- hat: tempo160 (10.4 ms), bass_low (32.0 ms), melodic (11.1 ms);
- 808: tempo95, meter68, swing58, glides, dynamics, producer (23.6 ms).

The plan misses the same, but for tempo160's hat. glides' kick, 13.4 ms at m2g, is 4.8 ms now.

## Next

### 1. The plan and the build, with Live's measured behaviour

Live 12.4.6 was measured on 2026-10-19, by a battery recorded in ableton-live-mcp's CLAUDE.md on its replica-mode branch, and on 2026-10-29, by two live checks of built plans ([07](07-building-in-live.md#live-behaviour-measured-here)). Against that:

| Live 12.4.6 | the plan and the build now | next |
|---|---|---|
| Transpose keeps whole semitones (a fraction is truncated toward zero). Detune keeps whole cents from −50 to +50. | `pad_pitch` and `tune_808` round both. Pads and the 808 get Detune; in Live pads play within 0.1 cent of the plan's pitch and the 808 within 0.11 cents of `render.py` (first live check). | Done. |
| A velocity sets a Simpler's level: dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over its Volume. "Vol < Vel" is 35 % on a Simpler made with `insert_device` and 45 % on a pad loaded through the browser. | Done (m2g-vel): the build sends "Vol < Vel" 35 % to every pad and to the 808, the velocities invert the law, and `render.py` and `estimate_levels` play it. In Live "Vol < Vel" reads back 0.35 and the law holds from velocity 35 to 127. A pad's Volume as text is held a display step under it (0.005 dB above −10 dB, 0.05 dB from −10 dB down). | Send the Volume as a number and read it back, or keep the step (≤ 0.05 dB). |
| Snap, on by default, ends a sample 10–30 ms early (4.7–27 ms on four pads of the first live check). | Off on every Simpler (m5): in Live hits play their whole files. Kicks and snares end within 0.04 ms of `render.py`'s (medians per pad); hats up to 0.8 ms earlier, likely as Simpler's top-end loss lowers their tails. | Done. |
| A choke group cuts the sounding pad with a linear 1 ms fade (groups 0–16). | No choke groups. `render.py` fades a cut over 5 ms. | Choke groups for the hats (rolls, producer and full choke their open hats), and a 1 ms fade in `render.py`. |
| Glide slides linearly in semitones over exactly the Glide Time, with one voice and overlapping notes. Notes 25 ms apart jump. | A legato note's predecessor is held 10 ms over it, and voices are set to 1 after `build_tracks`; on a probe in Live the slide played within 1.04 cents (p90) of `render.py`. Slides are off by default. | Slides that score better than merging the unattacked notes, and a plan with Glide built in Live. |
| `sample_path` loads the exact file and reads it back. A pad built that way starts in Classic mode with its filter on. | ableton-live-mcp 2.2.0 loads by name, and the browser held other files of a pad's name (2 pads of the first check). Each pad's file now goes into the User Library under a name of its content and is read back (m5): 30 of 30 Simplers held their files on the second check. | `sample_path` once ableton-live-mcp releases it, which spares the browser's search (its time for a content name was not measured). |
| `signature` sets the meter before anything is built. | The plan carries its meter, and the build sets it as Live's time signature (m5): meter34 is built in 3/4. Live's bar 1 still sits at 0 s. | The song's first bar line on bar 1. |
| A pad plays one voice: a hit that cuts its pad's sounding hit starts 1 ms late (391 hits). The 808 restarts a note of its pitch the same way (110 notes). | In `render.py` and the estimates, and such notes start 1 ms early (m5): in Live the 254 such hits of the second check started 0.004–0.019 ms from `render.py`'s (medians per pad). | Done. |
| The 808's Simpler, loaded through the browser, releases a note over 35.7 ms (1.683 dB a ms) and filters it (0.13–0.46 ms late by pitch). | The release is sent and played by `render.py`, the filter off (m5): in Live 408 notes within 0.11 ms of `render.py`'s. | Done. |
| Simpler loses highs: 0.32 dB at 8–12 kHz on a 48 kHz file, 1.6–1.8 dB at 12–16 kHz on a 44.1 kHz one. | Not modelled: in Live hats played 0.06–0.97 dB and snares 0.03–0.16 dB under `render.py`. | Model it per sample rate, or Simpler's high-quality interpolation if the LOM shows it. |
| A new cue snaps to the Arrangement's grid, which follows its zoom; a cue's time cannot be set. | A locator Live moves is deleted and reported (m5): 3 of meter34-1's 5 in Live. | Zoom the Arrangement in before the locators (a probe placed one so), and put the song's first bar line on bar 1. |
| Live's Remote Script serves at most 8 clients. | `live-build` says so when it is refused; both live checks sent its calls through their own session. | `live-build` end to end (item 3). |

The plan's own gaps:
- **Levels per beat.** A level per pad (done, m4) brought the plan to 1.25 dB from a perfect kit (htdemucs_ft 1.23) and balance 0.55 dB (0.50), and Live's playback in the estimate (m5) to 1.20 dB (1.15) and 0.51 dB (0.48). What stays open:
  - dynamics, 1.27 → 2.16 dB from the kit at m5: one voice per pad plays dynamics-2's drum track 1.3 dB quieter at unity, and its RMS-matched fader rises by as much;
  - the kick against the snare (at m4, not counted again at m5): further off than at m3 on 2 of 42 beats (6 of 43). On bass_low-2 the kick's hits play the hat's file (the analysis gave its kick sound the hat's file, and its hats, found 29 ms early, a kick file that is neither sound's); on tempo95-2 the kick plays where it did at m3 and the snare 2.2 dB louder: read under the plan's own 808, the kick reads 1.74 dB lower under the new one-shot than under m3's, and the RMS-matched drum fader made that up, which lifted the other pads;
  - balance is each sound against the median sound of both stem groups, whose kit offsets differ (the htdemucs drum stem 5.52 dB under the truth's drums, the bass stem 0.96 dB under the truth's bass): the 808's balance reads +2.50 dB (+2.70 at m4, −1.76 at m3). Score balance within each group.
  - Metric: per beat, `levels.kick.vs_kit_db` minus `levels.snare.vs_kit_db`; `levels.vs_kit_abs_db` and `levels.balance_abs_db`, both models. Target: no beat further off than at m3, then under 1 dB (an estimate).
- **The 808's level.** It sits 1.87 dB under a perfect kit (htdemucs_ft 2.01; m4 and m5), because its fader matches the bass stem's whole-track RMS. Match faders the way levels are scored.
- **Library layers.** Play `matches.json`'s layer on a second pad with the same notes, at its gain and lag.
- **One strike, by sound rather than by time.** The backbone's kick and snare are played once by a 15 ms timing rule, and a strike is played once per pad. Linking the views of one sound in the analysis (`same_sound_as`) would decide it there: a real kick layered on every backbone kick is not played now, and one recording on two pads can still sound twice (PIRAMIT).

**Measured by:** plan-level scores (drums F1@20 0.868 against 0.883 at analysis), and the levels: 1.20 dB from a perfect kit now.

### 2. The 808 in the plan: a note's own cents

m3's regression, bass_low-1 played a median 41.7 cents off, is resolved (m4, above): its plan note F1@50 is 1.0, and the suite's 0.934 (htdemucs_ft 0.934). One Detune per Simpler still plays every note at the line's tuning:
- **full.** The analysis writes one note a semitone high and hears 5 short notes 41 cents low; m3's one-shot, falling too fast, happened to offset both. Plan note F1@50 0.890 → 0.851 (htdemucs_ft 0.910 → 0.890).
- **producer.** Plan median error 6.8 → 8.2 cents (15.2 → 21.1), with its note F1@50 up, 0.770 → 0.804 (0.698 → 0.770). Not investigated further.
- **bass_low-2 on htdemucs_ft.** Its template is read from its tail, 57 cents flat; bass_low's plan note F1@50 there is 0.513 (bass_low-1 0.981).
- **By ear**, not checked: the one-shot's pitch step at its first crossing followed, and the quiet or cut head that puts its attack where the template's is.
- **The yardstick** reads a 110 ms note under 30 Hz over less than one period (`scenarios.tuning`), so its reading depends on phase: 89 and 160 cents for one drop, on two near-identical fitted one-shots.

**Measured by:** the plan's 808 note F1@50, median |cents| and settled pitch within 50 cents, overall and per scenario, both models. Target: keep the written note (0.970; 0.959), with no plan-level 808 number of any scenario worse than m3's by more than 0.01 (1 cent).

### 3. The build in Live

Checked on 2026-10-29 by two live checks ([07](07-building-in-live.md#live-behaviour-measured-here); done, above). Still open:
- `refdissect live-build` end to end. Live's Remote Script served 8 clients on both checks, which sent its calls through their own session: its read-back, locator deletion and failure messages ran as the same commands, not as its code.
- One `build_tracks` call of a song's size: PIRAMIT's 1,881 notes went as 5 calls. A build that stops half way, and its message.
- htdemucs_ft plans, a plan with Glide, choke groups.
- The locators off Live's grid (item 1), and how long the browser's search takes for a content name (not timed).
- PIRAMIT by ear.

**Measured by:** the recording's scores against the offline render's, read off the audio: strict F1@20 per role, levels (dB), pad and 808 pitch (cents), onset lags (ms); and the read-back of files, settings, notes and locators. Target: within 0.01 F1, 0.5 dB and 1 cent (an estimate), every Simpler holding its file, every locator where planned. Now, on four m5 plans (htdemucs) and PIRAMIT's: F1 equal on every role of the four suite beats; the level scores within 0.40 dB (balance), a bright hat 0.97 dB under per pad; hits within 0.04 ms and 808 notes within 0.11 ms; 30 of 30 files; 26 of 29 locators (3 of meter34-1's deleted). Gap: Simpler's top-end loss, 3 locators, and no build by `live-build` itself.

### 4. The closed loop

1. Record the build.
2. Measure it against the reference: the suite's scores where a truth exists; on a song, `compare`'s measures against the reference's own spread ([11](11-style-beat-lessons.md)) and a listening report per sound ([06](06-listening-for-a-model.md)).
3. Turn the differences into edits (level, transpose and detune, decay, EQ, drive, reverb, another sample) and apply them through ableton-live-mcp.
4. Rebuild, and repeat while the distance falls.

**Measured by:** the distance per iteration: levels against a perfect kit, cents, timing p90, and `compare`'s measures beyond the reference's spread. Target: every iteration lowers it, and it ends within the reference's own spread (an estimate). Gap: not started.

### 5. Speed

Now (m3a, 2 jobs, Apple M4 with Live open): 112.6 s a beat after separation, 45 minutes for the 48 beats (m2g: 141.9 s and 56 minutes, on a quiet machine).
- analyze takes 34.4 s (38.7 at m2g, 5.7 in m1). The 808 reader fits each start and transposition once: 21 % of its fits on v1-1 had repeated one already.
- match takes 28.0 s (49.1 at m2g, 24.4 in m1): a band's template windows are searched together, and a transposed candidate is resampled from the start its window holds. The silent-file check adds 4.7 % with an index built before its floor.
- fit-808 takes 46.2 s (50.9 at m2g): its band filter, window and time axis are made once. In m1 it was 56.2 of 90.2 s a beat (62 %); now it is 41 %.
- Done: kept results, read back by the `analyze` and `fit-808` commands and by `suite run --cache` (above).

A probe on 4 m1 beats (v1-1, kick808-1, glides-1, bass_low-2) re-ran `synth808.fit`:
- differential evolution with popsize 10 instead of 20 took 26.0–35.5 s against 57.0–66.2 s (1.9–2.2× faster), at equal mean loss (0.2673 against 0.2686);
- 60 generations instead of 150 was 2.2–2.7× faster, at 0.4–1.8 % more loss.

Not run on the 48 beats yet.

**Next:**
- a 48-beat run with popsize 10;
- a controlled timing of the whole suite (m3a shared the machine with Live, m2g did not);
- the tests in parallel (pytest-xdist): 356 passing tests take 328 s in one process (8fb09dc);
- a cache key per step's own modules: with the whole package hashed, any edit makes every kept result stale;
- match's own inverse transform per template and block (22 of 61 s profiled on v1-1), and the start of beat_this's and Basic Pitch's uv environments.

**Measured by:** seconds per beat, the fit's loss, and the plan's 808 scores. Target: fit-808 under 30 s a beat, with the mean loss within +1 % and no scenario more than 5 % worse (an estimate).

### 6. The open issues

Measured on m5 where m5 moved a number (the plan's levels, meter and 808 timing) and on m4 elsewhere, in brackets against m3 for what m4 changed (the plan's 808 and levels) and against m2g-vel for the rest, unless named; [12](12-replica-mode.md#known-limits) has the details. m3's two regressions, the plan's levels and its 808 pitch on bass_low-1, are done (above).

| issue | metric now | target |
|---|---|---|
| The kick against the snare per beat (item 1): further off than at m3 on bass_low-2, whose kick's hits play the hat's file, and on tempo95-2, whose snare plays 2.2 dB louder than at m3 and its kick where it did (the kick reads 1.74 dB lower under the new one-shot, and the RMS-matched drum fader lifted the other pads) | at m4: mean over scenarios of per-beat \|kick − snare\| against the kit 1.45 dB (4.67), further off than at m3 on 2 of 42 beats; htdemucs_ft 1.16 dB (2.86), 6 of 43 | no beat further off than at m3 |
| Levels per role in single scenarios: bass_low's kick (the hat's file on its hits); on htdemucs_ft, dynamics' and tempo72's snares | bass_low kick −13.5 dB against the kit (−15.4 at m4, −12.1 at m3), snare −0.6 (−2.5, −1.3); htdemucs_ft dynamics snare −4.3 (−3.8, −1.0), tempo72 snare −2.9 (−2.9, −0.4) | within 1 dB of the kit (an estimate) |
| Balance is read across the drum and bass groups, whose kit offsets differ (htdemucs drums 5.52 dB under the truth's, bass 0.96 dB) | the 808's balance +2.50 dB (+2.70 at m4; −1.76); htdemucs_ft −0.14 (−0.05; −3.96) | balance scored within each stem group |
| The 808's level: its fader matches the bass stem's whole-track RMS (item 1) | the 808 −1.87 dB against the kit (−1.98); htdemucs_ft −2.01 (−2.24) | match faders the way levels are scored |
| A kick is read under the plan's own 808, so a change to the one-shot moves the drum levels | against the levels alone: 109 of 227 pad Volumes moved, by up to 1.7 dB; the distance from the kit worse in 11 of 22 scenarios, by up to 0.20 dB, better overall (1.27 → 1.25 dB) | measure the levels again after every 808 change |
| A kick with no plan 808 to read under (no one-shot, or no bass stem) is read with the 808 in it, or on the drum stem alone | no suite beat lacks either | – |
| Simpler's top-end loss is not in `render.py` (item 1) | in Live hats 0.06–0.97 dB and snares 0.03–0.16 dB under `render.py`; v1orig-3's balance 0.17 → 0.57 dB | every pad within 0.5 dB of the offline render (an estimate) |
| One voice per pad moved dynamics (m4 → m5) | dynamics 1.27 → 2.16 dB from the kit; dynamics-2 0.34 → 2.03 dB | no scenario further off than at m4 |
| A pad is read against its hits summed, and a late hit where the plan writes it, 1 ms early | read where Live plays them, 37 pad Volumes of 15 m5 plans would move by −0.6 to +0.5 dB (htdemucs_ft 41 of 16, −0.2 to +0.6); against the hits cut, the plans sat 1.30 dB (1.27) from a perfect kit | read where Live plays them, no level worse |
| The 808 reader and the plan's estimate disagree on short restarted notes | bass_low-1: 64 of 156 notes 63–68 cents flat on the plan's own audio (6 with m4's notes), note F1@50 0.59 against the estimate's 1.0; the reader misreads 92 on the truth's own stem | a reader that reads the truth's stem right, and the estimate checked against it |
| `refdissect live-build` has not run end to end: Live's Remote Script serves 8 clients (item 3) | no build by the package's own code | a suite plan and PIRAMIT's built by `live-build` |
| Locators off Live's Arrangement grid are deleted, not placed | 3 of meter34-1's 5 in Live; 11 of the 169 locators of the m5 plans lie off Live's bar lines (11 of 170 on m5-ft; full, tempo72; not built) | every locator where planned |
| A note's own cents: one Detune per Simpler (item 2) | full's plan note F1@50 0.851 (0.890), htdemucs_ft 0.890 (0.910); producer's median error 8.2 cents (6.8), htdemucs_ft 21.1 (15.2) | no plan 808 number of any scenario worse than m3's by more than 0.01 (1 cent) |
| The 20 ms edge: one of swing58-1's 808 notes, 19.981 ms after the truth's at m3, sits 20.001 ms after it at m4, outside 20 ms | plan strict F1@20 0.8756 (0.8757), 808 onset F1@20 0.9344 (0.9348); htdemucs_ft unchanged | – |
| The one-shot's head, not judged by ear: the pitch step at the first crossing followed, and the quiet (20 %) or cut head that puts its attack where the template's is | not judged | listen |
| melodic-1's kick lands late (B9): its template's window starts on the body; a view later on the sound than its main still takes its hits, late (pinned as an expected failure) | kick F1 at 15 ms 0.679 (0.748), median error 14.7 ms (11.8), scored on the truth kick alone | on the attack, as htdemucs_ft times the same kicks (0.5 ms) |
| Hats held by a snare's top end at 15–24 ms (melodic-1, leadin-1, tempo160-1, producer); bass_low's hats almost missing | analysis hat timing p90 3.337 ms (3.329; 2.55 in the discovery lane); bass_low hat F1@20 0.006 | ≤ 3.3 ms over the suite, ≤ 10 ms in every scenario; 0.90 |
| The plan's snare p90: producer-1's snare, now played by a pad, carries the analysis' 41.8 ms p90 | plan snare p90 3.45 ms (3.04) over the 20 scenarios m2g-vel times a plan snare in; htdemucs_ft 3.60 (2.91); over m3's 22, 3.18 and 3.31 ms | no worse than m2g-vel |
| Signed timing medians, means over scenarios: the plan's hat mean is set by bass_low-2's hats, 29.1 ms early in the analysis | plan hat median −1.24 ms; 20 of 22 scenarios within 0.5 ms (19 on htdemucs_ft) | within ±0.5 ms |
| Which of two templates of one sound is the purer: merges keep the one with more clean hits only within one band (layers-2's low-band snare would find 13 more hats) | layers-2 hat 0.945 (0.993 with the low-band template) | – |
| A sound that always plays with another has no template: the layers scenario's clap | clap F1@20 0 on layers, 0.500 over the suite | – |
| Pitched copies of residual templates are off | re-planned on 20 beats: rolls-2 +0.054, producer-2 +0.051, rolls-1 −0.043 (m2g) | – |
| Residual templates on melodic-loop transients: recurrence keeps them off the pads only because the suite's loop plays at its own tempo | 1.96 sounds a beat that play no truth sound at analysis level, 0.05 with a pad; with recurrence waived 0.37 and 16.3 false hits a beat (an estimate) | junk evidence that does not depend on the loop's tempo |
| False hits rose on single beats (not investigated) | v1-2 6 → 9, drum_pitch-2 4 → 5; htdemucs_ft producer-2 4 → 9, rare-1 2 → 3 | – |
| htdemucs_ft's extra sounds: tempo160-2's snare-band template, once scored as the kick, now counts as an extra | 1.977 a beat at analysis level (1.955) | – |
| Pads that play a file of another role (not judged by ear): dynamics-2's kick (a snare file), melodic-1's and tempo160-1's snares (hat files); htdemucs_ft rolls-1's hat template matched as a kick | rolls-1 plan kick F1@20 0.466 (htdemucs_ft) | – |
| Library files over full scale play unscaled | 1 % of the entries peak over +6.4 dBFS, the loudest +19.4 | – |
| An index built before the −60 dBFS floor reads each listed entry's file once | 4.7 % of match over 6 beats | `index-library` again |
| Chords follow the sections, which follow the drum hits | full-2 majmin 0.999 → 0.961 in an intermediate run, with the same notes | ≥ 0.85 |
| bass_low-2's htdemucs_ft template is read from its tail, 57 cents flat; `settle()` refuses its reading and only lowers a template | written note right 0.130 on that beat; bass_low 0.559 on htdemucs_ft, its plan note F1@50 0.513 (bass_low-1 0.981) | ≥ 0.95 |
| 808 candidates inside a silence are dropped after the segmentation, not before it; an attack under 9 dB with no kick stays in the note before (once on PIRAMIT, 6.5 dB) | no truth on PIRAMIT; one suite note moved onto a rise 19 ms after the truth's onset (bass_low-2, htdemucs_ft) | – |
| The 808's slides (item 4 before). Legato notes are off by default, so no plan plays a slide; the plan and the build carry Glide Mode, Glide Time and one voice for a line with three legato notes or more, measured in Live on a probe only. The rule this replaced was right 6 times in 79; the opt-in rule (`slides=True`) made 18 legato notes, 6 of them right, on the beats its thresholds were chosen on | legato recall 0 (the truth: 24 on 5 beats) | slides that beat merging on note F1 |
| The 3 s one-shot leaves long legato chains' tails silent | not scored | – |
| Library layers are not played; kick sources are off by default | 75 of 95 templates with the right file (identify lane's 8 scenarios, m2g) | – |
| tempo72's bar line is wrong; meter68 is read in 4/4 at x2/3; the seed-1 swing beats read straight 8ths; Live's bar 1 sits at 0 s (the plan carries its meter since m5) | plan meter right 0.955 (0.909 at m4), plan downbeat F 0.924 (0.892); tempo72 downbeat F 0 | – |
| htdemucs_ft has better drums but worse 808 notes on short notes (bass_low) and on producer | plan drums 0.868 → 0.898; 808 note F1@50 at analysis 0.936 → 0.924 (in the plan 0.934 on both at m4) | the default changes when a model is better on both drums and 808 notes |
| PIRAMIT, not heard: its 3 new pads (a perc, the hat's file in the snare band, a snare-like sound); 6 of the perc's 8 notes lie within 15 ms of a main-hat note | no truth | listen |
| PIRAMIT's levels, read on its stems: the perc, the loudest pad, mostly on hits it shares with the hat; the kick under the plan's 808 | the perc 2.4 dB lower on its 2 clear hits; the kick at −5.6 or −10.1 dB with that 808 1 dB quieter or louder (−8.1 dB as planned); no truth | listen |
| PIRAMIT's one-shot: its pitch now follows the template's, its waveform matches the fit's reference less | NCC 0.880 (0.921 with the m3 code's one-shot) | listen |

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
