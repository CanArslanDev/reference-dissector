# 12 – Replica mode

Replica mode takes a trap or hip-hop reference apart and rebuilds it 1:1 in Ableton Live through [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp):
- every drum hit, layered hits included;
- the 808's notes, with their cents and glides;
- the melodic layers and the chords;
- the sounds, and their levels.

Then it measures the rebuild against the reference.

**Where it stands.** Everything below was measured on the 48 synthetic beats of the calibration-v2 suite ([03](03-calibration.md)), on htdemucs stems unless a line says otherwise.
- The analysis (what the pipeline heard) is measured.
- The plan (what Live will play, rendered offline by `render.py`) is measured.
- The build in Live was checked twice on 2026-10-29 ([07](07-building-in-live.md#live-behaviour-measured-here)): eight m4 plans and PIRAMIT's, then four m5 plans and PIRAMIT's with the fixes the first check led to. On the second, Live played every drum hit within 0.04 ms and every 808 note within 0.11 ms of `render.py`, and its levels differed from `render.py`'s only by Simpler's top-end loss (hats up to 0.97 dB). `refdissect live-build` itself has not run end to end: Live served 8 clients both times, and the checks sent its calls through their own session.

| means over 22 scenarios, analysis / plan | before (base-htdemucs) | m2g | m3 | m4 | now (m5) |
|---|---|---|---|---|---|
| drum hits and 808 onsets, strict F1@20 | 0.368 / 0.282 | 0.886 / 0.667 | 0.888 / 0.876 | 0.888 / 0.876 | 0.888 / 0.876 |
| drums, strict F1@20 | 0.397 / 0.283 | 0.880 / 0.612 | 0.883 / 0.868 | 0.883 / 0.868 | 0.883 / 0.868 |
| open hat, strict F1@20 | 0.326 / 0 | 0.979 / 0.043 | 0.980 / 0.981 | 0.980 / 0.981 | 0.980 / 0.981 |
| 808 notes, F1 (50 ms, 50 cents) | 0.278 / 0.229 | 0.936 / 0.926 | 0.936 / 0.921 | 0.936 / 0.934 | 0.936 / 0.934 |
| 808 pitch, median abs error | 55.3 / 69.8 cents | 6.0 / 7.7 cents | 6.0 / 8.3 cents | 6.0 / 4.5 cents | 6.0 / 4.4 cents |
| 808 written note right | – | 0.947 | 0.970 | 0.970 | 0.970 |
| meter right | 0.91 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.95 |
| plan levels: median distance from a perfect kit | 7.18 dB | 3.67 dB | 4.09 dB | 1.25 dB | 1.20 dB |
| plan levels: median balance | 4.15 dB | 2.25 dB | 3.03 dB | 0.55 dB | 0.51 dB |

m2g's plan levels are the same under Live 12.4.6's velocity law, which the plan follows since m2g-vel (3.67 dB, balance 2.24 dB). m3's were worse: every pad played its sample at 0 dB, and the sounds the pad gate adds play louder than the songs hold them. m4's pads each have a level of its own, read where the stems hold its sound, and its 808 one-shot follows its template's own periods. m5 plays each pad on one voice and the 808's release as Live does, starts each note Live starts late 1 ms early, and carries the grid's meter, which the build sets in Live. The second live check built four m5 plans and found Live playing them as `render.py` does, but for Simpler's top-end loss.

`base-htdemucs` is the code before replica mode (471fd70). `m2g` is the merged code (9ccc759), and `m2g-vel` its plans under Live's velocity law, the baseline of the fixes of 2026-10-29. `m3` is replica mode with five of those fixes (6d96bc5), and `m4` with all seven (e7ea748, [below](#what-each-stage-measured)): m3 plus a level per drum pad and the 808 one-shot on its template's periods. `m5` is m4 with the fixes of the first live check merged (live/build, a4f2358): one voice per pad, the 808's release and restart, the plan's meter, and pads loaded by their content. Their scores are `results/calibration/suite-base-htdemucs.json`, `suite-m2g.json`, `suite-m2g-vel.json`, `suite-m3.json`, `suite-m4.json` and `suite-m5.json`; `suite-m3-ft.json`, `suite-m4-ft.json` and `suite-m5-ft.json` hold m3, m4 and m5 on htdemucs_ft stems.

## Running it

### A song

```bash
W=~/refdissect     # a work folder outside the clone: it holds audio and transcriptions (docs/09)
uv sync --extra live

# the library, once; index-library also writes the speckle index (library_speckle.npz)
uv run refdissect inventory "<Core Library>/Samples" "<your User Library>" --out "$W/work/inventory.json"
uv run refdissect index-library "$W/work/inventory.json" --out "$W/work/library.npz"
uv run refdissect index-library "$W/work/inventory.json" --808 --out "$W/work/library_808.npz"
uv run refdissect index-speckle "$W/work/library.npz"    # only for a library indexed before the speckle index

# the song
uv run refdissect separate song.mp3 --out "$W/song/stems"
uv run refdissect analyze "$W/song/stems" --work "$W/song/refdissect"
uv run refdissect melodic "$W/song/refdissect"
uv run refdissect match "$W/song/refdissect" --library "$W/work/library.npz"
uv run refdissect fit-808 "$W/song/refdissect" --library-808 "$W/work/library_808.npz"
uv run refdissect plan "$W/song/refdissect"
uv run --extra live refdissect live-build "$W/song/refdissect"
```

- `analyze` takes `--bpm-range lo hi` (default 100–180: the octave the tempo is counted in), `--bpm` (forces the tempo), `--meter N/D` (3/4, 6/8; default: the tracker's, 4/4 when unsure) and `--no-tracker` (the comb alone, as before replica mode).
- `analyze` and `fit-808` keep the 808 reader's result and the 808 fit in the work folder's `cache/` and read them back where nothing that decides them changed ([01](01-pipeline.md#808-notes-bass808transcribe)); `--no-cache` runs them again.
- `melodic` takes `--min-hz`, `--snap-ms`, `--no-vote` and `--drop-below`.
- `match` takes `--detune X` and `--kick-sources`. Kick sources are off by default: they need fit-808's one-shot, so `fit-808` must run first.
- Three models run in uv environments of their own and download on first use: Demucs (`separate`), beat_this (`analyze`, an 81 MB checkpoint) and Basic Pitch (`melodic`). Torch never loads into the main process.

### The suite

```bash
S="$W/work/calibration-v2"
uv run refdissect suite build "$W/work/inventory.json" --out "$S" --v1-truth "$V1_TRUTH"
uv run refdissect suite run "$S" --library "$W/work/library.npz" --library-808 "$W/work/library_808.npz" \
    --tag mine --jobs 2 --render
uv run refdissect suite run "$S" --library "$W/work/library.npz" --library-808 "$W/work/library_808.npz" \
    --tag mine-b --steps analyze,match,plan,score --reuse mine --jobs 2 --render
uv run refdissect suite report "$S" --tags mine,mine-b --out "$W/work/mine-b.md"
uv run python scripts/export_results.py --suite "$S" mine mine-b
```

**`suite build`** renders 21 scenarios at seeds 1 and 2 (`--scenarios`, `--seeds`) and converts the six v1 beats as `v1orig-1` to `v1orig-6`. `$V1_TRUTH` holds those beats rebuilt with truth stems: `beatN` folders with `truth.json`, `stems_truth/` and the mix files. No command in this repository makes them; without `--v1-truth` the suite has the 42 scenario beats. The `producer` scenario needs single-role producer MIDI patterns under `--midi-dir` (default: Ableton's User Library); without them it is left out. A beat already built is kept.

**`suite run`** runs every beat under a tag:
1. Demucs, once per beat and model, into `stems-<model>/` (seed 0, one shift). The stems are kept.
2. analyze, melodic, match, fit-808 and plan, into `runs/<tag>/`.
3. The scores of the analysis and of the plan.

Its options:
- `--model`: the Demucs model (default htdemucs).
- `--scenarios a,b`: a subset.
- `--jobs N`: beats in parallel processes. Each loads the library (about 0.5 GB) when match runs.
- `--render`: renders each plan offline for the level scores, then deletes the audio.
- `--resume`: skips the beats this tag has scored already; a beat that failed runs again.
- `--steps`: a subset of analyze, melodic, match, fit-808, plan and score. Without `--reuse`, a step whose input no step makes is refused before anything runs.
- `--reuse TAG`: copies the outputs of the steps not run from `runs/TAG/`, in place when TAG is the run's own tag. It refuses before running when a beat lacks those outputs, failed there, or analysed another model's stems.
  - A reused step that reads what a run step rewrites is marked `reused_stale`, and the warning and the report name it. melodic reads the 808 notes, match the drum templates, plan everything.
  - The 808 lane's runs reused m1's chords, built on the old 808 notes. Rebuilt on the new ones, majmin went 0.888 → 0.977 on the four melodic and full beats.
- `--cache`: keeps the 808 reader's results and the 808 fits in each beat's `cache/`, and reads them back where the stems (or the template), the settings and the package's code are the same. Off by default: every step runs, so a measurement's seconds are the code's. A run with it records what it read back (`cached` in `scores.json`), leaves those steps' seconds, and then the total, out of its means, and its report's tag line names them. When every tag shared each beat's cache, a read-back timed the cache, not the code: fit-808 took 1.6 s read back and 68–70 s run on v1-1 and glides-1.
- analyze's and match's options pass on unchanged.

**`suite report`** writes markdown tables of each headline number, per scenario and per drum role, for every tag, with the change from the first tag to the last.

**`scripts/export_results.py --suite`** publishes the means per scenario only, as `results/calibration/suite-<tag>.json`: no beat, file, path, note or event time.

**Time and memory** (m3a: the merged code at e2a5ab4, before the pad-files round, which changes only match; every step run, no cache, 48 beats, 2 jobs, `--render`, on an Apple M4 with Live open, stems already separated; m2g in brackets, on a quiet machine):
- A beat took 112.6 s (141.9 s): analyze 34.4 s (38.7), match 28.0 s (49.1), fit-808 46.2 s (50.9), melodic 3.3 s (2.7); plan, render and scores under 1 s together. Means over scenarios; [03](03-calibration.md#time-and-memory) has the rest.
- The 48 beats took 2,704 s, 45 minutes (3,378 s), and 5,234 s of CPU.
- A job peaked at 1.73 GB, the process tree at 2.9 GB (1.71 and 3.3 GB).
- A tag's run folders take about 225 MB.
- Separation, once per model: 2.1 s a beat with htdemucs, 8.5 s with htdemucs_ft (MPS).
- With `--cache`, read back on 6 beats: analyze 8.1 s and fit-808 1.0 s a beat, against 35.6 and 44.7 s run fresh, with the same outputs.
- m4 and m4-ft ran fit-808, plan and score only, at 4 jobs, no cache: 57.4 s a beat over scenarios (fit-808 56.2 s), 695 and 698 s wall for the 48 beats.
- m5 and m5-ft ran plan and score only, at 4 jobs, no cache: 5.8 and 4.4 s a beat over scenarios, the render 4.0 and 2.6 s of it (0.5 s at m4; not investigated), 74 and 53 s wall for the 48 beats.
- `uv run pytest -q`: 356 passed, 3 skipped and 2 expected failures (the pinned B9 cases, [Known limits](#known-limits)) in 328 s in one process (8fb09dc). No test needs the library, Demucs, Live or a model download.

## The stages and their files

```
song ──separate──▶ stems/
stems ──analyze──▶ analysis.json, bass808.json, templates/
      ──melodic──▶ melodic.json, basicpitch/
      ──match────▶ matches.json
      ──fit-808──▶ synth808.json, 808 (synth).wav
      ──plan─────▶ plan.json, midi/ ──live-build──▶ the Live set

events.json (the truth) ──▶ evaluate ◀── estimates from analysis.json and from plan.json
                                │
                                └──▶ scores.json, scores-<tag>.json
```

| step | writes | what replica mode changed |
|---|---|---|
| `separate` | `stems/*.wav`, `separation.json` | Seeded runs: seed 0 and one shift give byte-identical stems on one device. `separation.json` names the model, seed, shifts, device, package versions and the input's hash. |
| `analyze`: grid | `analysis.json` `grid` | beat_this's beats and downbeats decide the octave, the first bar line and the meter. The comb's phase moves onto the attacks. Swing per 8th and 16th, and a tempo map where the beats drift more than 10 ms. |
| `analyze`: drums | `analysis.json` `templates`, `events`, `layers`; `templates/` | Templates where sounds start, one per sound, then a residual pass. Every moment is fit with every template: layers, flams, rolls, pitched copies. Each view is fit right after its main. Each hit counted once. |
| `analyze`: 808 | `bass808.json`; `analysis.json` `notes_808`, `tuning` `808`; `cache/` | Notes read as transpositions of the song's own 808, to the cent. A note ends where the 808 falls silent, and a 9 dB rise attacks one. The 808's tuning is its template's settled pitch, read on the notes that reach past the template where its own notes never settle. |
| `melodic` | `melodic.json`, `basicpitch/` | New: Basic Pitch's notes per tonal stem, and chords with the 808 as their bass. |
| `match` | `matches.json` | Snare and hat bands at any pitch, to a quarter cent (the speckle index). Transpose and Detune per entry. Library layers are searched. A file that peaks under −60 dBFS is never a match, and a band whose five best sounds tie names none of them. |
| `fit-808` | `synth808.json`, `808 (synth).wav`, `templates/808.npy`; `cache/` | Fitted on the template's notes. The one-shot settles at the template's settled pitch and follows the template's own periods until its drop has settled, laid out as the template is (m4). |
| `plan` | `plan.json`, `midi/` | Pads on evidence: hits, a clean template, a close match or a place in the backbone; a residual sound must recur bar to bar. The backbone first, and each strike played once. Pads play their match's Transpose and Detune, each at a Volume of its own, read where the stems hold its sound, the loudest at 0 dB (m4). A drum note starts as much before `t_attack` as its pad's sample takes to attack, and 1 ms earlier where Live starts it late, as it cuts its pad's sounding hit (m5). The 808 plays its written pitches with Transpose, Detune and, with slides on, Glide; a note that restarts the voice of its pitch starts 1 ms earlier too (m5). A muted MIDI track per tonal stem; the grid's meter, in whole bars of it (m5); locators on bar lines. |
| `live-build` | the Live set | Detune and Volume per pad; the 808's one-shot, Detune, Glide Mode, Glide Time and voices. Snap off, the 808's filter off and its release sent; each pad's file copied under a name of its content and read back; Live's time signature from the plan, and a locator Live moves deleted (m5). Built and recorded with its calls on four m5 plans and PIRAMIT's ([07](07-building-in-live.md)); `live-build` itself not run end to end. |
| `suite run` | `scores.json`, `scores-<tag>.json` | Strict scores of the analysis and of the plan against the truth's `events.json`. Nothing kept or read back unless `--cache`. |

Each step is described in [01](01-pipeline.md), the 808 voice in [05](05-808-synthesis.md), the build in [07](07-building-in-live.md).

### analysis.json

- `source`: the stems and their duration; with the 808 reader's result read back, `cached` names the file it came from.
- `grid`: `bpm`, `beats_per_bar`, `beat_unit`, `phase` (the first bar line, s), the octave taken from the tracker and why, `phase_correction_ms`, `swing`, `tempo_map`, `downbeat` (`ambiguous` when unsure), `warnings`, `offsets_by_beat`.
- `tuning`: per stem, and their median. `808` is the 808's own tuning: its template's settled pitch over the nearest MIDI note, in semitones. `808_agreement` (0–1) says how well the notes' settled pitches agree on one offset from equal temperament.
- `templates`, per sound:
  - `role_guess` and `band`;
  - `attack_s`: where the sound starts in its window;
  - `same_sound_as`: a view of another sound, found in another band;
  - `found`: `grid` (a place in the beat) or `residual` (the residual pass);
  - `detected_hits`, and `clean`: the share of its own hits at NCC 0.8 or more.
- `events`, per sound, each hit:
  - `t`, 5 ms into the matched window;
  - `t_attack`, where it attacks: the template's `attack_s` after the window start, sooner for a pitched copy by as much as Simpler plays it faster;
  - `ncc`, and `gain`: its coefficient in the joint fit, the hit's level;
  - `source`: `matched`, or `layer` for a hit found under another sound, with `under`, the template it was under;
  - `semitones` for a pitched copy, `flam_of` for a grace note.
- `layers`: the joint fit's record. It lists the templates left out as composites, the merged ones with their offset, the echoes, the tones, what each template holds, the pair offsets, the pitched copies, and each template's usual gain and null. There is no separate `layers.json`.
- `notes_808`: `bass808.json`'s notes as the plan reads them:
  - `t`, `t_attack`, `end`;
  - `pitch`: the MIDI note written;
  - `midi_float`: the pitch heard;
  - `midi_settled` and `cents`;
  - `slide`, `peak`, `glide`, `flags`.
- `bars` and `sections`.

### The other files

- **bass808.json**: the 808's template and its notes.
  - The template: `hz`, `midi`, `root_midi`, `tuning_cents`; `method`, how its settled pitch was read (`qifft_<from>_<to>ms`, the QIFFT of the sustain, `_falling` where the drop still falls there; `drop_fit`, the asymptote of its fitted drop; `qifft_tail`, the QIFFT of its last 300 ms; `notes_<from>_<to>ms`, read on the notes that reach past it); the onsets it was learnt from; its pitch curve.
- **melodic.json**: per tonal stem, the notes (pitch, cents, beats, velocity) and the stem's tuning, and the chord segments. `basicpitch/` keeps the model's notes and posteriorgrams.
- **matches.json**, per template:
  - `role`, `matched_in_band`, `same_as`;
  - `search`: `speckle` or `semitones`;
  - `top`: the five best entries, each with `semitones` (fractional), `transpose` (whole semitones), `cents` (`detune_cents` is the same number under Simpler's name), `ncc` and `lag_ms`;
  - `margin`, `per_hit`, `null_test_db`, `verdict`;
  - `layers` and `layer_search`: a second sound and why it was or was not accepted;
  - `source` and `kick_source`, with `--kick-sources`;
  - `tied_band`, for a template matched in its own band: the band passed over, whose five best sounds tie, and the NCC of its first and fifth.
- **synth808.json**: the voice's parameters, and the pitch the one-shot was rendered at.
  - `pitch_hz` is that pitch. `pitch_from` names it: the measuring method, `fit` or `agree`.
  - `measured_hz` is the template's pitch, and `fit_vs_measured_cents` how far the fit's own pitch lies from it.
  - `fit`: the loss and the search's seconds; `cached` names the file a fit was read back from.
  - `pitch_track`: how the one-shot follows the template's periods (`periods`, `followed`, `until_s`, `settled`, `pooled`, `attack_scale`, `start_ms`, `quiet_ms`, `attack_ms`, …), null where it is the fitted voice.
  - `compare`: `synth`, the fitted voice against the reference; `tracked`, the one-shot laid out as the template.
- **cache/** (by default for the `analyze` and `fit-808` commands, in the work folder; in each suite beat only with `--cache`): the 808 reader's results with their log lines, and the 808 fits, each under a hash of what decides it (`cache.py`: the inputs' samples, the settings, numpy's, scipy's and soxr's versions, every `.py` file of the package as the process loaded it).
- **plan.json**:
  - `tempo`, `beats_per_bar` and `beat_unit` (the grid's meter), `length_beats` (whole bars of it), `stems`, `locators`, `levels`;
  - `tracks.drums`: `pads` (note → template, file, `transpose`, `detune_cents`, `volume_db`, `vol_vel`, NCC, verdict, role) and `notes`;
  - `tracks.808`: `notes`, `sample` (the one-shot's name in the User Library, with its digest), `sample_path`, `transpose`, `detune_cents`, `vol_vel`, `tuning_cents`, `heard_cents`, `slides` (every glide), and with Glide on `voices`, `glide_mode` and `glide_time_ms`;
  - `tracks.melodic`: a muted track per tonal stem.
- **events.json** (schema `refdissect.events/1`) holds a truth or an estimate in one form, so one scorer reads both.
  - A scenario writes the truth (`kind: "truth"`):
    - the grid (`time`);
    - every drum hit (`sound`, `t`, `t_attack`, `gain`, `velocity`, `layer_of`, `flags`);
    - every 808 note (`t`, `t_attack`, `end`; `midi`, the pitch it sounds at, cents included; `midi_settled`, the pitch it settles to; `glide`);
    - the melodic notes (`layer`, `t`, `end`, `midi`, `cents`, `velocity`) and the chords (Harte labels);
    - every sound (`role`, `path`, `sha256`, `semitones`, `cents`, `attack_s`).
  - `evaluate.estimate_from_analysis` and `evaluate.estimate_from_plan` build estimates (`kind: "estimate"`) in the same form. `evaluate.validate` checks a file, and `evaluate.save_events` writes one.
- **scores.json**, per beat and tag: the analysis's and the plan's scores, each step's status (`run`, `reused`, `reused_stale`, `skipped`), `cache` and `cached` (the results read back), the seconds per step and the problems. `scores-<tag>.json` holds the means per scenario, and counts the beats that read each result back.

All of these but the scores hold note times or library file names, and all of them stay in the work folder ([09](09-data-and-copyright.md)). The scores' means per scenario are what `scripts/export_results.py --suite` publishes.

## How it is scored

[03](03-calibration.md) has the details. In short:
- **Strict.** Truth and estimate both time a hit where its sound attacks (`audio.attack_time` in its role's band), and no offset is removed. A hit counts within 20 ms.
- **Two levels.** The analysis is scored from `analysis.json` and `matches.json`. The plan is scored from `plan.json` as `render.py` plays it: each pad's sample at its Transpose and Detune, one voice per pad, and the 808's one-shot per note, released as Live releases it (m5).
- **Drums.** Each estimated sound plays the truth sound it scores best on. A truth sound's F1 pools the hits of every estimated sound that plays it, so a hit played twice is a false hit. A pitched copy or a layer is a sound of its own.
- **Means.** Over a scenario's beats, then over the 22 scenarios.

## What each stage measured

Means over the 22 scenarios of calibration v2. Columns are tags: `base` = base-htdemucs (471fd70), `m1` = after the rhythm and melodic steps (e1bd9ce), `m2g` = after the four lanes and the merge's fixes (9ccc759), `m3` = after five fixes of 2026-10-29 (6d96bc5), `m4` = after all seven, m3 plus a level per pad and the 808 one-shot on its template's periods (e7ea748), `m5` = m4 with the fixes of the first live check (a4f2358), `base-ft`, `m2g-ft`, `m3-ft`, `m4-ft` and `m5-ft` the same code on htdemucs_ft stems. A cell is "analysis / plan" where both exist, and "–" where that tag's scorer had no such score or the step did not run.

| | base | m1 | m2g | m3 | m4 | m5 | base-ft | m2g-ft | m3-ft | m4-ft | m5-ft |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **End to end** | | | | | | | | | | | |
| drum hits and 808 onsets, F1@20 | 0.368 / 0.282 | 0.396 / 0.311 | 0.886 / 0.667 | 0.888 / 0.876 | 0.888 / 0.876 | 0.888 / 0.876 | 0.417 / 0.324 | 0.898 / 0.700 | 0.900 / 0.902 | 0.900 / 0.902 | 0.900 / 0.902 |
| **Grid** (beat_this, m1; the plan's meter, m5) | | | | | | | | | | | |
| beats, F (70 ms) | 0.715 / 0.920 | 0.962 / 0.962 | 0.962 / 0.962 | 0.962 / 0.962 | 0.962 / 0.962 | 0.962 / 0.962 | 0.686 / 0.936 | 0.962 / 0.962 | 0.962 / 0.962 | 0.962 / 0.962 | 0.962 / 0.962 |
| downbeats, F | 0.637 / 0.823 | 0.939 / 0.892 | 0.939 / 0.892 | 0.939 / 0.892 | 0.939 / 0.892 | 0.939 / 0.924 | 0.614 / 0.839 | 0.939 / 0.892 | 0.939 / 0.892 | 0.939 / 0.892 | 0.939 / 0.924 |
| meter right | 0.91 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.95 | 0.91 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.95 / 0.95 |
| tempo in the right octave | 0.89 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 | 0.91 |
| **Drums** (discovery and the joint fit, m2g; the kick view and the pad gate, m3) | | | | | | | | | | | |
| all drum sounds, F1@20 | 0.397 / 0.283 | 0.431 / 0.319 | 0.880 / 0.612 | 0.883 / 0.868 | 0.883 / 0.868 | 0.883 / 0.868 | 0.470 / 0.354 | 0.893 / 0.648 | 0.895 / 0.898 | 0.895 / 0.898 | 0.895 / 0.898 |
| kick | 0.247 / 0.182 | 0.305 / 0.201 | 0.872 / 0.845 | 0.882 / 0.871 | 0.882 / 0.871 | 0.882 / 0.871 | 0.581 / 0.637 | 0.927 / 0.919 | 0.935 / 0.925 | 0.935 / 0.925 | 0.935 / 0.925 |
| snare | 0.695 / 0.493 | 0.801 / 0.600 | 0.961 / 0.813 | 0.961 / 0.935 | 0.961 / 0.935 | 0.961 / 0.935 | 0.556 / 0.319 | 0.891 / 0.816 | 0.891 / 0.919 | 0.891 / 0.919 | 0.891 / 0.919 |
| hat | 0.508 / 0.549 | 0.504 / 0.569 | 0.859 / 0.845 | 0.860 / 0.861 | 0.860 / 0.861 | 0.860 / 0.861 | 0.626 / 0.589 | 0.913 / 0.916 | 0.913 / 0.919 | 0.913 / 0.919 | 0.913 / 0.919 |
| open hat | 0.326 / 0 | 0.303 / 0 | 0.979 / 0.043 | 0.980 / 0.981 | 0.980 / 0.981 | 0.980 / 0.981 | 0.324 / 0.006 | 0.993 / 0.022 | 0.993 / 0.993 | 0.993 / 0.993 | 0.993 / 0.993 |
| kick timing, median (p90), analysis | +16.6 (20.9) ms | +17.7 (22.0) ms | 0.0 (6.1) ms | +0.1 (5.3) ms | +0.1 (5.3) ms | +0.1 (5.3) ms | +2.0 (7.6) ms | −0.5 (3.4) ms | −0.5 (3.3) ms | −0.5 (3.3) ms | −0.5 (3.3) ms |
| snare timing | −6.4 (8.3) ms | −6.0 (7.7) ms | +0.8 (3.2) ms | +0.8 (3.2) ms | +0.8 (3.2) ms | +0.8 (3.2) ms | −6.1 (8.0) ms | +0.3 (3.8) ms | +0.3 (3.8) ms | +0.3 (3.8) ms | +0.3 (3.8) ms |
| hat timing | +1.1 (3.4) ms | −0.5 (2.9) ms | −0.4 (3.3) ms | −0.4 (3.3) ms | −0.4 (3.3) ms | −0.4 (3.3) ms | +0.2 (2.2) ms | −0.3 (1.7) ms | −0.3 (1.7) ms | −0.3 (1.7) ms | −0.3 (1.7) ms |
| kick timing, median (p90), plan | +9.6 (12.7) ms | +9.8 (12.6) ms | +2.1 (7.9) ms | −0.3 (5.6) ms | −0.3 (5.6) ms | −0.3 (5.6) ms | +3.2 (6.8) ms | +1.9 (3.4) ms | −0.6 (2.2) ms | −0.6 (2.2) ms | −0.6 (2.2) ms |
| sounds that play no truth sound, per beat | 0.16 / 0.05 | 0.16 / 0.05 | 1.96 / 0.71 | 1.96 / 0.05 | 1.96 / 0.05 | 1.96 / 0.05 | 0.18 / 0.02 | 1.95 / 0.32 | 1.98 / 0.02 | 1.98 / 0.02 | 1.98 / 0.02 |
| **808** (bass808, m2g; silences and the settled pitch, m3; the one-shot on its template's periods, m4) | | | | | | | | | | | |
| onsets, F1@20 | 0.270 / 0.303 | 0.270 / 0.303 | 0.935 / 0.935 | 0.935 / 0.935 | 0.935 / 0.934 | 0.935 / 0.934 | 0.207 / 0.214 | 0.944 / 0.944 | 0.944 / 0.944 | 0.944 / 0.944 | 0.944 / 0.944 |
| notes, F1 (50 ms, 50 cents) | 0.278 / 0.229 | 0.278 / 0.229 | 0.936 / 0.926 | 0.936 / 0.921 | 0.936 / 0.934 | 0.936 / 0.934 | 0.173 / 0.146 | 0.924 / 0.910 | 0.924 / 0.910 | 0.924 / 0.934 | 0.924 / 0.934 |
| notes, F1 (20 ms, 50 cents) | 0.160 / 0.159 | 0.160 / 0.158 | 0.911 / 0.901 | 0.911 / 0.896 | 0.911 / 0.909 | 0.911 / 0.909 | 0.102 / 0.092 | 0.913 / 0.896 | 0.913 / 0.896 | 0.913 / 0.920 | 0.913 / 0.920 |
| pitch, median abs error | 55.3 / 69.8 cents | 55.3 / 69.8 cents | 6.0 / 7.7 cents | 6.0 / 8.3 cents | 6.0 / 4.5 cents | 6.0 / 4.4 cents | 72.9 / 78.5 cents | 12.1 / 13.9 cents | 12.1 / 13.6 cents | 12.1 / 9.6 cents | 12.1 / 9.6 cents |
| written note right | – | – | 0.947 | 0.970 | 0.970 | 0.970 | – | 0.937 | 0.959 | 0.959 | 0.959 |
| onset timing, median (p90), analysis | −4.1 (37.1) ms | −4.1 (37.1) ms | +1.4 (8.0) ms | +1.4 (8.0) ms | +1.4 (8.0) ms | +1.4 (8.0) ms | −3.3 (37.4) ms | +1.4 (7.0) ms | +1.4 (7.0) ms | +1.4 (7.0) ms | +1.4 (7.0) ms |
| **Identification** (match v2, Detune per pad; plan only) | | | | | | | | | | | |
| truth sounds a pad plays with their own file | – | – | 0.574 | 0.839 | 0.839 | 0.839 | – | 0.641 | 0.898 | 0.898 | 0.898 |
| largest pitch error of such a pad | – | – | 0 cents | 0 cents | 0 cents | 0 cents | – | 0 cents | 0 cents | 0 cents | 0 cents |
| **Melodic** (Basic Pitch and chords, m1; 4 beats) | | | | | | | | | | | |
| notes, F1 (50 ms, 50 cents) | – | 0.630 | 0.630 | 0.630 | 0.630 | 0.630 | – | 0.635 | 0.635 | 0.635 | 0.635 |
| chords, majmin | – | 0.888 | 0.977 | 0.977 | 0.977 | 0.977 | – | 0.979 | 0.979 | 0.979 | 0.979 |
| **Levels** (plan, rendered; a level per pad, m4; one voice per pad and the 808's release, m5) | | | | | | | | | | | |
| median abs distance from a perfect kit | 7.18 dB | 7.19 dB | 3.67 dB | 4.09 dB | 1.25 dB | 1.20 dB | 3.81 dB | 3.21 dB | 3.70 dB | 1.23 dB | 1.15 dB |
| median abs balance | 4.15 dB | 3.82 dB | 2.25 dB | 3.03 dB | 0.55 dB | 0.51 dB | 2.67 dB | 1.98 dB | 2.84 dB | 0.50 dB | 0.48 dB |
| the 808 against the kit | +0.09 dB | +0.09 dB | −1.95 dB | −1.98 dB | −1.87 dB | −1.87 dB | +0.96 dB | −2.23 dB | −2.24 dB | −2.01 dB | −2.01 dB |
| **Seconds per beat** (jobs) | 190 (3) | 90 (2) | 142 (2) | 113 (2) | 57 (4) | 6 (4) | 203 (3) | 141 (2) | 128 (4) | 57 (4) | 4 (4) |

m3's and m3-ft's seconds are m3a's and m3a-ft's, which ran every step at e2a5ab4, the same code but for the pad-files round (match only); m3a-ft ran at 4 jobs. m4's and m4-ft's cover fit-808, plan and score only, the steps they ran, so they compare with no other column; m5's and m5-ft's cover plan and score only. The exports give m4's plan note F1@20 as 0.9095 and m4-ft's note F1@50 as 0.9345; from the beats' counts their means over scenarios are 0.9094 and 0.9344. m2g's levels are under linear velocities; m2g-vel, the same plans under Live's velocity law, scored 3.67 and 2.24 dB and the 808 −1.97 dB (htdemucs_ft 3.21, 1.98 and −2.24).

- **The grid** changed in m1 and is unchanged since. The four lanes did not touch it.
- **The 808** changed only in its lanes. Its numbers are the same in m2 and m2g; m3 reads the settled pitch on the notes where a template's drop never settled, and ends notes at a silence ([01](01-pipeline.md#808-notes-bass808transcribe)).
- **Chords** rose from m1 to m2g because melodic now reads the new 808 notes as the chords' bass. m3 changes no chord or melodic number.
- **Kick timing on htdemucs_ft** was already near 0 before (base-ft +2.0 ms). On htdemucs drums a kick's body comes 26 ms after its click, as the rest of the kick goes to the bass stem, and the old templates timed kicks on their body.
- **Levels under Live's velocity law.** m2g-vel and m2g-ft-vel (`suite-m2g-vel.json`, `suite-m2g-ft-vel.json`) ran plan and score again on m2g's and m2g-ft's other outputs, with velocities from Live 12.4.6's law at "Vol < Vel" 35 %: 3.67 and 3.21 dB from a perfect kit, balance 2.24 and 1.98 dB, the 808 −1.97 and −2.24 dB; every other number is unchanged. With v/127 velocities, Live would have played m2g's plans 9.30 dB from a perfect kit and m2g-ft's 9.59 dB (computed from the law: pads at 45 %, the 808 at 35 %).
- **m2 and m2-ft** (`suite-m2.json`, `suite-m2-ft.json`) measured the first merge (ba7b69d), before five merge fixes: drums 0.847 / 0.600, open hat 0.901, and pads up to 300 cents off pitch (producer-2; drum_pitch-2: 30 cents). For that last number `suite-m2.json` gives 7.5 cents, a mean over scenarios of each beat's largest error.
- **m3 and m3-ft** (`suite-m3.json`, `suite-m3-ft.json`) measured the first five fixes of 2026-10-29, below. Against m2g-vel (m2g-ft-vel), no drum role of any scenario fell by more than 0.005, at analysis or plan level, and rhythm, melodic and chords are unchanged. Worse: the plan's levels and the plan's 808 on bass_low-1 (both resolved at m4); the plan's snare p90, 3.04 → 3.45 ms (2.91 → 3.60) over the 20 scenarios m2g-vel times a plan snare in (producer-1's snare, now played by a pad, carries the analysis' 41.8 ms p90); the analysis' hat p90, 3.329 → 3.337 ms; htdemucs_ft's extra sounds at analysis level, 1.955 → 1.977 a beat. False hits fell 23.0 → 9.6 a beat in the plan (10.2 → 3.7), and 42.4 → 41.7 at analysis level (35.23 → 35.15), but rose on single beats: v1-2 6 → 9, drum_pitch-2 4 → 5 (producer-2 4 → 9, rare-1 2 → 3).
- **m4 and m4-ft** (`suite-m4.json`, `suite-m4-ft.json`) measured the merge of the last two fixes, levels/pads (389eda2) and glide/808 (08c4fb1), at e7ea748: fit-808, plan and score over m3's other outputs, all 48 beats, `--render`.
  - Every analysis number, every drum note and pad but its Volume, and every melodic track equal m3's. Every 808 number equals glide/808's own tag (g8-f1), and every one-shot is the same, byte for byte.
  - Against m3 (m3-ft): plan levels 4.09 → 1.25 dB from a perfect kit (3.70 → 1.23), balance 3.03 → 0.55 dB (2.84 → 0.50); plan 808 note F1@50 0.921 → 0.934 (0.910 → 0.934), median error 8.3 → 4.5 cents (13.6 → 9.6). The two regressions of m3 are resolved.
  - Against levels/pads alone (lv-f3), the new one-shot moved 109 of 227 pad Volumes (102 of 229), by up to 1.7 dB (0.6), and the drum fader on 8 beats (9): a kick is read under the plan's own 808.
  - Worse than m3: plan strict F1@20 0.8757 → 0.8756, from swing58-1's 808 notes at 19.94–20.10 ms; full's 808 notes 0.890 → 0.851 (0.910 → 0.890); producer's median 808 error 6.8 → 8.2 cents (15.2 → 21.1); the kick against the snare on 2 of 42 beats (6 of 43); the 808's balance ([Known limits](#known-limits)).
- **m5 and m5-ft** (`suite-m5.json`, `suite-m5-ft.json`) measured the merge of live/build at a4f2358: plan and score over m4's other outputs, all 48 beats, `--render`. Its 96 plans equal the branch's own (lb-5, lb-5-ft).
  - Better: distance from a perfect kit 1.247 → 1.198 dB (1.231 → 1.152), balance 0.552 → 0.512 dB (0.497 → 0.476); the plan's meter right 0.909 → 0.955 and its downbeat F 0.892 → 0.924 on both models, meter34 planned in 3/4 over 54 beats (56 in 4/4 before); the 808's onset error median 1.360 → 1.355 ms (1.435 → 1.433), its median error 4.47 → 4.42 cents (9.65 → 9.63). Per role against the kit, kick −0.96 → −0.84 dB (−0.08 → −0.04), snare −0.17 → +0.04 (−0.80 → −0.75), hat −1.10 → −0.90 (−0.69 → −0.57), crash −0.25 → −0.05 (−1.26 → −1.16), and on htdemucs_ft the open hat −0.83 → −0.72. bass_low went 2.55 → 0.67 dB from the kit (bass_low-2's drum fader −23.9 → −20.5 dB).
  - The same: every analysis number, every F1 of the plan, and each drum role's timing median and p90 within 0.01 ms.
  - Worse: the distance from the kit in 10 (7) scenarios, dynamics by 0.90 dB (1.27 → 2.16) and the others by 0.12 dB at most (0.045); balance in 11 (6), by up to 0.23 dB (0.08), full 0.94 → 1.17 (1.33 → 1.41) and flams 0.22 → 0.40; per role clap +0.60 → +0.70 (+1.58 → +1.73), tom +0.57 → +0.67 (−0.74 → −0.84), rim +0.64 → +0.84, open hat −0.09 → +0.15 dB; htdemucs_ft's 808 onset p90 7.0389 → 7.0393 ms. On dynamics-2 (0.34 → 2.03 dB) the drum track at unity plays 1.3 dB quieter with its pads on one voice (79 hits cut), so the RMS match raises its fader −23.5 → −22.2 dB and its snare and hats sit 2.0–2.4 dB over the kit.
  - Of the 34 (32) beats whose distance from the kit moved, 17 (12) moved with one voice per pad and 28 (25) with the 808's release and restart, some with both, none with the meter.
  - In Live: four of these plans (v1orig-3, meter34-1, bass_low-1, dynamics-1) are those the second live check built and recorded ([In Live](#in-live)); the other 92 are not built.

**The first five fixes of 2026-10-29**, each on a branch of its own from 4f30623, measured there against m2g-vel (m2g-ft-vel), then merged; m3a ran every step of the merged code (e2a5ab4), and m3 the last round on top of it:
- **The pad gate** (fix/pad-gate, `live.py`; plan level, in memory on m2g-vel's analyses, render included: the harness gave the suite's own plan scores exactly on 4 code and tag pairs). Pads on evidence, the backbone first, each strike once, each note early by its sample's attack: plan drums 0.612 → 0.866 (0.648 → 0.898), open hat 0.043 → 0.980 (0.022 → 0.993), snare 0.813 → 0.935, false hits 23.0 → 10.3 a beat (10.2 → 3.7); levels 3.67 → 4.16 dB (3.21 → 3.79).
- **The kick view** (fix/kick-view, `layers.py`; re-fit in memory on m2g-vel's templates and stems, which reproduced every stored score, and the suite with render on the 16 (6) beats whose hits or plans change). A view right after its main, a merge's survivor by clean hits within one band, hand-offs at a tenth, no probe under an echo: analysis drums 0.880 → 0.883 (0.893 → 0.895), kick 0.872 → 0.882 (0.927 → 0.935), kick p90 6.13 → 5.28 ms.
- **The 808** (fix/808, `bass808.py`; `written()` replayed on inputs captured on all 96 beat runs, and analyze, fit-808, plan and score in a scratch folder on the changed beats). Notes end at a silence and start on their rise; the settled pitch read on the notes: written note right 0.947 → 0.970 (0.937 → 0.959), plan 808 note F1@50 0.926 → 0.921.
- **Speed** (fix/speed; tag fx-speed-2 at its own code, every step run, no cache). The 808 fit's filter, window and time axis made once; each of the 808 reader's fits made once; match's windows searched together; results kept by the `analyze` and `fit-808` commands, and by `suite run` only with `--cache`. Every output identical to m2g-vel's on 48 of 48 beats and to m2g-ft-vel's on the 8 htdemucs_ft beats run. A beat 140.3 → 112.0 s at the merged code, the mean over the 48 beats (m3a; 141.9 → 112.6 s over scenarios, [The suite](#the-suite)).
- **Pad files** (m3fix/pad-files, `library.py` and `pipeline.match`; match, plan and score run again over m3a). A silent file is never a match, a tied band names no sound: no pad plays a file under −60 dBFS (1 of 227 → 0; 1 of 229 → 0), plan levels 4.21 → 4.09 dB (3.79 → 3.70) against m3a. On htdemucs_ft full-1's balance went 1.17 → 3.14 dB: with its true kick at its kit level, the drum fader rose 6.7 dB and the beat's other pads play over their levels.

Merged, every lane's measured number held at m3a, the levels aside: with full-1's kick pad on a synth-bass file, m3a's plan levels were 4.21 dB (4.16 with the pad gate alone), which the pad-files round took to 4.09. The merge gained over the lanes: false hits 10.3 → 9.6 a beat, and layers-2's plan snare 1.0, which the kick view alone had at 0.

**The last two fixes**, each on a branch of its own and measured there, then merged as m4:
- **Pad levels** (levels/pads, `live.py` and `render.py`; tag lv-f3, plan and score over outputs equal to m3's). Each pad at a Volume of its own, read where the stems hold its sound, a kick on drums + bass less the plan's own 808, the loudest pad at 0 dB, the drum fader stopped where the loudest pad meets its stem: plan levels 4.09 → 1.27 dB (3.70 → 1.31), balance 3.03 → 0.56 dB (2.84 → 0.54).
- **The 808 one-shot** (glide/808, `synth808.py`, `pipeline.fit_808` and `live.py`; tag g8-f1, against m3a, whose 808 is m3's). The fitted voice on its template's periods, laid out as the template is, and `heard_cents` read from where the plan starts each note: plan 808 note F1@50 0.921 → 0.934 (0.910 → 0.934), median error 8.3 → 4.5 cents (13.6 → 9.6), bass_low-1 0.718 → 1.0 (0.449 → 0.981).

Merged, every 808 number held at g8-f1's. The levels moved from lv-f3's 1.27 to 1.25 dB (1.31 → 1.23) and 0.56 to 0.55 dB (0.54 → 0.50): better overall, worse in 11 of 22 scenarios by up to 0.20 dB (3, by up to 0.15).

**The Live fixes** (live/build, seven commits over 74931ce after the first live check, merged as m5; tags lb-2, lb-4 and lb-5, plan and score over m4's other outputs, each against the one before):
- **One voice per pad** in `render_plan`, the level estimate and `estimate_from_plan`, and a note Live starts late placed 1 ms early (lb-2): distance from a perfect kit 1.247 → 1.197 dB (1.231 → 1.186), balance 0.552 → 0.489 (0.497 → 0.502, 0.005 dB over m4's); every drum timing median within 0.02 ms of m4's.
- **The 808's release and restart** (lb-4): 1.197 → 1.198 dB (1.186 → 1.152), balance 0.489 → 0.512 (0.502 → 0.476); the 808's onset error median 1.360 → 1.355 ms (1.435 → 1.433), its median error 4.47 → 4.42 cents (9.65 → 9.63). A variant that read a late note's pitch from its written start took htdemucs_ft bass_low-1's plan 808 note F1@20 0.98 → 0.76; a note's pitch is read from where Live starts the sample.
- **The meter** (lb-5): meter right 0.909 → 0.955, downbeat F 0.892 → 0.924; nothing else moved.
- Snap off, the 808's filter off, the pads by content and the build's messages change nothing `render.py` plays.
- m4's plans scored as Live plays them (lb-5s, score only) sit 1.259 dB (1.242) from a perfect kit, balance 0.566 dB (0.506), the 808's onset error median 1.636 ms (1.574): the fair baseline for the numbers above.
- Not adopted: each pad read against its hits cut as Live plays them, 1.302 and 0.600 dB (1.273 and 0.511; lb-1); read only on the hits clear of its own hit before, 1.156 and 0.530 dB (1.281 and 0.542; lb-1b).

**The lanes of m2g, each alone**, on m1's other steps:
- **Drum discovery** (m1 → l1a-fix, 48 beats, `suite-l1a-fix.json`):
  - drums 0.431 → 0.830 at analysis, 0.319 → 0.547 in the plan;
  - kick 0.305 → 0.871 and open hat 0.303 → 0.979 at analysis;
  - the kick's median error +17.7 → +0.04 ms.
- **The joint fit** (m1 → l1b-fix, still on m1's templates). On the 22 beats of the layers, rolls, flams, dynamics, v1, v1orig, producer, full and kick808 scenarios:
  - drums 0.421 → 0.584 at analysis, 0.282 → 0.440 in the plan;
  - hats within 20 ms of a snare or clap found: 16 → 324 of 432;
  - false hits per beat: 102 → 32.

  Over all 48 beats (`suite-l1b-fix.json`): drums 0.431 → 0.530 and 0.319 → 0.396.
- **The 808 reader**, on the 808 lane's 10 scenarios (bass_low, glides, kick808, v1, v1orig, full, producer, melodic, tempo72, tempo160; 24 beats): note F1@50 0.289 → 0.918 at analysis, 0.230 → 0.902 in the plan (m1 → m2g; the merge left the 808's numbers as its lane left them).
- **Identification** (identify-fix, 20 beats of 8 scenarios, `suite-identify-fix.json`):
  - templates whose top match is their sound's own file: 49 of 78 → 55 of 80;
  - drum_pitch's largest pad pitch error: 30 → 0 cents;
  - plan strict F1@20 unchanged at 0.326: identity changes which sample a pad plays, not when.

After the merge, the same counts on the lanes' beats: hats near a snare or clap 376 of 432, and 75 of 95 templates with the right file.

## In Live

The plan carries every setting below, and `live.build_spec` sends it to `build_tracks` ([07](07-building-in-live.md)). Two live checks built plans with the build's calls and recorded them in Live 12.4.6, on 2026-10-29: eight m4 plans and PIRAMIT's, then four m5 plans and PIRAMIT's, after the fixes the first check led to.

**Measured in Live:**
- **Each pad plays its own file.** On the first check two pads played other files of their names, and a build stopped at a name the browser's search did not return. Each pad's file now goes into the User Library under a name of its content, and the build reads back the file each Simpler holds: on the second check all 30 Simplers held their plan's files.
- **Transpose and Detune** play the plan's pitch: pads at Transpose −1 and Detune +30, or Detune +25, and a pitched copy a semitone up, within 0.1 cent; the 808 within 0.11 cents of `render.py` at Detune −8, −1 and +1 (first check).
- **Pad Volumes**, sent as display text, are held a display step under it ("−6.6 dB" holds −6.605, "−14.9 dB" −14.95), and a hit plays what is held within 0.003 dB.
- **"Vol < Vel"** reads back 0.35, and a hit plays 0.4 × 0.35 × (velocity − 70) − 0.096 dB over its Volume, from velocity 35 to 127, on the pads and on the 808.
- **One voice per pad.** A hit that comes while its pad's hit before still plays cuts it and starts 1 ms late (391 hits on 18 pads, first check). `render.py` now plays it so: on the second check the 253 such hits started 0.004–0.019 ms after `render.py`'s (medians per pad), and every one of the 1,320 planned hits within 0.04 ms.
- **Snap off.** Hits play their whole files, their ends within 0.02 ms of `render.py`'s on kicks and snares; Snap on had cut them 4.7–27 ms short.
- **The 808**: the one-shot on its template's periods, the notes at their onsets, Live's 50 ms release, which `render.py` now plays, and a restart 1 ms late after a note of its pitch. On the second check 408 notes started within 0.11 ms of `render.py`'s, at −42.7 to −55 dB of residual. With its filter on, the first check found the notes delayed 0.13–0.46 ms by their pitch; the filter is off now.
- **Glide**, on a probe: Glide Mode and Glide Time read back, voices set to 1, the slide within 1.04 cents (p90) of `render.py`. No plan turns Glide on.
- **The meter.** A 3/4 plan is built in 3/4.
- **The scores.** On the four m5 beats, drum F1@20 per role read off Live's audio equals `render.py`'s. The recorded levels scored within −0.25 to +0.12 dB of `render.py`'s distance from a perfect kit and −0.12 to +0.40 dB of its balance: Simpler's top-end loss, which `render.py` does not model (hats up to 0.97 dB under it).
- **Melodic tracks and stems** (PIRAMIT, second check): 3 muted Drift tracks with their 1,150 notes as sent, 5 stems, 8 locators where planned.

**Not verified in Live:**
- `refdissect live-build` end to end. Live's Remote Script served 8 clients on both checks, which sent the build's calls through their own session.
- One `build_tracks` call of PIRAMIT's size (it went as 5), and a build that stops half way.
- htdemucs_ft plans, a plan with Glide, and choke groups. A choke group cuts with a 1 ms fade (the battery); `render.py` fades a cut over 5 ms, and the plan sets none.
- Locators off the Arrangement's grid: Live moves them, and the build deletes them (3 of meter34-1's 5).
- Bar 1: Live's bar 1 sits at 0 s, so full's first bar line, 0.4 s in, lands 400 ms off it.
- Simpler's top-end loss as a curve per sample rate and transposition.

## Known limits

All measured on the suite (m5 unless named; m4 equals m3 on everything but the plan's 808 and levels, and m5 equals m4 on everything but the plan's levels, meter and 808 timing; htdemucs_ft in brackets where given).

**Levels**
- **The kick against the snare, per beat.** The plan sits 1.20 dB from a perfect kit [1.15] and its balance at 0.51 dB [0.48] (1.25 [1.23] and 0.55 [0.50] at m4), against 4.09 [3.70] and 3.03 [2.84] at m3, and no scenario is further off on either than at m3. Per beat at m4 (not counted again at m5), |kick − snare| against the kit is 1.45 dB as a mean over scenarios [1.16] (4.67 [2.86] at m3), but further off than at m3 on 2 of 42 beats [6 of 43: drum_pitch-2, dynamics-2, full-2, meter68-2, producer-2, tempo160-2]:
  - bass_low-2, −19.09 → −26.59 dB, whose kick's hits play the hat's file (the analysis gave its kick sound the hat's file, and its hats, found 29 ms early, a kick file that is neither sound's); bass_low's kick sits 15.4 dB under the kit (12.1 at m3);
  - tempo95-2, +0.74 → −1.46 dB: its kick plays where it did at m3 (−1.44 dB from the kit at both) and its snare 2.2 dB louder (−2.18 → +0.02 dB). Its kick reads 1.74 dB lower under the new one-shot than under m3's (levels alone): its Volume went −0.5 → −2.2 dB and the drum fader, matched by RMS, −24.8 → −23.2 dB, which lifted the other pads.
  - on htdemucs_ft, dynamics' snare sits 3.8 dB under the kit (1.0 at m3-ft) and tempo72's 2.9 dB (0.4).
- **A kick is read under the plan's own 808**, so the drum levels move with the 808. Against levels/pads alone, the new one-shot moved 109 of 227 pad Volumes [102 of 229], by up to 1.7 dB [0.6], and the distance from the kit got worse in 11 of 22 scenarios [3], by up to 0.20 dB [0.15], while better overall. A kick with no plan 808 to read under (no one-shot, or no bass stem) is read with the 808 in it, or on the drum stem alone; no suite beat lacks either.
- **Why a kick is read on drums + bass.** Each pad set to the drum stem's level at its hits, with the drum fader at the stem's own level, scored 4.04 dB [1.54] over m3a's outputs, with the htdemucs kick 5.85 dB under the kit: htdemucs moves much of a kick into the bass stem.
- **Balance mixes two stem groups.** It is each sound against the median sound, the 808 included, and the htdemucs drum stem sits 5.52 dB under the truth's drums [2.51] where its bass stem sits 0.96 dB under the truth's bass [1.31]. With the drums at their kit level the 808's balance reads +2.50 dB [−0.14] (+2.70 [−0.05] at m4, −1.76 [−3.96] at m3), and from m3 to m4 per-role balance moved away from 0 in 25 [21] scenario values: on htdemucs mostly the 808's own (14), on htdemucs_ft mostly a role that was itself the median sound at m3 (15). Per-role `level_db`, against the truth stem, is worse for the same reason: over the suite by up to 5.8 dB (hats), in one scenario by 7.8 dB (rolls' hats) [only snares, by 3.0 dB, and toms, by 1.0]; `vs_kit_db` is the number within the kit.
- **The levels are the stems'.** A song has no truth to say where it holds each sound. PIRAMIT's perc, its loudest pad, is read on hits it mostly shares with the hat; on its 2 clear hits it reads 2.4 dB lower (3.0 at 74931ce; [02](02-case-study-piramit.md#in-live)).
- **One voice per pad moved dynamics** (m4 → m5): 1.27 → 2.16 dB from the kit. dynamics-2's drum track at unity plays 1.3 dB quieter with 79 hits cut, so the RMS match raised its fader −23.5 → −22.2 dB, and its snare and hats sit 2.0–2.4 dB over the kit.
- **A pad is read against its hits summed**, as the suite's truth sums them. A reference whose sampler cuts a sound's hits, as Live does, holds less of each, and there the reading runs low (PIRAMIT's kick: 81 of its 93 hits cut the one before). Read against the hits cut, the suite's plans sat 1.30 [1.27] dB from a perfect kit instead of 1.20 [1.19]; read only on the hits clear of their own hit before, 1.156 [1.281]. Neither is adopted. A late hit is also read where the plan writes it, 1 ms early: read where Live plays them, 37 [41] pad Volumes of 15 [16] of the m5 plans would move by −0.6 to +0.5 [−0.2 to +0.6] dB. Not fixed.
- **Simpler's top-end loss is not modelled.** In Live hats played 0.06–0.97 dB and snares 0.03–0.16 dB under `render.py` (second check), which took v1orig-3's balance from 0.17 to 0.57 dB. A 48 kHz file loses 0.32 dB at 8–12 kHz and 0.9–2.5 dB above 12 kHz, a 44.1 kHz file 1.6–1.8 dB at 12–16 kHz.
- **full-1 on htdemucs_ft**: with its true kick at its kit level (the pad-files round), the drum fader rose 6.7 dB and its hats, snare and open hat played over their levels: balance 1.17 → 3.14 dB at m3, 2.33 dB at m4.

**Drums**
- **melodic-1's kick lands late (B9).** The kick's own low-band template now times it: the suite's kick F1@20 on the beat went 0.748 → 0.929. But its window starts on the body too. Scored on the truth kick alone, its F1 at 15 ms went 0.748 → 0.679 and its median error 11.8 → 14.7 ms (m2g-vel → m3a); htdemucs_ft times the same kicks at 0.5 ms.
  - A view its main explains within ±10 ms is left out. One whose window sits later on the sound stays, and the merge keeps the template averaged from more hits: a view 12–20 ms late on a synthetic kick takes all 24 kicks, late (a test pins it as an expected failure, at 12 and at 20 ms).
  - Merging into the template the fit takes first fixes that case but costs htdemucs_ft producer-2's plan snare (0.401 → 0.262) and PIRAMIT's kick pad 93 → 78 notes. Not adopted.
- **Which of two templates is the purer sound.** Merges keep the template with more clean hits only within one band. On layers-2 the snare-band snare template is kept; the low-band one would find 13 more hats under the snares (0.945 → 0.993). The plan plays the snare either way (plan snare on layers 0.5 → 1.0 since the pad gate).
- **Hat timing.** The analysis' p90 is 3.34 ms (3.33 at m2g-vel; 2.55 ms in the discovery lane).
  - Hats held by a snare's own top end at 15–24 ms remain on melodic-1, leadin-1 and tempo160-1, whose hat p90s are 22.2, 16.6 and 20.7 ms (their seed-2 beats: under 0.1 ms), and in producer (5.3 ms, the scenario's mean).
  - bass_low's hats are almost missing: F1@20 0.006, p90 32.0 ms. bass_low-2's are found 29.1 ms early, and in the plan their pad sets the mean of the hat's signed median: −1.24 ms over the suite, while 20 of 22 scenarios lie within 0.5 ms.
- **The plan's snare p90** rose 3.04 → 3.45 ms [2.91 → 3.60] over the 20 scenarios m2g-vel times a plan snare in (tempo72's and tempo95's snares had no pad; over m3's 22, 3.18 [3.31] ms): producer-1's snare, now played by a pad, carries the analysis' 41.8 ms p90. Without producer it fell, 3.16 → 2.53 ms [3.04 → 2.69].
- **A sound that always plays with another has no template of its own.** The layers scenario's clap sits on every snare, 0–8 ms late, and scores 0. Claps are 0.500 over the suite.
- **Pitched copies of residual templates are off.** Re-planned on 20 beats, allowing them gave rolls-2 +0.054 and producer-2 +0.051, but rolls-1 −0.043 (m2g).
- **False hits.** 41.7 a beat at analysis level over the 48 beats (42.4 at m2g-vel) [35.1]; on the layer lane's 22 beats, 35.6 a beat at m2g, against 31.5 in that lane.
  - Sounds that play no truth sound: 1.96 a beat in the analysis [1.98], 0.05 with a pad in the plan [0.02]. Most are residual templates on transients of the melodic loop that htdemucs left in the drum stem.
  - Recurrence keeps them off the pads because the suite's loop plays at its own tempo. A loop on the beat's tempo recurs: with recurrence waived, 0.37 such sounds get a pad [0.23] and the plan has 16.3 false hits a beat [7.1] (an estimate for such songs, on m2g-vel's analyses). No test of junk that does not depend on the loop's tempo is in the code.
  - Single beats got more in the plan: v1-2 6 → 9, drum_pitch-2 4 → 5 [producer-2 4 → 9, rare-1 2 → 3]. Not investigated.
  - On v1-2 and rare-2 a hat-band view of the kick (14 and 11 notes) gets a pad of its own. It plays no strike twice: all its notes lie more than 50 ms from the main kick's, and 8 and 9 of them within 20 ms of truth kicks the main missed.
- **One recording on two pads** can play one strike twice: the rule that plays a strike once looks at one pad. PIRAMIT's new hat doubles 2 of the main hat's notes 0.3 ms away with the same recording ([02](02-case-study-piramit.md#in-live)).
- **The backbone's strike rule is a timing rule.** A kick or snare template's hit within 15 ms of a backbone hit is not played, so a real kick layered on every backbone kick would not be (the layers scenario's second kick scored 0 before the rule too). Linking band views in the analysis would decide it there; not done.
- **A template's own ring is no hit.** That rule could drop a real ghost note that sits at a snare's tail level. The suite has no such case: on 48 beats the rule removed 55 snare-band hits, none of them a true hit of the template's sound.
- **A residual sound heard once every 16 bars never gets a pad.** Recurrence looks 1 to 8 bars away, so 5 hits 16 bars apart recur at 0 (8 bars apart: 1.0), however clean or close their match: a crash on each 16-bar section of a 72-bar song, which the gate's floor (5 hits there) lets through. The suite's beats are too short to hold one; not measured on a song.
- **Cases the code does not handle, met on no suite beat** (found in review):
  - clean hits are counted once, before the merges, so a later merge round compares counts that an earlier round's merge did not update (`layers._one_each`); no suite beat has a two-way merge within one band;
  - an `analysis.json` written before templates carried `found` (before m2g) reads every template as found on the grid: a residual template is then not held to recurrence and can be backbone. Analyze again;
  - `render_plan` raises where its one-voice cut gets a negative length: a pad hit less than 1 ms after a late hit of the same pad, two notes of one pad at one time after a late hit, or a restarted 808 note shorter than 1 ms. `make_plan` writes none of these (a template's hits lie 8 ms apart or more, and a pad plays a strike once within 15 ms); a plan edited by hand can.

**808**
- **A note's own cents.** One Detune per Simpler plays every note at the line's tuning ([05](05-808-synthesis.md#in-live)). On full the analysis writes one note a semitone high and hears 5 short notes 41 cents low, which m3's one-shot, falling too fast, happened to offset: full's plan note F1@50 0.890 → 0.851 [0.910 → 0.890]. producer's plan median error rose 6.8 → 8.2 cents [15.2 → 21.1], its note F1@50 0.770 → 0.804 [0.698 → 0.770]. m3's regression on bass_low-1, one Detune for notes of every length under a fitted one-shot that fell faster than the 808, is resolved: plan note F1@50 0.718 → 1.0 [settled within 50 cents 0 → 0.987].
- **The one-shot's head** is not judged by ear: the pitch step at the first crossing followed, the quiet (20 %) head of a one-shot whose voice attacks early and the cut head of one whose voice attacks late. On PIRAMIT its waveform matches the fit's reference less than the m3 code's one-shot did (NCC 0.880 against 0.921), while its pitch follows the template's.
- **The 20 ms edge.** One of swing58-1's 808 notes, 19.981 ms after the truth's at m3, sits 20.001 ms after it at m4, outside 20 ms: plan strict F1@20 0.8757 → 0.8756, 808 onset F1@20 0.9348 → 0.9344.
- **bass_low-2 on htdemucs_ft**: its template is read from its tail 57 cents flat. `settle()` refuses its reading (the window's halves lie 120 cents apart), and would only lower a template anyway, so its written pitch stays at 0.130. On htdemucs_ft bass_low's note F1@50 is 0.338 (plan 0.513 at m4, bass_low-1 0.981).
- **An attack under 9 dB** over a note still ringing, with no kick on it, stays in that note (once on PIRAMIT, 6.5 dB). Candidates inside a silence are dropped after the segmentation, not before it.
- **Legato notes are off by default.** Legato recall is 0; the truth has 24 legato notes on 5 beats.
  - The opt-in slide rule was tuned on the beats it was measured on.
  - Merging every unattacked note into the note before loses a few real notes (tempo160-1 1.000 → 0.946 in the lane).
  - Two 16th notes at one pitch at 160 BPM are split only in part: 2 of the 4 pairs of the unit-test bar still merge.
- **The plan's 808 sits 1.9 dB under a perfect kit** (−1.87 dB [−2.01] at m4 and m5; −1.98 dB at m3, −1.97 dB at m2g-vel, −1.95 dB at m2g; +0.09 dB in m1). Its fader matches the bass stem's whole-track RMS, which now sees the notes' real lengths.
- **The tails of long legato chains are silent.** The one-shot is 3 s long.
- **The 808 reader and the plan's estimate disagree on short restarted notes.** The suite's plan-level 808 scores come from `estimate_from_plan`. Run on the plan's own 808 audio, Live's or `render.py`'s, the pipeline's reader reads bass_low-1's 64 notes of 110 ms 63–68 cents flat: note F1@50 0.59, where the estimate gives 1.0. With m4's notes, before restarts started 1 ms early, it read 6 of the 156 more than 50 cents off. On the truth's own 808 stem it misreads 92 of them, so it cannot say which is right.
- **With Glide** (one voice), the restart and the release's overlap are not modelled. No plan of the suite or of PIRAMIT turns Glide on.
- **A beat without an 808** gets an 808 track that doubles its kicks where the bass stem holds the kick's leak: the reader learns a template from the leak, fit-808 fits a one-shot to it, and the plan plays it on the kicks. On synthetic stems (8 bars at 140 BPM, 32 kicks, no 808) a leak at 0.15 or 0.5 of the kick's level gave 31 notes from a 71.2 Hz template, the same at 4f30623; a silent bass stem gives none. Nothing refuses a template learnt on drum attacks alone, and the suite has no beat without an 808.
- **A kick inside a held note** is taken back into it (`legato()`: an attacked note of 120 ms or less whose follower settles at the held note's pitch). That rule can take a note that a rest or a rise kept, and hold the note before over the rest (found in review). No suite beat reaches it: the silence and rise rules change one beat run of 96.

**Identification**
- **Library layers** (`matches.json` `layers`) are searched but not played. No layer was given on the 20 beats of the identify lane.
- **Kick sources are off by default.** fit-808 runs after match, and with fit-808's one-shot the rule found 13 of 20 kicks against 12 without it.
- **Wrong files.** 75 of 95 templates have the right file on the identify lane's 8 scenarios (m2g). The misses are hat templates that carry the snare on beat 3, and near-copies across kits.
- **Pads that play a file of another role** (not judged by ear): dynamics-2's kick pad plays a snare file (NCC 0.813), melodic-1's snare pad a hat file (0.928), tempo160-1's snare pads a hat file (0.696); on htdemucs_ft rolls-1's hat template, matched as a kick, plays the truth's hat file on kick pads (plan kick F1@20 0.466 there).
- **Files over full scale** play unscaled: 1 % of the library's entries peak over +6.4 dBFS, the loudest +19.4.
- **An index built before the −60 dBFS floor**, as the suite's is, reads each listed entry's file once: 4.7 % of match over 6 beats. `index-library` again removes the reads. A `matches.json` written before the floor can still name a silent file; match again.

**Melodic and rhythm**
- **Chords follow the sections, which follow the drum hits.** In an intermediate run (m2e), full-2's section A split and majmin fell 0.999 → 0.961 with the same notes.
- **The plan's time.** The plan carries the grid's meter (m5), which the build sets in Live.
  - The plan's meter is right on 0.955 of the scenarios (0.909 at m4), and its downbeat F is 0.924 (0.892): meter34 is planned and built in 3/4.
  - Live's bar 1 is still at 0 s: full's first bar line, at 0.4 s, lands 400 ms off.
  - A locator off Live's grid is deleted: 3 of meter34-1's 5 in Live, where the grid follows the Arrangement's zoom. 11 of the 169 locators of the m5 and m5-ft plans lie off Live's bar lines (full and tempo72), likely to go the same way (not built).
- **Bar lines and meters missed.**
  - tempo72's bar line is wrong on both beats (downbeat F 0).
  - meter68 is read in 4/4, at x2/3.
  - The seed-1 swing beats read as straight 8ths.

**Speed and models**
- **analyze takes 34.4 s a beat** at 2 jobs (m3a; 38.7 s at m2g, 5.7 s in m1). The 808 reader was about 30 s of it at m2g by its lane's measure; the rest is the joint fit and the attack-based discovery, not separated. match takes 28.0 s (49.1 at m2g, 24.4 in m1), fit-808 46.2 s (50.9). melodic went 2.7 → 3.3 s, not investigated.
- **The seconds are not a controlled comparison.** m3a ran with Live open (load 3.2–6.1), m2g on a quiet machine. The fit's popsize was not changed (a DE popsize of 10 fitted the 808 1.9–2.2× faster at equal mean loss on 4 m1 beats, not run on all 48), and the tests run in one process (no pytest-xdist).
- **The caches key on the whole package.** Any edit to any `.py` file makes every kept result stale, so `--cache` helps re-runs at the same code only. Code patched in memory is not seen, nor an edit undone before the next check.
- **Long input.** In the discovery lane, analyze peaked at 2.27 GB on 192 s of tiled input. That was not measured again after the merge; on PIRAMIT's 126 s it peaks at 2.16 GB (e2a5ab4).
- **htdemucs_ft is not the default.** Its drums are better (plan 0.868 → 0.898; at m2g +0.039 [+0.017, +0.062], paired over the 48 beats). But its 808 notes are worse (0.936 → 0.924; bass_low 0.949 → 0.338), and so are its snares (0.961 → 0.891).
- **Songs have no truth.** These numbers come from synthetic beats; PIRAMIT's measurements are in [02](02-case-study-piramit.md).

## Next

In order ([10](10-roadmap.md) has the metrics, targets and gaps):
1. **The plan with Live's measured behaviour:** Simpler's top-end loss in `render.py`, choke groups, bar 1 and the locators off Live's grid; the levels per beat (the kick against the snare, dynamics) and balance within each stem group. Velocities follow Live's law (m2g-vel), pads are given on evidence (m3), each pad has a level of its own (m4), and Snap, one voice per pad, the 808's release and the meter follow Live (m5).
2. **The 808 in the plan:** a note's own cents (full, producer), the reader against the estimate on short restarted notes, and the one-shot's head by ear. Its pitch on short, low notes is done (m4).
3. **The build in Live:** `refdissect live-build` end to end, one `build_tracks` call of a song's size, htdemucs_ft plans, Glide and a build that fails. Suite plans and PIRAMIT's were built, recorded and scored as the plan is scored on 2026-10-29 ([In Live](#in-live)).
4. **The closed loop:** measure the recording, correct levels, pitch and timing, rebuild, and stop when the distance stops falling.
5. **Speed:** popsize 10 on the 48 beats, the tests in parallel, and a controlled timing of the whole suite.
6. **The open limits above.**

## Sources

Where a choice rests on published work:
- **Separation.** Demucs, htdemucs and htdemucs_ft: Rouard, Massa and Défossez, "Hybrid Transformers for Music Source Separation", ICASSP 2023 ([arXiv:2211.08553](https://arxiv.org/abs/2211.08553)); code at [github.com/adefossez/demucs](https://github.com/adefossez/demucs).
- **Beats and downbeats.** Foscarin, Schlüter and Widmer, "Beat this! Accurate beat tracking without DBN postprocessing", ISMIR 2024 ([arXiv:2407.21658](https://arxiv.org/abs/2407.21658)); code and weights MIT, at [github.com/CPJKU/beat_this](https://github.com/CPJKU/beat_this). madmom's models were left out: they are CC BY-NC-SA.
- **Melodic notes.** Bittner, Bosch, Rubinstein, Meseguer-Brocal and Ewert, "A Lightweight Instrument-Agnostic Model for Polyphonic Note Transcription and Multipitch Estimation", ICASSP 2022 ([arXiv:2203.09893](https://arxiv.org/abs/2203.09893)); [github.com/spotify/basic-pitch](https://github.com/spotify/basic-pitch).
- **Scoring.** [mir_eval](https://github.com/mir-evaluation/mir_eval) 0.8 (Raffel et al., ISMIR 2014): one-to-one event matching, note F1, the beat F-measure and the chord comparisons.
- **The joint fit.** Its relative is NMF deconvolution with fixed drum templates, which explains overlapping hits together in the magnitude domain. See the review by Wu et al., IEEE/ACM TASLP 2018 (its [result tables](https://audiolabs-erlangen.com/resources/MIR/2017-DrumTranscription-Survey)), and [libnmfd](https://github.com/groupmm/libnmfd). Here the fit runs on waveforms, one moment at a time, with non-negative least squares.
- **The 808's settled pitch.** QIFFT, quadratic interpolation of FFT magnitude peaks: Abe and Smith, AES 117th Convention, 2004 ([PDF](https://ccrma.stanford.edu/files/papers/stanm117.pdf)); J. O. Smith, [Spectral Audio Signal Processing](https://ccrma.stanford.edu/~jos/mus423h/Sinusoidal_Parameter_Estimation_based.html).
- **The speckle index** is this project's own. Related: Raffel and Ellis, "Estimating Timing and Channel Distortion Across Related Signals", ICASSP 2014 ([PDF](https://www.ee.columbia.edu/~dpwe/pubs/RaffelE14-remix.pdf)), which estimates the resampling between a remix and its source; Cohen, "The Scale Representation", IEEE Transactions on Signal Processing, 1993 ([PDF](https://www.ee.columbia.edu/~dpwe/papers/Cohen93-scale.pdf)).
- **Simpler.** Its Transpose (±48 semitones) and Detune (±50 cents) are in the [Live 12 manual](https://www.ableton.com/en/live-manual/12/live-instrument-reference/). That both keep whole numbers was measured in Live 12.4.6, as above.
