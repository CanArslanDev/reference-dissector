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
- The build in Live is not verified: no replica-mode plan has been built in Live and recorded yet.

| means over 22 scenarios, analysis / plan | before (base-htdemucs) | now (m2g) |
|---|---|---|
| drum hits and 808 onsets, strict F1@20 | 0.368 / 0.282 | 0.886 / 0.667 |
| drums, strict F1@20 | 0.397 / 0.283 | 0.880 / 0.612 |
| 808 notes, F1 (50 ms, 50 cents) | 0.278 / 0.229 | 0.936 / 0.926 |
| 808 pitch, median abs error | 55.3 / 69.8 cents | 6.0 / 7.7 cents |
| plan levels: median distance from a perfect kit | 7.18 dB | 3.67 dB |

`base-htdemucs` is the code before replica mode (471fd70) and `m2g` the merged code (9ccc759). Their scores are `results/calibration/suite-base-htdemucs.json` and `suite-m2g.json`.

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
- `--resume`: skips the beats this tag has scored already.
- `--steps`: a subset of analyze, melodic, match, fit-808, plan and score. Without `--reuse`, a step whose input no step makes is refused before anything runs.
- `--reuse TAG`: copies the outputs of the steps not run from `runs/TAG/`, in place when TAG is the run's own tag. It refuses before running when a beat lacks those outputs, failed there, or analysed another model's stems.
  - A reused step that reads what a run step rewrites is marked `reused_stale`, and the warning and the report name it. melodic reads the 808 notes, match the drum templates, plan everything.
  - The 808 lane's runs reused m1's chords, built on the old 808 notes. Rebuilt on the new ones, majmin went 0.888 → 0.977 on the four melodic and full beats.
- analyze's and match's options pass on unchanged.

**`suite report`** writes markdown tables of each headline number, per scenario and per drum role, for every tag, with the change from the first tag to the last.

**`scripts/export_results.py --suite`** publishes the means per scenario only, as `results/calibration/suite-<tag>.json`: no beat, file, path, note or event time.

**Time and memory** (m2g: 48 beats, 2 jobs, Apple M4, stems already separated):
- A beat took 142 s: analyze 38.7 s, match 49.1 s, fit-808 50.9 s, melodic 2.7 s; plan, render and scores under 1 s together.
- The 48 beats took 3,378 s (56 minutes).
- A job peaked at 1.71 GB, the process tree at 3.3 GB.
- A tag's run folders take about 225 MB.
- Separation, once per model: 2.1 s a beat with htdemucs, 8.5 s with htdemucs_ft (MPS).

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
| `analyze`: drums | `analysis.json` `templates`, `events`, `layers`; `templates/` | Templates where sounds start, one per sound, then a residual pass. Every moment is fit with every template: layers, flams, rolls, pitched copies. Each hit counted once. |
| `analyze`: 808 | `bass808.json`; `analysis.json` `notes_808`, `tuning` `808` | Notes read as transpositions of the song's own 808, to the cent. The 808's tuning is its template's settled pitch. |
| `melodic` | `melodic.json`, `basicpitch/` | New: Basic Pitch's notes per tonal stem, and chords with the 808 as their bass. |
| `match` | `matches.json` | Snare and hat bands at any pitch, to a quarter cent (the speckle index). Transpose and Detune per entry. Library layers are searched. |
| `fit-808` | `synth808.json`, `808 (synth).wav`, `templates/808.npy` | Fitted on the template's notes. The one-shot is rendered at the template's settled pitch. |
| `plan` | `plan.json`, `midi/` | Pads play their match's Transpose and Detune. Drum notes go to `t_attack`. The 808 plays its written pitches with Transpose, Detune and, with slides on, Glide. A muted MIDI track per tonal stem; locators on bar lines. |
| `live-build` | the Live set | Detune per pad; the 808's Detune, Glide Mode, Glide Time and voices. Not verified in Live. |
| `suite run` | `scores.json`, `scores-<tag>.json` | Strict scores of the analysis and of the plan against the truth's `events.json`. |

Each step is described in [01](01-pipeline.md), the 808 voice in [05](05-808-synthesis.md), the build in [07](07-building-in-live.md).

### analysis.json

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
  - The template: `hz`, `midi`, `root_midi`, `tuning_cents`; `method`, how its settled pitch was read (QIFFT of the sustain, or the asymptote of its fitted drop); the onsets it was learnt from; its pitch curve.
- **melodic.json**: per tonal stem, the notes (pitch, cents, beats, velocity) and the stem's tuning, and the chord segments. `basicpitch/` keeps the model's notes and posteriorgrams.
- **matches.json**, per template:
  - `role`, `matched_in_band`, `same_as`;
  - `search`: `speckle` or `semitones`;
  - `top`: the five best entries, each with `semitones` (fractional), `transpose` (whole semitones), `cents` (`detune_cents` is the same number under Simpler's name), `ncc` and `lag_ms`;
  - `margin`, `per_hit`, `null_test_db`, `verdict`;
  - `layers` and `layer_search`: a second sound and why it was or was not accepted;
  - `source` and `kick_source`, with `--kick-sources`.
- **synth808.json**: the voice's parameters, and the pitch the one-shot was rendered at.
  - `pitch_hz` is that pitch. `pitch_from` names it: the measuring method, `fit` or `agree`.
  - `measured_hz` is the template's pitch, and `fit_vs_measured_cents` how far the fit's own pitch lies from it.
- **plan.json**:
  - `tempo`, `length_beats`, `stems`, `locators`, `levels`;
  - `tracks.drums`: `pads` (note → template, file, `transpose`, `detune_cents`, NCC, verdict, role) and `notes`;
  - `tracks.808`: `notes`, `sample_path`, `transpose`, `detune_cents`, `tuning_cents`, `heard_cents`, `slides` (every glide), and with Glide on `voices`, `glide_mode` and `glide_time_ms`;
  - `tracks.melodic`: a muted track per tonal stem.
- **events.json** (schema `refdissect.events/1`) holds a truth or an estimate in one form, so one scorer reads both.
  - A scenario writes the truth (`kind: "truth"`):
    - the grid (`time`);
    - every drum hit (`sound`, `t`, `t_attack`, `gain`, `velocity`, `layer_of`, `flags`);
    - every 808 note (`t`, `t_attack`, `end`; `midi`, the pitch it sounds at, cents included; `midi_settled`, the pitch it settles to; `glide`);
    - the melodic notes (`layer`, `t`, `end`, `midi`, `cents`, `velocity`) and the chords (Harte labels);
    - every sound (`role`, `path`, `sha256`, `semitones`, `cents`, `attack_s`).
  - `evaluate.estimate_from_analysis` and `evaluate.estimate_from_plan` build estimates (`kind: "estimate"`) in the same form. `evaluate.validate` checks a file, and `evaluate.save_events` writes one.
- **scores.json**, per beat and tag: the analysis's and the plan's scores, each step's status (`run`, `reused`, `reused_stale`, `skipped`), the seconds per step and the problems. `scores-<tag>.json` holds the means per scenario.

All of these hold note times or library file names: they stay in the work folder ([09](09-data-and-copyright.md)).

## How it is scored

[03](03-calibration.md) has the details. In short:
- **Strict.** Truth and estimate both time a hit where its sound attacks (`audio.attack_time` in its role's band), and no offset is removed. A hit counts within 20 ms.
- **Two levels.** The analysis is scored from `analysis.json` and `matches.json`. The plan is scored from `plan.json` as `render.py` plays it: each pad's sample at its Transpose and Detune, the 808's one-shot per note.
- **Drums.** Each estimated sound plays the truth sound it scores best on. A truth sound's F1 pools the hits of every estimated sound that plays it, so a hit played twice is a false hit. A pitched copy or a layer is a sound of its own.
- **Means.** Over a scenario's beats, then over the 22 scenarios.

## What each stage measured

Means over the 22 scenarios of calibration v2. Columns are tags: `base` = base-htdemucs (471fd70), `m1` = after the rhythm and melodic steps (e1bd9ce), `m2g` = after the four lanes and the merge's fixes (9ccc759), `base-ft` = base-htdemucs_ft and `m2g-ft` the same code on htdemucs_ft stems. A cell is "analysis / plan" where both exist, and "–" where that tag's scorer had no such score or the step did not run.

| | base | m1 | m2g | base-ft | m2g-ft |
|---|---|---|---|---|---|
| **End to end** | | | | | |
| drum hits and 808 onsets, F1@20 | 0.368 / 0.282 | 0.396 / 0.311 | 0.886 / 0.667 | 0.417 / 0.324 | 0.898 / 0.700 |
| **Grid** (beat_this, m1) | | | | | |
| beats, F (70 ms) | 0.715 / 0.920 | 0.962 / 0.962 | 0.962 / 0.962 | 0.686 / 0.936 | 0.962 / 0.962 |
| downbeats, F | 0.637 / 0.823 | 0.939 / 0.892 | 0.939 / 0.892 | 0.614 / 0.839 | 0.939 / 0.892 |
| meter right | 0.91 / 0.91 | 0.95 / 0.91 | 0.95 / 0.91 | 0.91 / 0.91 | 0.95 / 0.91 |
| tempo in the right octave | 0.89 | 0.91 | 0.91 | 0.91 | 0.91 |
| **Drums** (discovery and the joint fit, m2g) | | | | | |
| all drum sounds, F1@20 | 0.397 / 0.283 | 0.431 / 0.319 | 0.880 / 0.612 | 0.470 / 0.354 | 0.893 / 0.648 |
| kick | 0.247 / 0.182 | 0.305 / 0.201 | 0.872 / 0.845 | 0.581 / 0.637 | 0.927 / 0.919 |
| snare | 0.695 / 0.493 | 0.801 / 0.600 | 0.961 / 0.813 | 0.556 / 0.319 | 0.891 / 0.816 |
| hat | 0.508 / 0.549 | 0.504 / 0.569 | 0.859 / 0.845 | 0.626 / 0.589 | 0.913 / 0.916 |
| open hat | 0.326 / 0 | 0.303 / 0 | 0.979 / 0.043 | 0.324 / 0.006 | 0.993 / 0.022 |
| kick timing, median (p90), analysis | +16.6 (20.9) ms | +17.7 (22.0) ms | 0.0 (6.1) ms | +2.0 (7.6) ms | −0.5 (3.4) ms |
| snare timing | −6.4 (8.3) ms | −6.0 (7.7) ms | +0.8 (3.2) ms | −6.1 (8.0) ms | +0.3 (3.8) ms |
| hat timing | +1.1 (3.4) ms | −0.5 (2.9) ms | −0.4 (3.3) ms | +0.2 (2.2) ms | −0.3 (1.7) ms |
| sounds that play no truth sound, per beat | 0.16 / 0.05 | 0.16 / 0.05 | 1.96 / 0.71 | 0.18 / 0.02 | 1.95 / 0.32 |
| **808** (bass808, m2g) | | | | | |
| onsets, F1@20 | 0.270 / 0.303 | 0.270 / 0.303 | 0.935 / 0.935 | 0.207 / 0.214 | 0.944 / 0.944 |
| notes, F1 (50 ms, 50 cents) | 0.278 / 0.229 | 0.278 / 0.229 | 0.936 / 0.926 | 0.173 / 0.146 | 0.924 / 0.910 |
| notes, F1 (20 ms, 50 cents) | 0.160 / 0.159 | 0.160 / 0.158 | 0.911 / 0.901 | 0.102 / 0.092 | 0.913 / 0.896 |
| pitch, median abs error | 55.3 / 69.8 cents | 55.3 / 69.8 cents | 6.0 / 7.7 cents | 72.9 / 78.5 cents | 12.1 / 13.9 cents |
| written note right | – | – | 0.947 | – | 0.937 |
| onset timing, median (p90), analysis | −4.1 (37.1) ms | −4.1 (37.1) ms | +1.4 (8.0) ms | −3.3 (37.4) ms | +1.4 (7.0) ms |
| **Identification** (match v2, Detune per pad; plan only) | | | | | |
| truth sounds a pad plays with their own file | – | – | 0.574 | – | 0.641 |
| largest pitch error of such a pad | – | – | 0 cents | – | 0 cents |
| **Melodic** (Basic Pitch and chords, m1; 4 beats) | | | | | |
| notes, F1 (50 ms, 50 cents) | – | 0.630 | 0.630 | – | 0.635 |
| chords, majmin | – | 0.888 | 0.977 | – | 0.979 |
| **Levels** (plan, rendered) | | | | | |
| median abs distance from a perfect kit | 7.18 dB | 7.19 dB | 3.67 dB | 3.81 dB | 3.21 dB |
| median abs balance | 4.15 dB | 3.82 dB | 2.25 dB | 2.67 dB | 1.98 dB |
| the 808 against the kit | +0.09 dB | +0.09 dB | −1.95 dB | +0.96 dB | −2.23 dB |
| **Seconds per beat** (jobs) | 190 (3) | 90 (2) | 142 (2) | 203 (3) | 141 (2) |

- **The grid** changed in m1 and is unchanged since. The four lanes did not touch it.
- **The 808** changed only in its lane. Its numbers are the same in m2 and m2g.
- **Chords** rose from m1 to m2g because melodic now reads the new 808 notes as the chords' bass.
- **Kick timing on htdemucs_ft** was already near 0 before (base-ft +2.0 ms). On htdemucs drums a kick's body comes 26 ms after its click, as the rest of the kick goes to the bass stem, and the old templates timed kicks on their body.
- **m2 and m2-ft** (`suite-m2.json`, `suite-m2-ft.json`) measured the first merge (ba7b69d), before five merge fixes: drums 0.847 / 0.600, open hat 0.901, largest pad pitch error 7.5 cents.

**Each lane alone**, on m1's other steps:
- **Drum discovery** (m1 → l1a-fix, 48 beats, `suite-l1a-fix.json`):
  - drums 0.431 → 0.830 at analysis, 0.319 → 0.547 in the plan;
  - kick 0.305 → 0.871 and open hat 0.303 → 0.979 at analysis;
  - the kick's median error +17.7 → +0.04 ms.
- **The joint fit** (m1 → l1b-fix, still on m1's templates). On the 22 beats of the layers, rolls, flams, dynamics, v1, v1orig, producer, full and kick808 scenarios:
  - drums 0.421 → 0.584 at analysis, 0.282 → 0.440 in the plan;
  - hats within 20 ms of a snare or clap found: 16 → 324 of 432;
  - false hits per beat: 102 → 32.

  Over all 48 beats (`suite-l1b-fix.json`): drums 0.431 → 0.530 and 0.319 → 0.396.
- **The 808 reader**, on the 10 scenarios with an 808 to read (24 beats): note F1@50 0.289 → 0.918 at analysis, 0.230 → 0.902 in the plan (m1 → m2g; the merge left the 808's numbers as its lane left them).
- **Identification** (identify-fix, 20 beats of 8 scenarios, `suite-identify-fix.json`):
  - templates whose top match is their sound's own file: 49 of 78 → 55 of 80;
  - drum_pitch's largest pad pitch error: 30 → 0 cents;
  - plan strict F1@20 unchanged at 0.326: identity changes which sample a pad plays, not when.

After the merge, the same counts on the lanes' beats: hats near a snare or clap 376 of 432, and 75 of 95 templates with the right file.

## Built, but not verified in Live

The plan carries every setting below, and `live.build_spec` sends it to `build_tracks`. None of it has been read back from Live or recorded with a replica-mode plan.

- **Detune per pad.** Each pad's Simpler gets the match's Transpose (whole semitones) and Detune (whole cents). The plan-level pitch score (0 cents off for every pad that plays its sound's file) reads the plan, not Live.
- **Pitched pads.** From note 54 up, the same file transposed further. A pad there keeps its base pad's Detune.
- **The 808's Transpose and Detune** (`tune_808`, with `heard_cents`), sent as Simpler parameters.
- **Glide.** With slides on and three legato notes or more, the 808 gets Glide Mode "Glide" and the median Glide Time. Slides are off by default, so no plan of the suite turns Glide on (0 legato notes written under m2g).
- **Voices.** One voice for Glide, set with `lom_set_property` after `build_tracks`: voices is a property of the device, not a parameter.
- **Melodic tracks.** A muted MIDI track per tonal stem, with Drift as a placeholder instrument.
- **The build as a whole.** The plan-level numbers come from `render.py`, which assumes how Simpler plays. Live 12.4.6 was measured on 2026-10-19 (ableton-live-mcp's replica-mode branch records it in its CLAUDE.md), and three of those assumptions differ from it:
  - **Velocity.** `render.py` assumes a linear v/127. In Live a velocity sets the level by dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over the Volume: "Vol < Vel" is 35 % on a Simpler made with `insert_device` and 45 % on a pad loaded through the browser.
  - **Snap.** It is on by default and ends a sample 10–30 ms early; `render.py` plays every sample to its end.
  - **Chokes.** A choke group cuts with a 1 ms fade; `render.py` fades a cut over 5 ms, and the plan sets no choke groups.

  The onset scores do not depend on these three; the level scores do.
- **Bar 1.** The plan holds no meter, and Live's bar 1 sits at 0 s.
- **Loading.** Pads still load through Live's browser by file name (ableton-live-mcp 2.2.0). The replica-mode branch of ableton-live-mcp adds `sample_path`, which loads the exact file and reads it back; the build does not use it yet.

## Known limits

All measured on the suite (m2g unless named).

**Drums**
- **The plan's 12-hit pad gate.**
  - Open hats have 4–8 hits: they score 0.043 in the plan against 0.979 in the analysis.
  - tempo72's and tempo95's snares have 8 hits and get no pad: 0 in the plan on every tag.
- **Drum notes go to `t_attack`** without taking out the pad sample's own attack: plan medians kick +2.1 ms, snare +1.1 ms, hat +0.6 ms.
- **melodic-1's kick.** It scored 0.929 in the discovery lane (l1a-fix) and 0.748 now.
  - `relations()` takes templates by hit count, and the low-band kick (15 hits) became a composite of a snare-band residual view of itself (18 hits).
  - Taking grid templates before residual ones restores the kick (0.929) but costs that beat's hat (0.453 → 0.306). Not adopted.
- **The plan's snare.** It scored 0.834 in the discovery lane and 0.813 now.
  - On layers-2 a merge moved the snare to its less clean template (clean 0.44, under make_plan's 0.5), so no pad plays it.
  - Merges, composites and `distinct()` choose between templates by energy and hit count, not by alignment.
- **Hat timing.** The p90 is 3.3 ms (2.55 ms in the discovery lane).
  - Hats held by a snare's own top end at 15–24 ms remain on melodic-1 (10.9 ms), leadin-1 (8.3 ms), tempo160-1 (10.4 ms) and producer (5.3 ms).
  - bass_low's hats are almost missing: F1@20 0.006, p90 32.0 ms.
- **A sound that always plays with another has no template of its own.** The layers scenario's clap sits on every snare, 0–8 ms late, and scores 0. Claps are 0.500 over the suite.
- **Pitched copies of residual templates are off.** Re-planned on 20 beats, allowing them gave rolls-2 +0.054 and producer-2 +0.051, but rolls-1 −0.043.
- **False hits.** On the layer lane's 22 beats there are 35.6 a beat at analysis level, against 31.5 in that lane.
  - Sounds that play no truth sound: 1.96 a beat in the analysis, 0.71 with a pad in the plan.
  - Most are residual templates on transients of the melodic loop that htdemucs left in the drum stem.
- **A template's own ring is no hit.** That rule could drop a real ghost note that sits at a snare's tail level. The suite has no such case: on 48 beats the rule removed 55 snare-band hits, none of them a true hit of the template's sound.

**808**
- **bass_low's written pitch** is right for 0.487 of its notes.
  - bass_low-1 is written a semitone high on every note: its template settles 59.7 cents sharp, from a drop fitted on 110–220 ms notes.
  - On htdemucs_ft, bass_low's note F1@50 is 0.338 (plan 0.229, 182.6 cents).
- **Legato notes are off by default.** Legato recall is 0; the truth has 24 legato notes on 5 beats.
  - The opt-in slide rule was tuned on the beats it was measured on.
  - Merging every unattacked note into the note before loses a few real notes (tempo160-1 1.000 → 0.946 in the lane).
  - Two 16th notes at one pitch at 160 BPM are split only in part: 2 of the 4 pairs of the unit-test bar still merge.
- **The plan's 808 sits 1.9 dB under a perfect kit** (−1.95 dB; +0.09 dB in m1). Its fader matches the bass stem's whole-track RMS, which now sees the notes' real lengths.
- **The tails of long legato chains are silent.** The one-shot is 3 s long.

**Identification**
- **Library layers** (`matches.json` `layers`) are searched but not played. No layer was given on the 20 beats of the identify lane.
- **Kick sources are off by default.** fit-808 runs after match, and with fit-808's one-shot the rule found 13 of 20 kicks against 12 without it.
- **Wrong files.** 75 of 95 templates have the right file on the identify lane's 8 scenarios. The misses are hat templates that carry the snare on beat 3, and near-copies across kits.

**Melodic and rhythm**
- **Chords follow the sections, which follow the drum hits.** In an intermediate run (m2e), full-2's section A split and majmin fell 0.999 → 0.961 with the same notes.
- **The plan's time.** It holds no meter, and Live's bar 1 is at 0 s.
  - The plan's meter is right on 0.909 of the scenarios, and its downbeat F is 0.892.
  - full's first bar line, at 0.4 s, lands 400 ms off.
- **Bar lines and meters missed.**
  - tempo72's bar line is wrong on both beats (downbeat F 0).
  - meter68 is read in 4/4, at x2/3.
  - The seed-1 swing beats read as straight 8ths.

**Speed and models**
- **analyze went 5.7 → 38.7 s a beat** (2 jobs). The 808 reader takes about 30 s of it by its lane's measure; the rest is the joint fit and the attack-based discovery, not separated. match went 24.4 → 49.1 s.
- **Long input.** In the discovery lane, analyze peaked at 2.27 GB on 192 s of tiled input. That was not measured again after the merge.
- **htdemucs_ft is not the default.** Its drums are better (plan +0.039 [+0.017, +0.062], paired over the 48 beats). But its 808 notes are worse (0.936 → 0.924; bass_low 0.949 → 0.338), and so are its snares (0.961 → 0.891).
- **Songs have no truth.** These numbers come from synthetic beats; PIRAMIT's measurements are in [02](02-case-study-piramit.md).

## Next

In order ([10](10-roadmap.md) has the metrics, targets and gaps):
1. **The plan with Live's measured behaviour:** velocities from Live's law, Snap off, choke groups, pads loaded by path, the meter and bar 1, a lower pad gate.
2. **The build, checked in Live:** build suite plans, record them, and score the recording as the plan is scored.
3. **The closed loop:** measure the recording, correct levels, pitch and timing, rebuild, and stop when the distance stops falling.
4. **Speed.** A DE popsize of 10 fitted the 808 1.9–2.2× faster at equal mean loss on 4 beats. It is not yet run on all 48.
5. **The open limits above.**

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
