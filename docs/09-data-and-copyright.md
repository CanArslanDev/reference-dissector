# 09 – Data and copyright

This project analyses commercial recordings and uses sample libraries. What it publishes is chosen so that nothing protected leaves your machine.

## Published here

- **The code**: MIT.
- **Measurements:**
  - tempo, meter, tuning, sections;
  - per-sound statistics (counts, grid slots, descriptors);
  - match scores and verdicts;
  - the fitted 808's parameters;
  - calibration scores.

  Facts about a recording, not the recording.
- **Figures:** spectrograms of short averaged hits and of library sounds. Analysis images: low resolution, magnitude only, a fraction of a second each.

`scripts/export_results.py` writes `results/` and `figures/` from a work folder:
- `results/piramit/summary.json`:
  - the grid without its beat times (`tempo_map` stays local), the tuning and the sections;
  - per sound: counts, slots, descriptors and its top three matches;
  - the 808: note counts per pitch and the synth's parameters;
  - the plan: pads, levels and section locators.
  - No event time and no note.
- `results/calibration/summary.json`: the six v1 calibration beats' scores.
- `results/calibration/suite-<tag>.json`: one calibration-v2 run's means per scenario and overall, and how the run was made.

## Kept local (git-ignored, written to your work folder)

- **The songs and their stems.**
- **The templates:** averaged hits cut from the stems, as `.npy` and `.wav`. They are close copies of the song's sounds, so they are for **personal study only**. Do not publish them, and do not use them in work you release.
- **Note-level transcriptions:**
  - `analysis.json`: every drum hit, the 808 notes, what plays in each bar, and the beat times where they drift (`tempo_map`);
  - `bass808.json`: the 808 notes, the template's note onsets and its pitch curve;
  - `melodic.json` and `basicpitch/`: notes, chords, posteriorgrams;
  - `plan.json` and `midi/`.

  A transcription of a song's drum pattern and bassline is a derivative of the composition.
- **What the analysis decided per sound:** `analysis.json`'s `layers` (the joint fit's merges, held sounds, usual gains) and `matches.json` (library paths, transpositions, layers, per-hit votes, null tests). They name your files. The export keeps the scores and verdicts, with names replaced as below.
- **Your library:**
  - the inventory, the index (`library.npz` holds audio) and its speckle index (`library_speckle.npz`);
  - file and kit names. `scripts/export_results.py` replaces names from a personal library with role + hash. Ableton's Core Library keeps its names.
- **The synthesized 808 one-shot** (`808 (synth).wav`, and the copy `live-build` puts in your User Library). It is ours, but fitted to a recording; the parameters are published, and the sound can be re-rendered from them.
- **Rendered plans** (`render.render_plan`): the song's pattern played with your samples. The suite renders each plan for its level scores and deletes the render.

## Using the results

| route | what it is | for release? |
|---|---|---|
| (a) hits cut from the stems | the song's own sound | study only |
| (b) library samples, matched | depends on the sample's license | check the kit's license first: some kits, especially "leaked" copies of paid ones, have none you can rely on |
| (c) sounds synthesized in Live | yours | yes |
| generated with MusicGen (ableton-live-mcp `generate_audio`) | model weights CC-BY-NC 4.0 | non-commercial only |

**The arrangement and the patterns.** A drum pattern alone is generally not protected. A melody, a hook or a recognizable combination is. A "replica" is a study tool; a release should take the *style*, not the song: different melody, your own sounds. That is the "style mode" on the [roadmap](10-roadmap.md).

## The calibration beats

They are made from your own library and stay local.
- **v1** (`refdissect calibrate`, six beats): only their scores are published (`results/calibration/summary.json`), and those name no files.
- **v2** (`refdissect suite`, 48 beats). A beat keeps everything that makes it a test:
  - its mixes and truth stems, and `events.json`, every hit and note with its time;
  - its Demucs stems per model;
  - each run's outputs and scores (`runs/<tag>/`, `scores-<tag>.json`), which name files and hold note times.

  `scripts/export_results.py --suite DIR TAG` publishes `results/calibration/suite-<tag>.json`. It holds the means per scenario and overall, and the run's model, seed, versions, options and steps. The export refuses a run description that is not numbers or short identifiers. It also refuses an output that looks like a local path or an audio, MIDI, array or Live file name.
