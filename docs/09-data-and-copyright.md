# 09 – Data and copyright

This project analyses commercial recordings and uses sample libraries. What it publishes is chosen so that nothing protected leaves your machine.

## Published here

- **The code**: MIT.
- **Measurements:**
  - tempo, tuning, sections;
  - per-sound statistics (counts, grid slots, descriptors);
  - match scores and verdicts;
  - the fitted 808's parameters;
  - calibration scores.

  Facts about a recording, not the recording.
- **Figures:** spectrograms of short averaged hits and of library sounds. Analysis images: low resolution, magnitude only, a fraction of a second each.

## Kept local (git-ignored, written to your work folder)

- **The songs and their stems.**
- **The templates:** averaged hits cut from the stems, as `.npy` and `.wav`. They are close copies of the song's sounds, so they are for **personal study only**. Do not publish them, and do not use them in work you release.
- **Note-level transcriptions** (`analysis.json` events, `plan.json`, `midi/`). A transcription of a song's drum pattern and bassline is a derivative of the composition.
- **Your library:** inventory, index (`library.npz` holds audio), and file and kit names. `scripts/export_results.py` replaces names from a personal library with role + hash. Ableton's Core Library keeps its names.
- **The synthesized 808 one-shot.** It is ours, but fitted to a recording; the parameters are published, and the sound can be re-rendered from them.

## Using the results

| route | what it is | for release? |
|---|---|---|
| (a) hits cut from the stems | the song's own sound | study only |
| (b) library samples, matched | depends on the sample's license | check the kit's license first: some kits, especially "leaked" copies of paid ones, have none you can rely on |
| (c) sounds synthesized in Live | yours | yes |
| generated with MusicGen (ableton-live-mcp `generate_audio`) | model weights CC-BY-NC 4.0 | non-commercial only |

**The arrangement and the patterns.** A drum pattern alone is generally not protected. A melody, a hook or a recognizable combination is. A "replica" is a study tool; a release should take the *style*, not the song: different melody, your own sounds. That is the "style mode" on the [roadmap](10-roadmap.md).

## The calibration beats

They are made from your own library and stay local. Only their scores are published (`results/calibration/summary.json`), and those name no files.
