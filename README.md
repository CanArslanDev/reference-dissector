# reference-dissector

Take a reference track apart and rebuild it in Ableton Live. The pipeline:

1. finds the tempo grid and the tuning;
2. finds every drum hit and the 808's notes;
3. identifies the exact samples in your library (or the nearest ones);
4. fits a synthesized 808;
5. builds the result in Live, with the original stems muted underneath for A/B, through [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp).

The work started from one question: *can an AI that cannot hear tell whether the sound it placed is the sound in the song?* It can, if listening is turned into measurements it can read (numbers and pictures) and those measurements are calibrated against cases with a known answer. This repository holds the code, the measurements and what was learned on the way, from a Turkish trap track (Mavi – *PIRAMIT*) and a set of synthetic beats built for the purpose.

![PIRAMIT's hat, snare and 808 against their library matches and the synthesized 808](figures/piramit_sounds.png)

## Results at a glance

**PIRAMIT** (126 s, Demucs stems), from `refdissect analyze`, `match` and `fit-808`:

| What | Found |
|---|---|
| Tempo grid | **137.00 BPM**, first beat 31.9 ms in. Every bar's onsets sit within a few ms of it (p90 4.6–11.8 ms). The 136.36 BPM a beat tracker gave drifts by up to 49 ms. |
| Tuning | The melodic stems at **A440** (other −2, piano +1.5 cents). The **808 alone sits at +33 cents**: its sample is detuned, not the track. (An earlier version read +0.28 semitones for everything; that was `librosa.estimate_tuning`'s error, see [01](docs/01-pipeline.md#tuning-gridtuning).) |
| Hi-hat | **The same sample** as a hat in the local library. NCC 0.981; it wins 187 of 189 hits; null-test residual −14.3 dB (next best −7.4 dB). |
| Snare (even bars) | **The same sample**, pitched **−1 semitone**. NCC 0.953; it wins 27 of 27 hits; residual −10.4 dB (next best −6.4 dB). |
| Snare (odd bars) | Not in the library: the best match is NCC 0.47. |
| 808 | 126 notes, 107 of them on C1 (the 808 sits +33 cents sharp). A fitted synth voice reaches NCC 0.975 and a 0.9 dB envelope error, against the best of 865 library 808s at 0.947 and 2.0 dB. The library 808 is closer in the upper harmonics: 2.3 dB off, against 7.7 dB for the synth. |
| Sections | A 1–24 · B 25–28 · C 29–32 · A 33–47 · D 48–51 · E 52–56 (breakdown) · A 57–71 · F 72–73 |
| Live | `refdissect live-build` builds it into an empty Live set with one `build_tracks` call: 6 Drum Rack pads, 549 drum notes, 126 808 notes, reference stems for A/B, 8 locators, in 4.9 s on Live 12.4.2 (9.7 s track by track before). |

**Calibration**: 6 synthetic beats built from known library samples, mastered loud, passed through mp3 and Demucs.

| Sound | Identified at known hit times | Blind transcription F1 (mean) | Identified blind |
|---|---|---|---|
| Snare | 6/6 | **1.00** | 6/6 |
| Open hat | 6/6 | 0.83 | 5/5 |
| Hat | 5/6 | 0.77 | 7/10 templates |
| Kick | 4/6 | 0.70 | 5/10 templates |
| 808 | 6/6 (865 candidates, ±12 semitones) | — | — |

At known hit times, 27 of 30 sounds are found among 8,854 one-shots. Details are in [docs/03-calibration.md](docs/03-calibration.md).

## How it works

```
song.mp3 ──Demucs──▶ stems ──▶ grid + tuning ──▶ drum templates ──▶ hits (matched filter)
                                 │                      │
                                 │                      └──▶ library match ──▶ verdicts
                                 └──▶ 808 notes ──▶ 808 fit (synth) ──┐
                                                                       ▼
                         plan.json (notes in beats, pads, levels, locators) ──▶ Live (ableton-live-mcp)
```

- **Grid**: a comb search over tempo and phase on the onset envelope, coarse then fine. Half and double time are settled by `--bpm-range`.
- **Drums**: in each band (low, snare, hat), strong onsets on the sixteenth grid are grouped per slot by waveform and averaged into templates. Averaging N hits lowers whatever plays along by about √N.
  - Each template then slides over the whole stem as a matched filter.
  - Templates that are the same sound are merged. That needs both the same moments and the same waveform in both bands, because a snare that always lands on a hat looks like the hat in the hat band.
- **Identification**: normalized cross-correlation (NCC) against every library one-shot, in three bands, over ±30 ms of lag.
  - A pitch-invariant shortlist, made from semitone-band spectra, proposes transpositions: a snare pitched −1 semitone correlates 0.15 at 0 and 0.95 at −1.
  - Identities are handed out from the strongest match down, so a hat that always plays along with a kick is not taken for the kick.
  - The verdict uses NCC, the margin over the next *different* sound, per-hit votes and a null test.
- **808**: notes come from the bass stem, with the track's tuning removed before rounding. A sine + glide + envelope + tanh-drive voice is fitted by differential evolution. It is scored on waveform, harmonics and envelope, not on correlation alone.
- **Live**: [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp)'s `build_tracks` builds the whole set from one spec. Our tracks are matched to the stems' levels and the stems are muted for A/B.

Each step is described in [docs/01-pipeline.md](docs/01-pipeline.md).

## Quick start

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). For `separate`, Demucs is installed by uv in its own environment on first use (a few hundred MB). For `live-build`, Ableton Live must be running with [ableton-live-mcp's Remote Script](https://github.com/CanArslanDev/ableton-live-mcp#install).

```bash
git clone https://github.com/CanArslanDev/reference-dissector && cd reference-dissector
uv sync                                   # add --extra live for live-build

W=~/Music/"Reference Dissect"
# your sample library, once
uv run refdissect inventory "/Applications/Ableton Live 12 Suite.app/Contents/App-Resources/Core Library/Samples" \
    ~/Music/Ableton/"User Library" --out "$W/work/inventory.json"
uv run refdissect index-library "$W/work/inventory.json" --out "$W/work/library.npz"
uv run refdissect index-library "$W/work/inventory.json" --808 --out "$W/work/library_808.npz"

# a song
uv run refdissect separate song.mp3 --out "$W/song/stems"
uv run refdissect analyze "$W/song/stems" --work "$W/song/refdissect"
uv run refdissect match "$W/song/refdissect" --library "$W/work/library.npz"
uv run refdissect fit-808 "$W/song/refdissect" --library-808 "$W/work/library_808.npz"
uv run refdissect plan "$W/song/refdissect"            # plan.json + midi/drums.mid, midi/808.mid
uv run --extra live refdissect live-build "$W/song/refdissect"   # into an empty Live set

# comparing and choosing (docs/11)
uv run refdissect separate rebuild.wav --out "$W/song/rebuild_stems"      # the same model as the reference
uv run refdissect compare "$W/song/stems" "$W/song/rebuild_stems" --out "$W/song/compare.json"
uv run refdissect listening-round ref.wav cand1.wav cand2.wav --question "Which chords?" --out "$W/song/round1"
uv run refdissect space "$W/song/stems/other.wav" --spans 40-56 100-116
uv run refdissect mix-gains "$W/song/recorded" --targets targets.json

# calibration: synthetic beats from your own library, scored
uv run refdissect calibrate "$W/work/inventory.json" --library "$W/work/library.npz" \
    --library-808 "$W/work/library_808.npz" --beats 6 --out "$W/work/calibration"
```

Every step writes JSON next to its inputs, so you can read what it decided before the next step uses it. `scripts/export_results.py` turns a work folder into the shareable `results/` and `figures/`: no audio, and no file names from a personal library.

## Documentation

| | |
|---|---|
| [01 – Pipeline](docs/01-pipeline.md) | Every step, its parameters and why they are what they are |
| [02 – Case study: PIRAMIT](docs/02-case-study-piramit.md) | Everything measured on the track, including what is still wrong |
| [03 – Calibration](docs/03-calibration.md) | Synthetic beats with a known answer: how identification and transcription score |
| [04 – Identifying samples](docs/04-identifying-samples.md) | What "the same sound" means, and how NCC, margin, votes and null tests decide it |
| [05 – Synthesizing the 808](docs/05-808-synthesis.md) | The voice model, the fit, and why one similarity number is not enough |
| [06 – Listening for a model](docs/06-listening-for-a-model.md) | How a model that cannot hear compares sounds: numbers, pictures, calibrated verdicts |
| [07 – Building in Live](docs/07-building-in-live.md) | The plan, the ableton-live-mcp tools used, and Live behaviour measured on the way |
| [08 – Pitfalls](docs/08-pitfalls.md) | Every mistake made here and how it showed up |
| [09 – Data and copyright](docs/09-data-and-copyright.md) | What is published, what stays local, and why |
| [10 – Roadmap](docs/10-roadmap.md) | What comes next |
| [11 – A new beat in a song's style](docs/11-style-beat-lessons.md) | Two rejected beats: the groove, effect and same-pipeline checks they led to, and what each step cost |

Türkçe özet: [README.tr.md](README.tr.md)

## Layout

```
src/refdissect/
  audio.py       read, bands, transposition (as Simpler does it), onset trimming
  xcorr.py       normalized cross-correlation and the matched filter (lag convention tested)
  hits.py        onsets, grouping hits by waveform, averaging into templates
  grid.py        tempo/phase fit, per-beat offsets, tuning
  detect.py      template discovery, matched-filter detection, 808 notes
  layers.py      every drum moment fit with every template: layers, flams, rolls, pitched copies
  library.py     inventory, index, matching (NCC + semitone shortlist), votes, null test, verdicts
  describe.py    producer-facing descriptors: pitch/cents, glide, decay, harmonics, brightness
  synth808.py    the 808 voice, its fit, one-shots
  structure.py   per-bar activity and sections
  groove.py      ducking under the kick, drum-role balance, one-shots cut by their notes
  space.py       width per octave, reverb tails, the band-pass between two sections
  rhythm.py      a layer's rhythm from onsets (48 points per bar)
  basicpitch.py  posteriorgram frames by time
  loudness.py    BS.1770 integrated loudness
  listening.py   A/B files: reference, beep, candidate at equal loudness
  mixsim.py      gains decided on recorded stems instead of re-recording
  symmetric.py   a rebuild and its reference measured the same way, against the reference's own spread
  rounds.py      listening rounds: numbered reference-beep-candidate files and a manifest
  live.py        plan, MIDI files, the build through ableton-live-mcp
  calibrate.py   synthetic beats and their scoring
  pipeline.py    the steps, each writing JSON
  cli.py         refdissect
results/         measured numbers (calibration, PIRAMIT) - no audio
figures/         spectrograms and the calibration plot
scripts/         export_results.py
tests/           synthetic drums: lag convention, grouping, grid, matching, 808 fit, plan
```

## Status

This is a working pipeline with measured limits, not a finished product:
- Kick and hat transcription is the weak part (F1 0.70 and 0.77 blind).
- Melodies are not transcribed yet: the original melodic stems play under the rebuilt beat.
- The 808's slides are reported but not played.

See the [roadmap](docs/10-roadmap.md).

## License

MIT for the code (see [LICENSE](LICENSE)). The songs, their stems and any samples are not part of this repository and keep their own rights: see [docs/09-data-and-copyright.md](docs/09-data-and-copyright.md).
