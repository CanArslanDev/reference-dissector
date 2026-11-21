# 13 – A song like this one: where the melodic layer comes from, and how to make it similar

Asked for "a song like this one", the tools give drums and an 808 that come close and a melodic layer that sounds unrelated to the reference: notes on a stock piano, pad or synth (on the replica path the notes are the reference's own transcription, played on Drift). This page traces where that layer's sound and notes come from on each of the three paths, measures what would choose a closer sound, and proposes the order of work. It is a research note: no code changes with it.

As in [09](09-data-and-copyright.md), no audio, stems, transcriptions, library file names or preset names are published. The numbers come from local runs. Code is cited at reference-dissector `8484988` (rd) and ableton-live-mcp `d6f28e5` (alm).

## Which path the earlier attempts took

The job records hold only two PIRAMIT dissects (2026-11-16), the replica path. The study folders show the rest:

| When | Reference | Path | Melodic sound | Verdict |
|---|---|---|---|---|
| 2026-10-11/12 | PIRAMIT | 3: the assistant's own scripts over alm's tools | drafts 1–3: Live presets chosen by name (a reed lead, a muted guitar, a sitar, a pad), then CLAP against another song's stem; draft 4: riff and pulse on library sounds; draft 5: each of PIRAMIT's melodic stems' CLAP first; draft 6: notes cut from PIRAMIT's own stems into a Drum Rack | 1–5 rejected; 6 the best so far |
| 2026-10-15/17 | HAYIR DEMELİYDİM | 3, with rd's modules ([11](11-style-beat-lessons.md)) | sounds chosen by ear in A/B files from a CLAP + band + envelope shortlist | v1 and v2 rejected: v1 for its drums and mix, v2 for its parts; the sounds had been picked by ear |
| 2026-10-17/18 | a third song | replica-style through the assistant's own scripts (the song's own melodic stems), then 3 | the song's own stems, which the user heard as the original's melody; then library sounds by CLAP and ear | kept after picks by ear |
| 2026-11-08 | four songs | 3 | Live presets: piano, then a sitar, then plucks; then, after separating again, a CLAP shortlist over preset previews and library one-shots, played in Live on the reference's phrase with its measured reverb (strings, pads, a synth) | no draft picked: the user heard no piano or synth in any of the four songs |

Path 2 (`refdissect candidates`) has never been heard. Every "song like this one" went through path 3 at least in part. No step of the tools ties the sound to the reference there; where a sound was tied to it (CLAP against a stem, notes cut from a stem), the assistant's own scripts did it.

## 1. Where the melodic sound and notes come from

### Path 1: replica (`dissect_reference` → `build_from_blueprint`)

- **Notes.** Basic Pitch per tonal stem (`melodic`), one MIDI track per stem in the plan (rd `src/refdissect/live.py:644-647`).
- **Sound.** Every such track gets `MELODIC_INSTRUMENT = "Drift"` (`live.py:116-118`, "a placeholder … until a sound is chosen per stem") and is built muted (`live.py:1516-1519`). alm inserts a device named by a string with its default preset (alm `MCP_Server/extended_tools.py:3047-3050`): Drift as its default preset in the Core Library sets it (read from the file, not checked in Live): a saw plus a sine an octave down, the filter open, envelope A 1 ms, D 600 ms, S 0.7, R 600 ms.
- **What is heard.** The reference's own other, piano and guitar stems: `STEMS_MUTED` holds only drums, bass and vocals (`live.py:108`, `:635-638`). Mute them and unmute the melodic tracks, and Drift's default patch is what plays.
- **Effects.** None reach a melodic track. alm's `build_tracks` takes an `effects` list per track (`extended_tools.py:2906-2907`); rd's blueprint writes none.
- **Step that does not look at the reference:** `live.py:118`, one fixed name for every stem of every song.

### Path 2: new beat (`refdissect candidates`)

- **Notes.** Drums, 808 and chords drawn from the plan's statistics (`candidates.py:162`, `style_stats`; `:382`, `_draw`; the chords `:287` and `:430`). No riff, pulse or melody.
- **Sound.** The chords render on a plain additive pad (`candidates.py:617-641`: six harmonics falling as 1/h^1.5, two voices 6 cents apart; "a sketch, not a sound choice", `:28-29`) and go to Live on Drift (`:606`).
- **Step that does not look at the reference:** both, by design: the sketch was meant for the harmony only.

### Path 3: a free request in alm

- **Prompt.** The only prompt that builds from a reference is `beat_like_reference` (alm `MCP_Server/prompts.py:166-197`), which is path 1; `match_reference` (`:77-91`) moves a mix's tonal balance and loudness toward a reference.
- **Sound.** The compose skill offers `browser_search(query="warm pad")` and `add_device(device="Drift")` (alm `skills/compose-in-live/SKILL.md:58-61`). `find_similar_sounds` is listed after them (`:62-64`), but no step asks for it against the reference's stem. Its default folders are the Core Library's one-shots and the User Library's samples (`extended_tools.py:2606-2610`): no presets.
- **Notes.** `generate_chord_progression` (`extended_tools.py:1817`) and `generate_melody` (`:2087`, "a phrase-structured melody (A A' B A'')") follow key, scale and chord rules. `generate_melody` writes a lead, the kind of part the user rejected three times on PIRAMIT.
- **Steps that do not look at the reference:** the name search, the stock device, both generators, and the absence of any step that reads what kind of part the reference's layer is.

## 2. Choosing a sound against the layer: measured (2026-11-21)

### Setup

- **Targets.** 8 bars each:
  - PIRAMIT's other stem: htdemucs, from the 2026-11-16 dissect job; bars 13–20, the riff; 73 Basic Pitch notes.
  - The suite's `melodic-1`: its three truth layers, clean, on their truth notes:
    - a pad, two saws;
    - a pluck, a square;
    - a lead, a saw.
  - The same beat's htdemucs other stem, which holds all three layers, on its 205 Basic Pitch notes.
  - `v1-1`'s other stem: its melodic layer is a recorded-music cut from the library, tiled at its own length.
- **Pool.**
  - 1,855 Live presets, each recorded in Live playing MIDI 60 (the HAYIR DEMELİYDİM study's probe, 2026-10-16).
  - 2,141 melodic files of the library: one-shots, Core Library multisamples and loops of 6 s or less, with a root pitch estimated per file in an earlier local study. The one-shot folders also hold long loops and cuts: 275 files over 6 s, up to 105 s.
  - "Instruments only" leaves out the files tagged as loops and anything over 8 s: 2,644 sounds.
- **Stock controls.**
  - Drift as `add_device` loads it: an offline stand-in built from its default preset's parameters.
  - The two Core Library instruments `browser_search("warm pad")` returns: an Instrument Rack and a Simpler preset, from their Live recordings.
  - `candidates`' additive pad.
- **Stage 1** is what `find_similar_sounds(like=stem)` does: CLAP (`laion/clap-htsat-unfused`) of each file's first 10 s against the target's (here the mean of the 8 bars' first and last 10 s).
- **Stage 2** takes stage 1's top 30, the stock sounds, the truth sounds where known, and a fitted voice (way B below).
  - Each one plays the target's own notes. Library files and preset recordings go through a Simpler-like sampler: repitched, cut at note-off plus 80 ms, at Live's velocity law. The stock synths play the notes directly.
  - Each goes through a reverb fitted to the target. A grid of decay time and wet share is fitted on the L/R correlation in three octave bands and on the level ringing 60–250 ms after note-offs: PIRAMIT 1.5 s at 75 % wet, the grid's wettest, so the stem may be wetter still; the suite's layers are dry and mono.
  - Each is matched to the target's RMS.
  - Two measures:
    - CLAP between the render and the target;
    - a phase-free descriptor distance. It takes the medians over notes of harmonics 2–8 re the first, centroid over f0, attack, the drop 150 ms after the peak and flatness, each in units of the target's own spread (its first half of notes against its second, floored), averaged.

### Stage 1: CLAP against whole files

| Target | Pool | CLAP top / median | Truth sound's rank | "warm pad" ranks | Top 10 |
|---|---|---|---|---|---|
| PIRAMIT other | all 3,996 | 0.62 / −0.02 | – | 1,429 and 1,721 | 10 loops, 8 of them 59–105 s long |
| PIRAMIT other | instruments 2,644 | 0.40 / −0.03 | – | 832 and 1,024 | one-shots of 2–8 s |
| melodic-1 pluck | all / instruments | 0.59 / 0.17 (all) | 48 / 38 | 173, 916 / 141, 761 | 7 / 6 share the target's AST family ("Synthesizer") |
| melodic-1 lead | all / instruments | 0.68 / 0.24 (all) | 13 / 11 | 400, 784 / 318, 628 | 2 / 3 share its family ("Synthesizer") |
| melodic-1 pad | all / instruments | 0.66 / 0.26 (all) | 54 / 38 | 298, 512 / 242, 418 | 3 / 6 share its family ("Keyboard") |
| melodic-1 other (3 layers) | all 3,996 | 0.73 / 0.09 | pad 516, pluck 1,527, lead 278 | 670 and 1,774 | 10 loops, 8 of them 64–105 s |
| v1-1 other (a sampled loop) | 1,698 loops | 0.29 / 0.05 | the true loop file: 1,157 | – | |

- Against a separated stem, CLAP's nearest whole files are long loops of a minute or more, as recorded-music cuts were in the 2026-11-08 study. Only a clean, single layer brings instruments of its AST family into the top 10: 6–7 of 10 on the pluck, 3–6 on the pad, 2–3 on the lead.
- A shortlist of 30 from stage 1 would miss the pluck's and the pad's true sounds (38th each). Stage 1's order barely predicts stage 2's: Spearman −0.17 to 0.25 across the suite's targets, and −0.11 on PIRAMIT's instrument pool. The 0.61 on PIRAMIT's whole pool is the long loops scoring high in both stages.
- **The sampled loop is found by neither model.** In `v1-1`, CLAP puts the true file 1,157th of 1,698 loops. AST hears the separated stem as "Music, Tick-tock, Tick, Shuffle" (drum bleed) and the file itself as "Music, Rapping, Hip hop music": Demucs has moved the cut's rap and drums to other stems.

### Stage 2: the same notes, the same space

CLAP (render against target) / descriptor distance (lower is closer); "instruments" pool:

| | PIRAMIT other | melodic-1 pluck | melodic-1 lead | melodic-1 pad | melodic-1 other |
|---|---|---|---|---|---|
| the reference against itself (ceiling): PIRAMIT's bars 13–20 against four other 8-bar spans; the suite's layers, bars 1–8 against 9–16 | 0.83–0.94 / 0.71–4.08 | 0.978 | 0.951 | 0.997 | 0.974 |
| truth sound | – | **0.985 / 0.63** | **0.956 / 0.44** | **0.943 / 1.06** | lead 0.856 / 1.69; pad 0.785 / 2.92; pluck 0.765 / 14.5 |
| fitted voice (way B) | 0.533 / **0.95** | 0.908 / 1.91 | 0.830 / 1.13 | 0.918 / 3.00 | – |
| pool, best by CLAP | **0.647** / 4.83 | 0.884 / 14.1 | 0.910 / 1.35 | 0.931 / 4.42 | 0.922 / 4.11 |
| pool, best by descriptors | 0.396 / 1.24 | 0.767 / 12.7 | 0.910 / 1.35 | 0.786 / 3.21 | 0.693 / 1.66 |
| Drift's default (stock) | 0.513 / 1.74 | 0.838 / 17.1 | 0.803 / 1.10 | 0.806 / 3.82 | 0.850 / 2.02 |
| "warm pad" hits (stock) | 0.203 / 7.82; 0.296 / 4.76 | 0.723 / 14.1; 0.765 / 24.8 | 0.680 / 16.5; 0.713 / 5.81 | 0.760 / 13.9; 0.824 / 7.13 | 0.553 / 13.9; 0.600 / 6.00 |
| `candidates`' additive pad | 0.420 / 6.21 | 0.739 / 13.7 | 0.815 / 8.72 | 0.865 / 11.3 | 0.735 / 10.1 |
| Spearman, CLAP vs descriptors over the pool's 30 | 0.21 | 0.52 | 0.51 | −0.05 | 0.26 |

- **On a clean layer with the true sound among the candidates, both measures put it first** (pluck, lead, pad). The fitted voice comes next on the pluck by both measures and on the pad by the descriptors. On the lead, by the descriptors, Drift's default saw comes between them; by CLAP the fitted voice is 10th and Drift's default 17th.
- **On PIRAMIT the two measures disagree** (Spearman 0.21; 0.06 over the 30 shortlisted from the whole pool). CLAP's first is a keys preset at 0.647, the descriptors' first the fitted voice at 0.95. No candidate reaches the song's own spread on CLAP (0.83–0.94). The fitted voice, the best library one-shot and Drift's default all fall inside it on the descriptors (0.71–4.08), so on this stem that measure is too coarse to choose with.
- **Against the stock flow, by the numbers.** On PIRAMIT the shortlist's best by CLAP beats the "warm pad" hits by 0.35–0.44 and Drift's default by 0.13. It was picked by the same CLAP it is scored on, so the margin is biased upward. By the descriptors it is 20th of 35 (4.83, outside the song's own 0.71–4.08), and Drift's default is 3rd (1.74). The "warm pad" hits are the last two by CLAP (0.20–0.30), 19th and 30th by the descriptors. On the suite, whose voices are plain saws and squares, Drift's default saw is close by construction: 0.80–0.85 CLAP. That puts it ahead of most of the shortlist on the pluck (24 of 30) and the three-layer stem (27), about midway on the lead (17) and behind two thirds of it on the pad (9). The suite cannot show what a stock synth costs on a real song; here the real-song evidence is one 8-bar excerpt of PIRAMIT.
- **Space first.** Through the fitted reverb, CLAP on PIRAMIT moved by −0.12 to +0.21 against the same renders dry. The median was +0.06, and 24 of the 35 rose. The stock sounds and the shortlist's first all rose: the best preset 0.473 → 0.647, the fitted voice 0.395 → 0.533, Drift's default 0.309 → 0.513. The order changed too (Spearman 0.80, dry against wet): two of the dry top 3 are not in the wet top 3. Ranking dry candidates against a wet stem picks other sounds, as in the HAYIR DEMELİYDİM study ([11](11-style-beat-lessons.md#2-effects-before-sounds-space)).
- **Separated against separated is not fair either.** Each PIRAMIT finalist was mixed with PIRAMIT's own drums, bass and vocals stems and separated by the same htdemucs.
  - The candidates' scores went from 0.20–0.65 to 0.72–0.93 and reordered (Spearman 0.27 against the clean scores).
  - `candidates`' plain additive pad came first (0.926).
  - The reference re-separated scores 0.985. Those three stems alone, separated, leave an other stem 26 dB under the target that scores 0.42.
  - The separation's shared texture lifts every candidate, most of all those that scored lowest clean (a "warm pad" hit +0.52, the additive pad +0.51). A symmetric test needs the candidate inside a beat of its own, not inside the reference's stems.
- **Live's preset previews can shortlist presets without Live.** CLAP of the 1,853 previews Live ships took 57 s, against 35 minutes to record 1,856 presets in Live ([11](11-style-beat-lessons.md#what-it-cost)).
  - A preview against its Live recording: cosine 0.845 (median).
  - Ranked against four targets (PIRAMIT, the pluck, the three-layer stem, `v1-1`'s loop stem), the two orders agree with Spearman 0.55–0.73, lowest on the loop stem. Their top 10s share 3 to 6 presets.
  - The recordings' first preset is the previews' 22nd on PIRAMIT and 90th on the loop stem; the recordings' top 10 sit at a median preview rank of 5 to 27.
  - So previews shortlist, and the shortlist is recorded in Live (30 presets: about a minute).

### What the user's picks say about the measures

Three listening tests of the HAYIR DEMELİYDİM study have a pick by ear among three candidates: test 4's chords, test 4's arp and test 2's riff (whose file also held the stem's own notes as a control, left out here). The candidates were measured by CLAP against the reference half of the very files the user heard:

| Test | Pick's rank by CLAP | by long-term log-mel spectrum | by L/R correlation |
|---|---|---|---|
| chords (test 4, with the fitted space) | 1 of 3 | 2 | 1 |
| arp (test 4, with the fitted space) | 3 of 3 | 3 | 2 |
| riff (test 2, before the space was fitted) | 3 of 3 | 2 | 1 (all mono) |

- CLAP put the pick first once in three, which is chance.
- The study's own scores put the picks first twice in three (chords, riff); the arp's pick was second. For the chords and the riff that score was the mean of CLAP and band, envelope, centroid and texture similarities on its own renders; for the arp, a score of the partials at C4.
- Three picks are not enough to choose a measure, nor to tell CLAP from chance. They give no reason to let CLAP choose alone.

## 3. The options

What each way of finding a sound needs, costs and gave.

The effects are measured per layer. `space` gives width per octave and the band-pass between two spans; the reverb's decay and wet share are this note's own fit (section 2) and are not in rd yet. Each way carries them to Live the same way, as the layer track's `effects` in the blueprint, which alm's `build_tracks` accepts: Reverb (Decay Time, Dry/Wet), Utility (Width), EQ Eight (high- and low-pass). Two gaps remain:
- one Utility Width cannot carry a width per octave;
- the fit's wet share mixes the dry signal with a reverb at the same RMS, which is not Live Reverb's Dry/Wet. The mapping has to be calibrated by recording Live's Reverb, as the HAYIR DEMELİYDİM study did for its chords and arp.

Ways A and B are ranked through that space; C has it baked in.

| Approach | What it needs | Cost | Expected quality (measured above) | Risk |
|---|---|---|---|---|
| **A. Search the library and Live's presets by embedding, then re-rank on the layer's notes** | CLAP (615 MB, Apache-2.0); AST (346 MB, BSD-3) for the family; an index; the layer's notes and space | Index: 3,996 files with CLAP and AST in 615 s, once. Presets from Live's previews in 57 s, not 35 minutes of recording. Per layer: 10–45 s for 35 renders and their vectors. | PIRAMIT by CLAP: 0.647 against Drift's default 0.513 and the "warm pad" hits 0.20–0.30, ceiling 0.83–0.94; picked by the same CLAP, so biased upward. By the descriptors the same preset is 20th of 35 and Drift's default 3rd. Suite lead: 0.910 against the truth's 0.956. | Stage 1 alone misleads: long recordings on mixed stems, and the right sound at rank 38. CLAP and the descriptors disagree on real stems. A one-note recording repitched is not the preset over its whole range. The pool may lack the family (2026-11-08: no piano or synth in the songs, mostly presets in the pool). |
| **B. Fit a voice to the layer (synthesizer sound matching)** | A numpy voice with Drift-like controls (here 13: one harmonic shape played by two detuned copies, a sub, a low-pass with envelope and resonance, ADSR, noise), a loss, an optimiser. Rendered per pitch into a multisample for Sampler, as `fit-808` renders a one-shot for the 808; mapping it onto Drift, Operator or Wavetable needs a check in Live, which renders instruments only in real time (1.12 s a render). | 6–11 minutes a layer: 2,704 evaluations, one core, differential evolution | PIRAMIT: first by the descriptors (0.95, inside the song's own 0.71–4.08), 0.533 CLAP. Suite pluck: 0.908, second only to the truth. | It is fitted on the descriptor loss, which therefore flatters it; CLAP is the independent check. Subtractive timbres only: no piano, strings or recorded sounds. The earlier local study found no 2024–2026 work that fits Live's own devices; the learned matchers that work need 3·10⁴–2·10⁶ renders of one synth, and Live renders in real time. |
| **C. A multisample from the stem's own notes** | Basic Pitch notes, each pitch's median over its hits, laid out as Simpler or Sampler zones | Minutes | PIRAMIT draft 6 (2026-10-12): the riff 0.759 CLAP against the song's own sections' 0.762; its keys 0.495 | **It is the song's own recording: personal study only, never in a release or a public repository** ([09](09-data-and-copyright.md)). Bleed and reverb come with it. Pitches the song never plays are repitched. Chords and pads do not cut into notes. A song that sounds "like" its reference by being its sound. |
| **D. Generative models** | MusicGen-melody (1.5 B parameters, 6.2 GB, weights CC-BY-NC 4.0); MusicGen small (2.4 GB, CC-BY-NC 4.0, which alm already uses: an 8 s loop in ~19 s on Apple Silicon); DDSP timbre transfer (Apache-2.0, TensorFlow, pretrained on four acoustic instruments); RAVE (CC BY-NC 4.0, trained per timbre) | Downloads of 2.4–6.2 GB; RAVE and DDSP need hours of training per timbre | Not measured | MusicGen's and RAVE's weights are non-commercial. DDSP is Apache-2.0 but runs on TensorFlow, has four acoustic instruments and needs hours of training for any other. Melody conditioning feeds the reference's chroma, so it copies the notes. No hold on the exact timbre. A model trained on the reference's stem copies it. |


## 4. Reading what kind of part a layer is

The ear judged the kind of part first: PIRAMIT has a three-pitch riff in 16ths and no lead, and its three drafts with a lead were rejected (the history above). Basic Pitch's fine statistics on a reverb-heavy layer measure the transcription, not the music; only coarse ones survived ([11](11-style-beat-lessons.md#3-statistics-have-to-survive-the-same-pipeline-rhythm)). These coarse statistics were read over 8 bars:
- notes sounding at once (median over sounding 16ths);
- median note length in 16ths;
- distinct onset 16ths per bar;
- pitch classes per bar;
- range (5th to 95th percentile).

A first rule over them, its thresholds set on the same truth layers it is shown on below (in-sample, not a test):
- chord pad: 2.5 or more notes at once and 6 or more 16ths long;
- else, 8 or more onsets a bar with fewer than 2.5 notes at once: pulse/arp with 3 or more pitch classes a bar, riff with fewer;
- else, 3 or fewer pitch classes a bar and a median length of 3 16ths or less: riff;
- lead: anything else.

| Layer (bars 5–12 of melodic-1, PIRAMIT 13–20) | at once | length (16ths) | onsets/bar | pitch classes/bar | range (st) | rule says | is |
|---|---|---|---|---|---|---|---|
| melodic-1 pad, truth notes | 5 | 6.0 | 2.2 | 4.5 | 26 | chord pad | chord pad |
| melodic-1 pluck, truth notes | 1 | 0.85 | 12.8 | 5.5 | 23 | pulse/arp | arpeggiated pluck |
| melodic-1 lead, truth notes | 1 | 2.0 | 5.8 | 2 | 10 | riff | lead |
| melodic-1 other stem, Basic Pitch, all notes | 4 | 1.9 | 11.9 | 6 | 32 | lead | three layers |
| the same, split in 3 by register: low / mid / high | 2 / 2 / 1 | 2.2 / 2.0 / 0.8 | 5.2 / 8.9 / 5.2 | 3 / 4.5 / 3.5 | 13 / 10 / 11 | riff / pulse / lead | all three / pluck and lead / mostly pluck |
| PIRAMIT other stem, Basic Pitch | 2 | 2.5 | 6.4 | 3 | 24 | riff | riff (and some keys) |

- On truth notes the rule reads the pad and the pluck, as it was set to. It calls the suite's lead a riff: the suite's lead plays 2 pitch classes a bar, and nothing in these numbers tells a sparse lead from a riff. What would is the period: a riff repeats every one or two bars, a lead phrase does not. The waveform repeat below measures that, but the stem's other content gets into it.
- On the separated stem the layers are not separable this way. The suite's pad plays five notes at once for six 16ths in the truth. The Basic Pitch notes on the other stem that match it read one note at once for 1.9 16ths, and only about half of its 80 truth notes are found (37–51, depending on how a note two layers share is counted): its chords are cut into fragments. Each register cluster holds notes of all three layers, and the highest is mostly the pluck's top notes, not the lead. So every reading of a layer's kind needs its notes from that layer alone, and Demucs gives one stem for all of them. **A layer split inside the other stem is missing**, and it comes before any choice of kind or sound. The HAYIR DEMELİYDİM study split its layers by hand, with its own scripts.
- PIRAMIT reads as a riff: three pitch classes a bar, the 1, 5 and ♭6 of its key.

**Writing new notes of that kind.** `candidates` already draws drums and 808 from per-16th shares and keeps any bar under a copy score of 0.75 against the reference's bars. The same machinery extends to a riff or a pulse:
- per 16th, the share of bars with an onset (the [onset profile](11-style-beat-lessons.md#3-statistics-have-to-survive-the-same-pipeline-rhythm) at 48 points a bar is the transcription-free version);
- the pitch-class shares and the step between successive notes, in the register's 5th–95th percentile;
- the period (one, two or four bars) from the repeat;
- the copy score on (16th, pitch) per bar, as for the 808.

Basic Pitch's fragments then cost little: the shares count onsets and pitch classes, and a pad's notes are read from pitch classes per 8th note, as `melodic` reads chords ([01](01-pipeline.md)), rather than from note lengths. Every statistic has to hold on a candidate's own render passed through the same separation and transcription first ([11](11-style-beat-lessons.md#3-statistics-have-to-survive-the-same-pipeline-rhythm)).

**Sampled loops.** A layer can be a recorded loop or chop rather than played notes. Measured on the suite (seed 1, the htdemucs other stem, 40 s from 4 s): the highest normalised waveform autocorrelation between 0.5 and 20 s was 0.19–0.35 on the 17 beats that tile a library loop and 0.08–0.09 on the two that do not (melodic-1, full-1); three loop beats are left out, as their songs of 22–23 s leave under a second of overlap at the lags found. The loop's own length was the best lag on 4 of the 17; on the rest it was two bars (11), four (1) or eight (1). PIRAMIT's other stem, in which an earlier local study found no library file playing, scored 0.35 at 4 bars. Repetition says "loop-like", not "sampled". CLAP and AST say neither: AST has no class for a sample, and CLAP put `v1-1`'s true loop file 1,157th of 1,698 loops (section 2). A loop is identified when its file is in the library: in an earlier local study SampleID found the loops of the six v1 calibration beats at rank 1 in 12 of 12 unchanged stems, and in 34 of 48 shifted by 1–3 semitones or 5 % in tempo. Otherwise it is not identified.

## 5. Is there a measure of "similar" that predicts the ear?

Not yet, on this evidence:
- **CLAP alone:** on the user's own A/B files it put the pick first in 1 of 3 tests (section 2), and on HAYIR DEMELİYDİM the style score of a rejected beat equalled the song's own halves (0.849 against 0.850, [11](11-style-beat-lessons.md)).
- **CLAP against the descriptors:** on PIRAMIT they rank the same 30 candidates almost independently (Spearman 0.21).
- **Separated against separated,** with the reference's own stems: every candidate rises into 0.72–0.93 and the plain additive pad wins.
- **Where both measures work:** on a clean layer whose true sound is among the candidates (the suite), both put it first. A real song gives neither condition: its layer is separated, and its true sound is seldom in the pool.

So the measures shortlist and the ear chooses. The cheapest protocol that also teaches the measures:
1. **Per layer, one round.** One `listening-round` of 8 bars:
   - the top 3 of stage 2 and the fitted voice;
   - the stock sound as a control;
   - each against the stem through its space, at the same loudness.

   That is 5 files of about 30 s, about 3 minutes of listening. PIRAMIT's round from this note (6 files: the top 3, the fitted voice, Drift's default and a "warm pad" hit) stays local.
2. **Record each pick** with the candidates' numbers (the proposed `sounds --pick N`, section 6).
3. **After about 20 picks,** score every measure by how often it puts the pick first, against chance. A measure that does not beat chance is shown, not used. MERT, not tested here, goes into that comparison; so do the study's composite score and CLAP inside the layer's AST family.

## 6. The flow in ableton-live-mcp

Today a "song like this one" in alm has no prompt of its own. Proposed order, the producing side (rd) first, then alm's pin:

**In reference-dissector** (commands that write files a model can read; none needs Live):
1. `refdissect layers WORK`: per tonal stem, the layers inside it (split by register, onset density and band, as the HAYIR DEMELİYDİM study did by hand), each with:
   - its kind (section 4);
   - its instrument family (AST's instrument labels, with their probabilities);
   - its space (`space`: reverb time and wet share fitted as in section 2, width per octave, any band-pass);
   - its repeat (period and strength).
   It writes `layers.json`.
2. `refdissect index-presets`: CLAP vectors of Live's preset previews, beside the library index (section 2: about a minute, no Live).
3. `refdissect sounds WORK --layer L`: the two stages of section 2 for one layer.
   - Stage 1 by CLAP, inside the layer's family. Untested: AST's family scores on separated stems are weak (PIRAMIT's top instrument label 0.057, the three-layer stem's 0.055), so the filter has to be measured before it is trusted.
   - Stage 2: the top 30 played on the layer's own notes through its fitted space, together with a voice fitted to the layer, rendered as a multisample, and the stock sound as a control.
   - Both measures and their ranks, the top 3 and the control into one `listening-round`, and the pick written back with `--pick N`.
4. `refdissect candidates --parts riff,pulse`: new notes of the layer's kind (section 4), each candidate played on the picked sound.
5. `plan` and `blueprint`: a melodic track gets the picked sound as `{"sample_path": …}` and its space as `effects` instead of `"Drift"`:
   - Reverb (Decay Time, Dry/Wet);
   - Utility (Width);
   - EQ Eight for a band-pass.

   alm's `build_tracks` already takes both. Then `check-rebuild` measures the result as before.

**In ableton-live-mcp**, after rd releases these:
- **A prompt, `song_like_reference`:** dissect → layers → per layer: sounds, listening round, the user's pick → candidates for the parts, listening round → build → `compare_to_reference`. It is a new prompt, not a change to `beat_like_reference`, which stays the 1:1 replica.
- **A rule in it and in the compose skill.** For a melodic layer of a reference, no instrument is loaded by name (`browser_search`, `add_device`) unless it is in that layer's stage-2 shortlist or the user asks for it. `generate_melody` is not used to imitate a reference that has no lead. The CLAP order is required, but it is not enough: the ear picks among the shortlist (section 5).
- **The pin** to the rd release, as for 2.4.x.

## 7. Recommended path

1. **Read the layer before choosing anything.** Split the other stem into layers, then read each layer's kind, family, space and repeat. The ear judges the kind first, and every measure above needs one layer's notes (section 4).
2. **Choose its sound from a shortlist, not by name.** Ways A and B together:
   - search the library and Live's previews, inside the layer's family once that filter is measured;
   - fit a voice;
   - play the shortlist on the layer's own notes through its fitted space and rank it by both measures;
   - three candidates, the fitted voice and the stock control go to the ear.

   Way C stays a personal-study control: the ceiling a candidate is heard against, never shipped. Way D is not proposed: MusicGen's and RAVE's weights are non-commercial, MusicGen's melody conditioning copies the notes, and DDSP needs a model trained per timbre.
3. **Write new notes of the layer's kind,** with the copy rule `candidates` already has, each candidate played on the picked sound.
4. **Build** the picked sound and its space into the blueprint, then measure with `check-rebuild`.

### The first three jobs

| Job | What | Measured by |
|---|---|---|
| 1. `layers` | Split the other stem into layers; per layer: kind, AST family, fitted space (reverb, width, band-pass), repeat; `plan`/`blueprint` carry a melodic sound slot and its `effects` instead of `"Drift"` | On the suite's melodic beats: each truth layer's notes found in one layer (note F1 per layer against the truth), its kind read right; the space fitted within the reference's own spread on `compare`'s width and tail measures; in Live, the blueprint's Reverb, Utility and EQ Eight reproduce the offline space within 1 dB per octave |
| 2. `index-presets` and `sounds` | Live's preview vectors beside the library index; per layer, stage 1 inside the family, stage 2 on the layer's notes through its space with a fitted voice and the stock control; a listening round; `--pick` | The truth sound's rank on the suite's clean and separated layers (here, on the instruments pool, 11th to 1,074th at stage 1; at stage 2 first on the clean layers, 4th to 12th on the separated stem). Stage 2's best against the stock control on PIRAMIT and HAYIR DEMELİYDİM: here +0.13 CLAP over Drift's default and +0.35 to +0.44 over the "warm pad" hits on PIRAMIT. The Live recordings' top 10 inside the previews' top 50 (here their median preview rank was 5–21). The user's picks are logged. |
| 3. `candidates --parts riff,pulse` | Riff and pulse drawn from the layer's onset shares, pitch-class shares, steps, register and period, kept under the copy score per bar, played on the picked sound | Copy score per bar under 0.75 on (16th, pitch); the coarse statistics of section 4 on the candidate's own render, separated and transcribed, inside the reference's own spread; the user's pick in an 8-bar round |

## 8. Open questions

- **Which measure predicts the ear?** Three picks so far. About twenty, from the protocol in section 5, before any measure chooses alone.
- **How to split one stem into layers.** A register split did not separate the suite's layers on Basic Pitch's notes; onset density was not tried. Candidates are a band per layer from the stem's spectrum (as the HAYIR DEMELİYDİM study did by hand), chroma per half bar for pads, and identifying a lead's sound before transcribing it (an earlier local study: 4 of 5 library leads found from the other stem by note templates).
- **Sampled layers.** Neither CLAP nor AST finds a cut from a record. A loop in the library is found by SampleID (Sony, MIT; 12 of 12 unchanged stems of the v1 calibration beats in an earlier local study). A cut from a record that is not in the library can only be told apart as "loop-like".
- **Preset parity.** A preset is recorded at one note and repitched here. Over PIRAMIT's two octaves (24 semitones between its notes' 5th and 95th percentiles) its envelope and filter do not follow as the preset would. A shortlist recorded in Live at the layer's own notes settles it (about a minute for 30 presets).
- **The fitted voice in Live.** Rendered to a multisample it plays as rendered. Mapped onto Drift or Operator, it needs a parity check in Live, as the 808 had.
- **Where way C ends.** A multisample of the stem's notes is the closest sound and the least new. The line between a personal reference and a song "like" the original is the user's call ([09](09-data-and-copyright.md)).
- **The suite's melodic voices are plain synths.** Stock Drift is close to them by construction. A held-out set of real songs with picks by ear is the only test of the sound choice that counts.
