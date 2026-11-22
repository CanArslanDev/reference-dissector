# 13 – A song like this one: where the melodic layer comes from, and how to make it similar

Asked for "a song like this one", the tools give drums and an 808 that come close and a melodic layer that sounds unrelated to the reference: notes on a stock piano, pad or synth (on the replica path the notes are the reference's own transcription on Drift, built muted; the reference's own stems play by default). This page traces where that layer's sound and notes come from on each of the three paths, measures what would choose a closer sound, and proposes the order of work. It is a research note; the one command that came out of it, `refdissect melody-check`, is a separate change.

**In short** (2026-11-21/22, PIRAMIT, bars 13–20):
- The user heard none of six files as close: four candidates chosen against the layer and two stock controls. Three versions of a sound resynthesized from the stem's own hits were rejected: noise and wrong notes, then worse twice.
- Every one of those tests played notes nobody had confirmed. Where the user judged the melody (the resynthesized versions, a melody check) they heard it as wrong. The stem the notes were read from holds more than one sound: the riff, a sustained layer at the same pitches, and bleed.
- No reading tried here gets a layer's notes out of such a stem: on the suite the best is a precision of 0.80 at a recall of 0.72, with a second separator and a second transcription model. Alone, the suite's pluck is read at 0.99 (its pad at 0.44).
- A split of the stem from known notes, without a learned model, did not give a clean layer on PIRAMIT, with the notes the user had confirmed: 3.2–3.9 dB against the layer, where the unsplit stem scores 4.2–4.3 dB.
- What worked was the other order. A separator that puts each sound in its own stem gave three stems the user heard as cleanly separated. Basic Pitch on each of two of them gave notes the user confirmed by ear without changing one (88 and 17). One layer's own sound (cut from the layer: personal study), played on its confirmed notes, then scored inside the layer's own spread (CLAP 0.947 against 0.942), and the user heard it as right; on the mixed stem the same test had scored 0.47–0.55.
- So single layers come first, then each layer's notes, confirmed by ear, then the render path, then a sound (section 8). The separator that did this has no clear licence, and htdemucs_6s does not come near it by the measures here (the ear has not judged it): that is the first open question. All of this is one song and one listener.

As in [09](09-data-and-copyright.md), no audio, stems, transcriptions, library file names or preset names are published. The numbers come from local runs. Code is cited at reference-dissector `8484988` (rd) and ableton-live-mcp `d6f28e5` (alm).

## Which path the earlier attempts took

The job records hold only two PIRAMIT dissects (2026-11-16), the replica path. The study folders show the rest:

| When | Reference | Path | Melodic sound | Verdict |
|---|---|---|---|---|
| 2026-10-11/12 | PIRAMIT | 3: the assistant's own scripts over alm's tools | drafts 1–3: Live presets chosen by name (a reed lead, a muted guitar, a sitar, a pad), then CLAP against another song's stem; draft 4: riff and pulse on library sounds; draft 5: each of PIRAMIT's melodic stems' CLAP first; draft 6: notes cut from PIRAMIT's own stems into a Drum Rack | 1–5 rejected; 6 the best so far |
| 2026-10-15/17 | HAYIR DEMELİYDİM | 3, with rd's modules ([11](11-style-beat-lessons.md)) | sounds chosen by ear in A/B files from a CLAP + band + envelope shortlist | v1 and v2 rejected: v1 for its drums and mix, v2 for its parts; the sounds had been picked by ear |
| 2026-10-17/18 | a third song | replica-style through the assistant's own scripts (the song's own melodic stems), then 3 | the song's own stems, which the user heard as the original's melody; then library sounds by CLAP and ear | kept after picks by ear |
| 2026-11-08 | four songs | 3 | Live presets: piano, then a sitar, then plucks; then, after separating again, a CLAP shortlist over preset previews and library one-shots, played in Live on the reference's phrase with its measured reverb (strings, pads, a synth) | no draft picked: the user heard no piano or synth in any of the four songs |
| 2026-11-21/22 | PIRAMIT | the assistant's own scripts, offline (this note) | a shortlist by CLAP and descriptors, a fitted voice, the stock sounds; then sounds resynthesized from the stem's hits | the first day: none heard as close, the notes heard as wrong. The second day: with each sound in its own stem, two layers' notes confirmed by ear, and one layer's own sound heard as right (section 6) |

Path 2 (`refdissect candidates`) has never been heard. Every earlier "song like this one" went through path 3 at least in part. No step of the tools ties the sound to the reference there; where a sound was tied to it (CLAP against a stem, notes cut from a stem), the assistant's own scripts did it.

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
  - 2,141 files of the library chosen for melodic use in an earlier local study (one-shot folders, Core Library multisamples, and files tagged as loops of 6 s or less), each with a root pitch estimated there. Their tags: 1,191 loops, 686 other, 143 percussion, 112 effects, 9 vocal. 572 of them are over 6 s (444 in one-shot folders, 128 multisamples), the longest 319 s; 275 of those are tagged as loops, up to 123 s.
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

- Against a separated stem, CLAP's nearest whole files are long loops of a minute or more, as recorded-music cuts were in the 2026-11-08 study. A clean, single layer brings 2–7 of the top 10 from its AST family (6–7 on the pluck, 3–6 on the pad, 2–3 on the lead); a separated stem brings 0–1.
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
- **Against the stock flow, by the numbers.** On PIRAMIT the shortlist's best by CLAP beats the "warm pad" hits by 0.35–0.44 and Drift's default by 0.13. It was picked by the same CLAP it is scored on, so the margin is biased upward. By the descriptors it is 20th of 35 (4.83, outside the song's own 0.71–4.08), and Drift's default is 3rd (1.74). The "warm pad" hits are 33rd and 35th of 35 by CLAP (0.20–0.30), 19th and 30th by the descriptors. On the suite, whose voices are plain saws and squares, Drift's default saw is close by construction: 0.80–0.85 CLAP. That puts it ahead of most of the shortlist on the pluck (ahead of 24 of 30) and the three-layer stem (27 of 30), about midway on the lead (17 of 30) and behind two thirds of it on the pad (ahead of only 9 of 30). The suite cannot show what a stock synth costs on a real song; here the real-song evidence is one 8-bar excerpt of PIRAMIT.
- **Space first.** Through the fitted reverb, CLAP on PIRAMIT moved by −0.12 to +0.20 against the same renders dry. The median was +0.06, and 24 of the 35 rose. The stock sounds and the shortlist's first all rose: the best preset 0.473 → 0.647, the fitted voice 0.395 → 0.533, Drift's default 0.309 → 0.513. The order changed too (Spearman 0.80, dry against wet): two of the dry top 3 are not in the wet top 3. Ranking dry candidates against a wet stem picks other sounds, as in the HAYIR DEMELİYDİM study ([11](11-style-beat-lessons.md#2-effects-before-sounds-space)).
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

The effects are measured per layer. `refdissect space` prints width per octave and the band-pass between two spans. `space.tail` reads a reverb time after a stop, but the command does not use it. A decay and wet share fitted for a reverb device (section 2) is not in rd yet. Each way carries them to Live the same way, as the layer track's `effects` in the blueprint, which alm's `build_tracks` accepts: Reverb (Decay Time, Dry/Wet), Utility (Width), EQ Eight (high- and low-pass). Two gaps remain:
- one Utility Width cannot carry a width per octave;
- the fit's wet share mixes the dry signal with a reverb at the same RMS, which is not Live Reverb's Dry/Wet. The mapping has to be calibrated by recording Live's Reverb, as the HAYIR DEMELİYDİM study did for its chords and arp.

Ways A, B and B2 are ranked through that space; C has it baked in.

| Approach | What it needs | Cost | Expected quality (measured above) | Risk |
|---|---|---|---|---|
| **A. Search the library and Live's presets by embedding, then re-rank on the layer's notes** | CLAP (615 MB, Apache-2.0); AST (346 MB, BSD-3) for the family; an index; the layer's notes and space | Index: 3,996 files with CLAP and AST in 615 s, once. Presets from Live's previews in 57 s, not 35 minutes of recording. Per layer: 10–45 s for 35 renders and their vectors. | PIRAMIT by CLAP: 0.647 against Drift's default 0.513 and the "warm pad" hits 0.20–0.30, ceiling 0.83–0.94; picked by the same CLAP, so biased upward. By the descriptors the same preset is 20th of 35 and Drift's default 3rd. Suite lead: 0.910 against the truth's 0.956. | Stage 1 alone misleads: long recordings on mixed stems, and the right sound at rank 38. CLAP and the descriptors disagree on real stems. A one-note recording repitched is not the preset over its whole range. The pool may lack the family (2026-11-08: no piano or synth in the songs, mostly presets in the pool). |
| **B. Fit a voice to the layer (synthesizer sound matching)** | A numpy voice with Drift-like controls (here 13: one harmonic shape played by two detuned copies, a sub, a low-pass with envelope and resonance, ADSR, noise), a loss, an optimiser. Rendered per pitch into a multisample for Sampler, as `fit-808` renders a one-shot for the 808; mapping it onto Drift, Operator or Wavetable needs a check in Live, which renders instruments only in real time (1.12 s a render). | 6–11 minutes a layer: 2,704 evaluations, one core, differential evolution | PIRAMIT: first by the descriptors (0.95, inside the song's own 0.71–4.08), 0.533 CLAP. Suite pluck: 0.908, second only to the truth. | It is fitted on the descriptor loss, which therefore flatters it; CLAP is the independent check. Subtractive timbres only: no piano, strings or recorded sounds. The earlier local study found no 2024–2026 work that fits Live's own devices; the learned matchers that work need 3·10⁴–2·10⁶ renders of one synth, and Live renders in real time. |
| **B2. Resynthesize from the layer's own hits** | Per pitch, the median over the stem's hits of harmonics 1–20, residual noise, and an 80 ms attack envelope and click measured in 1 ms frames, synthesized from that table (section 6); notes to play it on | Minutes | Not judged as a sound: the three versions the user heard played unconfirmed notes. Version 2 flickered (heard as a crackle); version 3 smoothed its envelopes, not confirmed by ear | Its timbre is the reference's, averaged over the stem's hits, from a stem that holds other sounds and reverb; the table carries them. An average of the song's hits is personal study only, as averaged templates are ([09](09-data-and-copyright.md)) |
| **C. A multisample from the stem's own notes** | Basic Pitch notes, each pitch's median over its hits, laid out as Simpler or Sampler zones | Minutes | PIRAMIT draft 6 (2026-10-12): the riff 0.759 CLAP against the song's own sections' 0.762; its keys 0.495 | **It is the song's own recording: personal study only, never in a release or a public repository** ([09](09-data-and-copyright.md)). Bleed and reverb come with it. Pitches the song never plays are repitched. Chords and pads do not cut into notes. A song that sounds "like" its reference by being its sound. |
| **D. Generative models** | MusicGen-melody (1.5 B parameters, 6.2 GB, weights CC-BY-NC 4.0); MusicGen small (2.4 GB, CC-BY-NC 4.0, which alm already uses: an 8 s loop in ~19 s on Apple Silicon); DDSP timbre transfer (Apache-2.0, TensorFlow, pretrained on four acoustic instruments); RAVE (CC BY-NC 4.0, trained per timbre) | Downloads of 2.4–6.2 GB; RAVE and DDSP need hours of training per timbre | Not measured | MusicGen's and RAVE's weights are non-commercial. DDSP is Apache-2.0 but runs on TensorFlow, has four acoustic instruments and needs hours of training for any other. Melody conditioning feeds the reference's chroma, so it copies the notes. No hold on the exact timbre. A model trained on the reference's stem copies it. |


## 4. Reading what kind of part a layer is

The ear judged the kind of part first: PIRAMIT has a riff of three pitch classes in 16ths and no lead, and its three drafts with a lead were rejected (the history above). Basic Pitch's fine statistics on a reverb-heavy layer measure the transcription, not the music; only coarse ones survived ([11](11-style-beat-lessons.md#3-statistics-have-to-survive-the-same-pipeline-rhythm)). These coarse statistics were read over 8 bars:
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
| the same, split in 3 by register: low / mid / high | 2 / 2 / 1 | 2.2 / 2.0 / 0.8 | 5.2 / 8.9 / 5.2 | 3 / 4.5 / 3.5 | 13 / 10 / 11 | riff / pulse / lead | mostly pad, some pluck / all three / mostly pluck, some lead |
| PIRAMIT other stem, Basic Pitch | 2 | 2.5 | 6.4 | 3 | 24 | riff | riff (and some keys) |

- On truth notes the rule reads the pad and the pluck, as it was set to. It calls the suite's lead a riff: the suite's lead plays 2 pitch classes a bar, and nothing in these numbers tells a sparse lead from a riff. What would is the period: a riff repeats every one or two bars, a lead phrase does not. The waveform repeat below measures that, but the stem's other content gets into it.
- On the separated stem the layers are not separable this way. The suite's pad plays five notes at once for six 16ths in the truth. The Basic Pitch notes on the other stem that match it (the same MIDI pitch, onset within 60 ms) read two notes at once for 3 16ths, and 37 of its 80 truth notes are found: its chords come out thinned and cut short. Only the middle register cluster mixes all three layers (35 lead, 29 pluck, 19 pad notes matched). The low one holds the pad's fragments with some of the pluck's low notes (24 and 8; 29 of its 61 notes match no truth note). The highest is mostly the pluck's top notes (40 of 46), not the lead. So every reading of a layer's kind needs its notes from that layer alone, and Demucs gives one stem for all of them. **A layer split inside the other stem is missing**, and it comes before any choice of kind or sound. The HAYIR DEMELİYDİM study split its layers by hand, with its own scripts.
- PIRAMIT reads as a riff: three pitch classes a bar.

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
- **Against a stem of several sounds, not even the song's own sound scores high.** PIRAMIT's own note samples, played on the same unconfirmed notes, scored 0.47–0.55 CLAP, as Drift's default did (section 6). A score of one sound against such a stem says little about that sound.

So the measures shortlist and the ear chooses. The cheapest protocol that also teaches the measures:
1. **Per layer, one round.** One `listening-round` of 8 bars:
   - the top 3 of stage 2 and the fitted voice;
   - the stock sound as a control;
   - each on the layer's confirmed notes, against the layer alone, at the same loudness (section 8: after the layer's own sound has passed the render path).

   That is 5 files of about 30 s, about 3 minutes of listening. PIRAMIT's round from this note (6 files: the top 3, the fitted voice, Drift's default and a "warm pad" hit) stays local.
2. **Record each pick** with the candidates' numbers (the proposed `sounds --pick N`, section 7).
3. **After about 20 picks,** score every measure by how often it puts the pick first, against chance. A measure that does not beat chance is shown, not used. MERT, not tested here, goes into that comparison; so do the study's composite score and CLAP inside the layer's AST family.

## 6. What the ear said: single layers first, then their notes

After section 2 the user listened, in rounds, to PIRAMIT's bars 13–20. Each file was the reference stem, a beep, then the candidate. The files stay local. Rounds 2 and 5 got no verdict by ear. In order:

1. **The shortlist** (section 2): its top 3 by CLAP, the fitted voice, Drift's default and a "warm pad" hit, on Basic Pitch's notes, through the fitted reverb.
   - Verdict: none like the reference. The user had said from the start that the chosen sounds were the problem, and these came from the same kinds of sources as before.
2. **A control** (not judged by ear): PIRAMIT's own note samples, cut from the stem and averaged per pitch in an October study, played the same way.
   - CLAP 0.52 dry, 0.47 wet: the level of Drift's default.
   - Two faults turned up. Two of those note files started 220–245 ms late. And the stem's broadband onsets (spectral flux) do not agree with Basic Pitch's notes: matched one to one within 50 ms, 50 of Basic Pitch's 66 distinct starts meet one of the 142 onsets.
   - Many of those 142 onsets are not the riff's. They fall in 109 of the 128 16ths, while the notes read later on the 16th grid fill 62; 33 more fall in a 16th already taken, and 71 lie over 25 ms off the grid.
   - With a note at each onset, or with Basic Pitch's notes, and the files trimmed to their attacks, it scored 0.48–0.55.
   - The stem holds the riff, a sustained layer at the same pitches and bleed. One sound, even the song's own, is not that stem.
3. **A sound resynthesized from the stem's hits** (the user's choice among the next steps):
   - per pitch, the median over the stem's hits of harmonics 1–20, a residual noise in 24 bands, and an 80 ms attack envelope and click measured in 1 ms frames;
   - synthesized from that table, not from the recording's samples.

   Three versions:

   | Version | Notes from | What the user heard | What was measured afterwards |
   |---|---|---|---|
   | 1 | an onset of the stem (broadband flux), pitched by harmonic salience | noise between the notes; wrong notes | the onsets read as partly drum bleed (off the grid, two to a 16th; not measured against the drum stem); the residual "noise" read as the stem's reverb and bleed |
   | 2 | rises of harmonic contrast at the riff's four pitches on the 16th grid, two voices, plus held notes | worse: a crackle; the melody unrelated to the song | the envelopes of harmonics 1–10 jumped 0.5–0.7 dB per 2.9 ms frame (median; 90th percentile 1.5–2.3 dB), and 6–30 % of their cells were zero, flickering on and off; between notes the render was quiet (over 6 kHz in the quietest fifth of 20 ms frames, −55 dB against the stem's −34 dB) |
   | 3 | Basic Pitch's notes again; envelopes smoothed (median 0.1–0.2 dB a frame, 90th percentile 0.4–0.9 dB, measured as for version 2) | the melody worse still | – |

   Each version changed more than one thing at once, and none was heard on confirmed notes.
4. **The melody alone:** Basic Pitch's posteriorgrams read on the 16th grid (onset or note-posterior rise, Otsu thresholds), at most two notes a 16th.
   - The result: 71 notes in 62 of the 128 16ths, 68 of them on four pitches. In 5 of the 8 bars the second half repeats one four-note figure.
   - Played on a plain tone after the stem, and against it in the other channel.
   - Verdict: not right. The user added that the reference itself holds several sounds, and that the earlier rounds may have failed on that.
5. **A split of the stem without a learned model** (the user's request; Demucs kept for drums, bass, vocals and other; its PIRAMIT result was not judged by ear). Each STFT bin's magnitude is a non-negative sum of fixed activation shapes; the coefficients are solved by non-negative least squares, and the parts are taken out by Wiener masks:
   - short notes: kernels at onsets that rise in 5 ms and fall with time constants of 30, 80 and 150 ms;
   - held notes: boxes over the note's length, flat or falling with time constants of 0.6 and 2 s;
   - a free bed: cubic splines with knots every 0.35 s.

   On the suite's `melodic-1`, bars 5–12, magnitude SDR against the truth (dB, higher is closer):

   | | pluck (short) | pad + lead (sustained) |
   |---|---|---|
   | no split (the input as both) | 1.5 (htdemucs other stem) | 7.5 |
   | short kernels at the stem's onsets, a free bed | 1.8 | 4.0 |
   | the same, at the truth's onsets | 2.8 | 5.5 |
   | each note's kernels only on its harmonics (± 40 cents), the truth's pluck notes | 5.0 | 10.2 |
   | short and held notes both on their harmonics, the truth's notes | 5.5 | 10.8 |
   | the same, notes read from Basic Pitch's posteriorgrams | 2.0–2.2 | 8.3 |

   - With the truth's notes the split moves the pluck 4 dB up. With read notes it moves it 0.6–0.8 dB.
   - The reading found 44 % of the 259 truth notes at a precision of 0.89, with at most two notes a 16th. It found 22 of the pad's 80 notes. Allowing up to four or six notes a 16th raised the recall only to 46 %, at a precision of 0.83. Basic Pitch's own notes find 57 % at 0.72.
   - On PIRAMIT, three splits with a model-free note correction between them left a short layer of 32 notes in bars 13–20, 12 dB under the stem (−35.9 against −23.8 dBFS). Over the 9 bars read (one bar of lead-in), 59 notes read as short at first; the two corrections moved 40 notes from short to held, as their energy held on. In the spectrogram, the riff's attacks stayed in the held part and the bed.

### Which reader gets the notes out of a mixed stem (2026-11-22)

Measured on the suite's four beats with melodic truth, all 16 bars: `melodic-1` and `melodic-2` (pluck, pad, lead) and `full-1` and `full-2` (pad, lead); 1,553 notes (521 pluck, 352 lead, 680 pad). A note counts when an estimate of the same pitch starts within 30 ms of it, one to one. Estimates carry no layer, so each layer has a recall, and the precision is over all estimates. The aim was a recall and a precision of 0.8 on the riff-like layer (the pluck).

| Stem read | Reader | Precision | Recall | F1 | Pluck recall | Lead recall | Pad recall |
|---|---|---|---|---|---|---|---|
| the pluck alone (truth) | Basic Pitch, refdissect's preset | 1.00 | 0.99 | 0.99 | 0.99 | – | – |
| the pad alone (truth) | the same | 0.46 | 0.44 | 0.45 | – | – | 0.44 |
| the truth's layers summed | the same | 0.70 | 0.62 | 0.66 | 0.66 | 0.76 | 0.52 |
| htdemucs `other` | Basic Pitch's defaults (0.5 / 0.3 / 127.7 ms, melodia) | 0.42 | 0.38 | 0.40 | 0.23 | 0.70 | 0.34 |
| htdemucs `other` | refdissect's preset (0.6 / 0.3 / 35 ms) | 0.65 | 0.53 | 0.59 | 0.61 | 0.71 | 0.39 |
| htdemucs `other` | the stem settings of the melodic study (0.6 / 0.4 / 27.6 ms) | 0.66 | 0.50 | 0.57 | 0.52 | 0.71 | 0.37 |
| htdemucs `other` | the grid reading, two notes a 16th | 0.86 | 0.43 | 0.57 | 0.46 | 0.66 | 0.28 |
| htdemucs `other` | the stem settings, plus the grid's notes where they have none | 0.67 | 0.56 | 0.61 | 0.61 | 0.75 | 0.43 |
| htdemucs `other` | YourMT3+, its duplicates across programs removed | 0.56 | 0.56 | 0.56 | 0.67 | 0.74 | 0.37 |
| htdemucs `other` | the same, plus the grid's notes where it has none | 0.58 | 0.68 | 0.62 | 0.74 | 0.90 | 0.51 |
| htdemucs_6s `other` + `guitar` + `piano` | refdissect's preset | 0.69 | 0.49 | 0.57 | 0.60 | 0.70 | 0.29 |
| BS-Roformer-SW `other` | refdissect's preset | 0.67 | 0.55 | 0.61 | 0.62 | 0.72 | 0.41 |
| BS-Roformer-SW `other` | YourMT3+, its duplicates across programs removed | 0.82 | 0.60 | 0.70 | 0.70 | 0.61 | 0.52 |
| BS-Roformer-SW `other` | the same, plus the grid's notes where it has none | **0.80** | **0.72** | **0.76** | 0.77 | 0.84 | 0.62 |

- **No reader reaches 0.8 and 0.8 on a stem of several layers.** The closest is YourMT3+ with the grid reading added, on BS-Roformer-SW's stem: precision 0.80, pluck recall 0.77.
- **For Basic Pitch, the layers on top of each other are the limit, not the separator.** Alone, the pluck is read at 0.99. In the perfect sum of the truth's layers its recall is 0.66, against 0.61 in htdemucs' stem. YourMT3+ does depend on the separator: its precision is 0.56 on htdemucs' stem and 0.82 on BS-Roformer-SW's. A layer alone is not always read well either: the pad at 0.44.
- **Lowering Basic Pitch's thresholds trades one for the other.** Over onset 0.3–0.5 and frame 0.2–0.4 on htdemucs' stem, the pluck's recall reaches 0.82 at a precision of 0.24.
- **A split that needs no notes did not help the reading.** Each STFT bin split by a running median over time (0.2, 0.4, 0.8 s) into a held and a short part, each read on its own: the pluck's recall 0.56–0.63 in the short part of htdemucs' stem at a precision of 0.53–0.73, the pad's 0.16–0.36 in the held part.
- **refdissect's loop vote did not move it:** 0.64 and 0.53 on htdemucs' stem.
- YourMT3+ runs in its own environment (weights Apache-2.0, code GPL-3.0); BS-Roformer-SW's licence is not clear. Both were used for a draft only. MuScriptor was not tried (non-commercial weights).

On PIRAMIT, BS-Roformer-SW puts the melodic content of bars 13–20 into three stems: `piano` at −27.8 dBFS (correlation with htdemucs' `other` 0.75), `other` at −34.1 dBFS (0.39) and `guitar` at −43.0 dBFS, against −25.8 dBFS for htdemucs' `other` (levels of the mono sum); the three tonal stems together correlate at 0.85. A draft for the user's correction was read by the best reader above on that sum: 103 notes. At the suite's figures about one note in five of it is wrong and more than one in four is missing. The user did not correct it: in htdemucs' stem they heard at least three sounds, and asked what to do about that.

### One sound at a time (2026-11-22)

**The stems.** The user listened to BS-Roformer-SW's `piano`, `other` and `guitar` stems of bars 13–20 one by one, and heard three cleanly separated sounds. The one named `guitar` may not be a guitar, they said. AST agrees: its top labels there are Sonar 0.25 and Synthesizer 0.04. By its channels' levels that stem sits 20 dB to the left, the `piano` stem near the centre (0.8 dB to the right).

**The notes, read per stem.** Basic Pitch with refdissect's preset, on each stem alone:

| Stem | Notes read in bars 13–20 | The user's verdict, the stem against the notes on a plain tone |
|---|---|---|
| `piano` | 88 | "this is the melody", no note changed |
| `guitar` (peak-normalised; 10 notes before that) | 17 | "this is the melody", no note changed |
| `other` | 5 by Basic Pitch's notes, 58 by the grid reading | no draft sent: the two readings disagree |

The riff the user means is the `piano` and `guitar` stems together.

**The layer from the notes, without a learned model.** With the confirmed notes, the split of round 5 above was run on htdemucs' `other`, and measured against BS-Roformer-SW's stems (magnitude SDR, dB; the stems are a yardstick here, the user having heard them as the sounds alone):

| | `piano` | `guitar` |
|---|---|---|
| no split (htdemucs' `other` as it is) | 4.2 | 0.4 |
| the split of round 5 (a template per note, mono; measured on the mono sum, where no split is 4.3) | 3.2; with held shapes only 3.6 | – |
| a template per layer and pitch, a level per channel for each layer, the bed penalised | 3.8 | 2.9 |
| the same with no bed | 3.9 | 2.5 |
| a mask from the level difference between the channels, no notes | – | 1.2 |

- For the `piano` layer no version beats leaving the stem as it is. The yardstick is another separator's output, not a truth.
- On the suite's `melodic-1` (htdemucs' stem, the truth's notes) the pluck stays where it was (5.5 dB with the first version, 5.4 dB with a template per note in the second, 5.6 dB with a template per layer and pitch), and the shared template moves the lead from 3.4 to 4.3 dB (no split: 4.0 dB); the pad stays near 1 dB, under the 2.1 dB of no split, and the pad and lead together, without the bed, fall from 7.5 dB unsplit to 4.7 dB (4.8 dB with the first version; the 10.8 dB of round 5's table counts the bed with them). On a sum of the truth's layers panned apart, the level per channel adds 0.5 dB on the pluck and 0.8 dB on the lead. The suite's layers are all in the centre, so its stems cannot show that gain.
- The user chose BS-Roformer-SW's stems as the clean layers, for this study only.

**htdemucs_6s does not come near this separation, by these measures.** On the same bars its `guitar` stem is silent (−78 dBFS, mono sum as above) and the level is in `piano` (−31.1 dBFS) and `other` (−30.7 dBFS):

| | htdemucs_6s `piano` | htdemucs `other` | BS-Roformer-SW `piano` |
|---|---|---|---|
| magnitude SDR against BS-Roformer-SW's `piano`, dB | 5.3 | 4.2 | – |
| the 88 confirmed notes found by Basic Pitch (within 30 ms) | 27 %, precision 0.55 (44 notes read) | – | 100 % (they were read there) |
| AST's top instrument label | Pizzicato 0.07 (Speech 0.31 above it) | 0.057 | Piano 0.46 (Keyboard 0.35, Electric piano 0.12) |

The ear has not judged htdemucs_6s' stems.

**The render path, with the layer's own sound** (cut from the song: personal study only, [09](09-data-and-copyright.md)). One cut per pitch from the layer itself, at the confirmed note of that pitch that sounds longest before another starts; the other pitches repitched from the nearest cut; all confirmed notes played by the sampler of section 2; no reverb added, since the cuts carry the room.

| Layer | Clean cuts | CLAP against the layer | The layer's first half against its second | The user |
|---|---|---|---|---|
| `piano` | 4 of 11 pitches | 0.947 | 0.942 | "right" |
| `piano`, the fitted reverb on top (1.5 s, 75 % wet) | – | 0.64 | – | not sent |
| `guitar` | 4 of 4 | 0.73 | 0.90 | "very big errors" |
| `guitar`, each group of notes at the layer's level | – | 0.74 | – | not sent |
| `guitar`, and each note as long as the layer sounds (median 0.37 → 0.58 s) | – | 0.78 | – | not sent |

- On the mixed stem the same test had scored 0.47–0.55 (round 2 above). Against a single layer it scores as the layer scores against itself.
- The `guitar` layer's notes were confirmed as a melody, but the plain tone says nothing about how long a note rings or how loud it is. Its events ring about two and a half beats, the notes read stop earlier, and all were played at one level. Lengths and levels from the layer close a third of the gap; the rest is open.
- The fitted reverb of section 2 is not to be trusted where notes overlap: it reads the gaps after note-offs, and this layer has few.

**The family, on a single layer.** AST's top instrument label is Piano at 0.46 on the `piano` layer, against 0.057 on the mixed stem. The user hears a piano there. Of the pool's 3,996 files, 57 score 0.2 or more on AST's piano, keyboard or electric-piano labels; by CLAP against the layer the best of them is 16th, and the twelve nearest files are long library recordings. No candidate was played: the study stopped here.

`refdissect melody-check` writes the check of a reading or of a corrected file (the stem or a layer file, a beep, the notes on a plain tone; the audio left and the tone right; a MIDI file), and counts what a correction changed against the reading of a stem. A layer file comes with its own notes, and has no reading to be compared with.

**What follows.**
- Every test of the first day played notes that nobody had confirmed. Where the user judged the melody (the resynthesized versions, the melody check), they heard it as wrong. In round 1 they blamed the sounds, but its notes were Basic Pitch's, which version 3 replayed and the user heard as a worse melody, so that verdict may mix the two.
- The notes do not come out of a stem of several sounds, and a split from notes does not clean that stem. Both were the wrong question on PIRAMIT: with each sound in its own stem, the reader refdissect already has gave notes the ear confirmed for two layers. That is one song and one listener, and a layer alone is not always read well (the suite's pad: 0.44; the third stem here: two readings that disagree).
- So the order is: first the single layers; then each layer's notes, confirmed by ear on a plain tone against that layer; then the render path, proven with the layer's own sound, with the notes' lengths and levels taken from the layer; only then a choice of sound.
- What separates the layers is open. The separator that did it here has no clear licence and cannot ship; htdemucs_6s does not come near it on this song by the measures here; the notes-informed split and the mask from the channels did not either.

## 7. The flow in ableton-live-mcp

Today a "song like this one" in alm has no prompt of its own. Proposed order, the producing side (rd) first, then alm's pin:

**In reference-dissector** (commands that write files a model can read; none needs Live):
1. `refdissect layers WORK`: the sounds inside each tonal stem, each as its own audio file (section 6: what does this is the first open question), each with:
   - its kind (section 4);
   - its instrument family (AST's instrument labels, with their probabilities: on a single layer they can be read, section 6);
   - its place in the stereo field and its space (`space`: width per octave, any band-pass; the reverb fit of section 2 only where the notes leave gaps);
   - its repeat (period and strength).
   It writes `layers.json`.
2. `refdissect melodic` per layer (its notes turned into beats from the first bar), then `refdissect melody-check WORK --stem LAYER --audio layer.wav --notes … --bars a-b`: the layer's notes on a plain tone, after the layer and against it in the other channel, and a file of the notes to correct. Nothing below starts on notes the user has not confirmed (section 6). The check confirms pitches and starts; lengths and levels are taken from the layer.
3. `refdissect index-presets`: CLAP vectors of Live's preset previews, beside the library index (section 2: about a minute, no Live).
4. `refdissect sounds WORK --layer L`: the two stages of section 2 for one layer, against the layer alone, after its own sound has passed the render path (section 8).
   - Stage 1 by CLAP, inside the layer's family. On a mixed stem AST's family scores are weak (PIRAMIT's top instrument label 0.057, the three-layer stem's 0.055); on PIRAMIT's single `piano` layer the top label is 0.46. One layer is not a test of the filter: it has to be measured on more before it is trusted.
   - Stage 2: the top 30 played on the layer's own notes through its fitted space, together with a voice fitted to the layer, rendered as a multisample, and the stock sound as a control.
   - Both measures and their ranks, the top 3 and the control into one `listening-round`, and the pick written back with `--pick N`.
5. `refdissect candidates --parts riff,pulse`: new notes of the layer's kind (section 4), each candidate played on the picked sound.
6. `plan` and `blueprint`: a melodic track gets the picked sound as `{"sample_path": …}` and its space as `effects` instead of `"Drift"`:
   - Reverb (Decay Time, Dry/Wet);
   - Utility (Width);
   - EQ Eight for a band-pass.

   alm's `build_tracks` already takes both. Then `check-rebuild` measures the result as before.

**In ableton-live-mcp**, after rd releases these:
- **A prompt, `song_like_reference`:** dissect → layers → per layer: notes, melody check, the user's confirmation; the layer's own sound through the render path (a local check, personal study, never part of the build); sounds, listening round, the user's pick → candidates for the parts, listening round → build → `compare_to_reference`. It is a new prompt, not a change to `beat_like_reference`, which stays the 1:1 replica.
- **A rule in it and in the compose skill.** For a melodic layer of a reference, no instrument is loaded by name (`browser_search`, `add_device`) unless it is in that layer's stage-2 shortlist or the user asks for it. `generate_melody` is not used to imitate a reference that has no lead. The CLAP order is required, but it is not enough: the ear picks among the shortlist (section 5).
- **The pin** to the rd release, as for 2.4.x.

## 8. Recommended path

1. **Single layers first.** A tonal stem is taken apart until each file holds one sound, and the user says so by ear. On PIRAMIT that took a second separator; the notes-informed split did not do it.
2. **Then each layer's notes, confirmed by ear.** Read on the layer alone, played on a plain tone after the layer and against it in the other channel. The user says "this is the melody", or corrects the notes. No sound test or candidate starts on notes that have not passed. On PIRAMIT two layers passed without a change.
3. **Then the render path, proven with the layer's own sound** (a local check; a sound cut from the song is personal study and goes into no build, [09](09-data-and-copyright.md)). The layer's own notes, played by its own sound through the same sampler, must be heard as the layer and score near its own spread before any candidate is played that way. Lengths and levels come from the layer, not from the reading. On PIRAMIT one layer passed (0.947 against a spread of 0.942) and one did not (0.78 against 0.90).
4. **Only then a sound, from a shortlist, not by name.** Ways A and B together:
   - the library and Live's previews, inside the layer's family;
   - a fitted voice;
   - the stock sound as a plainly labelled control.

   Ranked on the confirmed notes, against the layer alone. For an instrument that is sampled across its range (a piano), a preset recorded at one note and repitched is not the instrument: the shortlist is recorded in Live on the layer's notes (section 9, preset parity). Ways B2 and C stay personal-study controls ([09](09-data-and-copyright.md)): both are built from the song's own hits. Way D is not proposed: MusicGen's and RAVE's weights are non-commercial, MusicGen's melody conditioning copies the notes, and DDSP needs a model trained per timbre.
5. **Then new notes of the layer's kind,** with `candidates`' copy rule, each played on the picked sound, and the build with its space; `check-rebuild` measures it.

In every listening round, one thing changes at a time, and it is measured before the files go out.

### The first three jobs

| Job | What | Measured by |
|---|---|---|
| 1. Single layers | A separation that puts each sound of a tonal stem in its own file, with a licence refdissect can ship: other separators measured against the stems the user approved on PIRAMIT, and on HAYIR DEMELİYDİM; `refdissect layers` around it | On PIRAMIT, bars 13–20: magnitude SDR against the approved `piano` stem (htdemucs_6s 5.3 dB, htdemucs' `other` 4.2 dB) and the share of the 88 confirmed notes that Basic Pitch finds on the candidate's stem (htdemucs_6s 27 %). Then the user's ear, stem by stem |
| 2. Notes per layer, and the render-path gate | `melodic` on each layer; `refdissect melody-check --audio` (done); lengths and levels from the layer; the layer's own sound played back through the render path (local, personal study); `plan`/`blueprint` carry a melodic sound slot and its `effects` instead of `"Drift"` | The user's verdict on the melody check per layer. The own-sound playback within the layer's own spread and heard as the layer (PIRAMIT: `piano` 0.947 against 0.942, passed; `guitar` 0.78 against 0.90, not passed) |
| 3. `index-presets` and `sounds` | Live's preview vectors beside the library index; per layer, the shortlist of section 2 on the confirmed notes, against the layer alone, inside its family, recorded in Live for sampled instruments, with a fitted voice and the stock control; a listening round; `--pick` | On the songs, stage 2's best against the stock control by both measures; the user's picks, logged, until about twenty of them can score the measures (section 5) |

## 9. Open questions

- **What separates a stem into single sounds, with a licence that can ship?** BS-Roformer-SW did it on PIRAMIT, by the user's ear, and its licence is not clear. htdemucs_6s (MIT) does not come near it by the measures here: its `guitar` stem is silent there and its `piano` stem gives 27 % of the confirmed notes; the ear has not judged it. The notes-informed split and the mask from the channels' level difference did not either. One song and one listener so far.
- **How to get a layer's notes from a stem of several sounds.** On the suite no reader reaches a precision and a recall of 0.8 on a mixed stem (section 6): the best is 0.80 and 0.72. With single layers the question goes away for PIRAMIT's two confirmed layers; the third stem's two readings disagree (5 and 58 notes), and nobody has checked it.
- **A note's length and level.** The melody check on a plain tone confirms pitches and starts only. The `guitar` layer passed it and failed the render path. Lengths from the layer's decay and levels from the layer moved it from 0.73 to 0.78, against 0.90.
- **A layer's space where notes overlap.** The reverb fit reads what rings after note-offs; a layer with few gaps gets an answer that cannot be trusted (75 % wet on the `piano` layer).
- **Which measure predicts the ear?** Three picks so far, and the first day's PIRAMIT rounds of section 6 cannot count: their notes were not confirmed, and were heard as wrong where the melody was judged. About twenty, from the protocol in section 5, before any measure chooses alone.
- **A model-free split of one stem into layers.** A register split did not separate the suite's layers on Basic Pitch's notes; onset density was not tried. Candidates are a band per layer from the stem's spectrum (as the HAYIR DEMELİYDİM study did by hand), chroma per half bar for pads, and identifying a lead's sound before transcribing it (an earlier local study: 4 of 5 library leads found from the other stem by note templates).
- **Sampled layers.** Neither CLAP nor AST finds a cut from a record. A loop in the library is found by SampleID (Sony, MIT; 12 of 12 unchanged stems of the v1 calibration beats in an earlier local study). A cut from a record that is not in the library can only be told apart as "loop-like".
- **Preset parity.** A preset is recorded at one note and repitched here. Over PIRAMIT's two octaves (24 semitones between its notes' 5th and 95th percentiles) its envelope and filter do not follow as the preset would. A shortlist recorded in Live at the layer's own notes settles it (about a minute for 30 presets).
- **The fitted voice in Live.** Rendered to a multisample it plays as rendered. Mapped onto Drift or Operator, it needs a parity check in Live, as the 808 had.
- **Where way C ends.** A multisample of the stem's notes is the closest sound and the least new. The line between a personal reference and a song "like" the original is the user's call ([09](09-data-and-copyright.md)).
- **The suite's melodic voices are plain synths.** Stock Drift is close to them by construction. A held-out set of real songs with picks by ear is the only test of the sound choice that counts.
