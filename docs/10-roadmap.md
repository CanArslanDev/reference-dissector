# 10 – Roadmap

In rough order of value. Each item names how it will be measured: the calibration beats are the answer key for most of them.

1. **Better drum transcription for kicks and hats.** Blind F1 is 0.70 / 0.77 today; snares and open hats are fine.
   - The candidates:
     - a drum separation model (LarsNet, drumsep) before template discovery;
     - an ADT model;
     - NMF with the templates as fixed bases, so co-occurring hits are explained together instead of one at a time.
   - **Measured by:** `calibrate`, F1 per sound, and the hand-checked bars of PIRAMIT.
2. **Symmetric comparison.** Render the rebuilt parts from Live (ableton-live-mcp's `render_and_analyze` / `capture_stems`) and master them like the reference. Then separate them with the same Demucs and compare stems with stems: the separation's artifacts are then on both sides.
   - **Measured by:** per-sound null tests and descriptor differences before/after.
3. **The closed loop.** A listening report per sound ([06](06-listening-for-a-model.md)):
   - descriptor differences → suggested edits (decay, transpose/detune, EQ, drive, reverb);
   - apply them through ableton-live-mcp;
   - render, measure again, repeat while the distance falls.
   - **Measured by:** distance per iteration.
4. **808 slides and a grittier 808.**
   - Play slides with overlapping notes and Simpler's glide, or with pitch automation.
   - Extend the voice with asymmetric drive and a weighted upper-harmonic term.
   - **Measured by:** harmonics 6–12 within 3 dB, glide shape.
5. **Melodies and chords.**
   - Basic Pitch on `other` / `piano` / `vocals` (ableton-live-mcp `audio_to_midi`), with the track's tuning applied.
   - Instruments chosen by CLAP (`find_similar_sounds`) or synthesized.
   - **Measured by:** note F1 against hand-checked bars; chroma distance.
6. **Layer detection.** A snare that is a library snare plus a clap. After the best match is subtracted (null test residual), search the residual again (matching pursuit).
   - **Measured by:** calibration beats with layered sounds.
7. **Style mode and profiles** (the original brief):
   - pattern statistics per artist or album (hat roll rate, kick placement, 808 glide rate, chord vocabulary, section lengths) from several songs;
   - new beats generated from those statistics, not from one song.
8. **Packaging.**
   - Expose `analyze` / `match` / `plan` as jobs in ableton-live-mcp (`analyze_reference`, `build_from_blueprint`), so an AI client can run "make a beat like this song" end to end.
   - Run the packaged `live-build` against Live in CI-like checks.

9. **Style beats** ([11](11-style-beat-lessons.md)).
   - A sketch step between `plan` and the build: 8-bar candidates per section, chosen by ear.
   - Wire `groove`, `space` and `mixsim` into `refdissect` commands (today they are a library).
   - **Measured by:** the listener's picks, and ducking and balance against the reference.

**Not planned:** publishing transcriptions or audio of songs, for the reasons in [09](09-data-and-copyright.md).
