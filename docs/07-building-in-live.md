# 07 – Building in Ableton Live

`refdissect live-build` turns `plan.json` into an Arrangement in the open Live set. It works through [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp): the MCP server and its Remote Script. The build imports the server and calls its tools directly (`mcp.call_tool`), the same tools an AI client uses.

```bash
uv sync --extra live          # pulls ableton-live-mcp from GitHub
uv run refdissect live-build "$W/song/refdissect"
```

Live must be running with the AbletonMCP Remote Script selected as a control surface (see ableton-live-mcp's README). The build refuses unless every track in the set is empty. Empty tracks are reused (renamed) and the rest created, so nothing is overwritten.

## What gets built

| track | content |
|---|---|
| 808 (synth) | Simpler with the fitted one-shot, Transpose +36, an Arrangement clip with every 808 note |
| Drums (library) | Drum Rack: one pad per identified sound (library sample, transposed as matched), an Arrangement clip with every hit |
| REF drums / bass / other / piano / vocals | the stems as Arrangement audio clips from beat 0, warping off; drums, bass and vocals muted |

Also set:
- the song tempo, to the fitted grid (137.002 for PIRAMIT);
- locators at section starts;
- the view, to the Arrangement with the playhead at 0;
- levels: ours at the RMS of the stems they replace, everything 4 dB down for headroom.

**A/B:**
- mute *808* and *Drums*, unmute *REF drums* and *REF bass* to hear the original;
- solo pairs to compare one part.

## ableton-live-mcp tools used

| step | tool or command |
|---|---|
| look at the set | `get_session_overview` |
| tempo, loop | `set_song_properties` |
| tracks | `manage_track` (rename, create_midi, create_audio, set_color) |
| Drum Rack | `add_device` |
| samples onto pads and tracks | `browser_search`, `browser_load` (with `drum_pad`) |
| Simpler settings | `lom_get` / `lom_set` / `param_set_display` on the device's parameters and `sample` |
| clips and notes | `manage_clip` (create_arrangement_clip), `edit_clip_notes` (arrangement_clip_index), 200 notes per call |
| stems | LOM `Track.create_audio_clip(path, beat)`, then `warping = False` on the clip |
| mixer | `set_track_mixer` |
| locators | `place_locators` |
| view | `select_in_live`, `transport` |

Other ableton-live-mcp tools that fit this work:
- **`separate_stems`**: Demucs, the same model.
- **`audio_to_midi`**: Basic Pitch, for melodies.
- **`find_similar_sounds`**: CLAP search by description or by example.
- **`import_midi_file`**: takes `midi/drums.mid` and `midi/808.mid`.
- **`compare_to_reference`**: loudness, dynamics, stereo and tonal balance of the mix against a reference, with EQ Eight settings.
- **`render_and_analyze`**: records the set and analyzes it.
- **`generate_audio`**: MusicGen, CC-BY-NC.
- **`agent_changes(undo=N)`**: undoes the last N tool calls exactly.

## Live behaviour measured here (Live 12.4)

**Simpler:**
- **Loads samples at −12 dB.** The build sets 0 dB through the display value ("0.0 dB"), so the offline level estimate holds.
- **Plays a sample at its own pitch on C3 (MIDI 60).** To keep real note numbers, set Transpose to 60 − root; Transpose reaches ±48 semitones.

**Browser:**
- **New files.** A file written into the User Library (here `Samples/AbletonMCP Generated/`) is listed by the browser within a few seconds. Until then `browser_search` does not find it, so the build retries for up to 40 s.
- **Pads.** Samples load onto Drum Rack pads through the browser: select the pad, then `load_item`. A file elsewhere on disk has to be in the User Library or a Places folder first.

**Stems:** `create_audio_clip` places a file at a beat position on an audio track; with `warping` off, the clip plays at its own speed.

**Live moves a new audio clip's start.** Its warp analysis puts the clip's start at the first transient: 21–30 ms on four stems, 3.6 s on the `other` stem, whose intro is quiet. The start stays there after warping is turned off, so the stem plays that much early against the notes.
- **Fix.** Set the clip's `loop_start` to 0, which carries `start_marker` with it. The build does this and reports each stem's `start_marker` (0) and length.
- **Check.** Unwarped at 137 BPM, PIRAMIT's 126.17 s stems are then 288.09 beats long. Before the fix, `other` was 279.86 beats long.

**Locators:** `place_locators` at fractional beats (0.0685) failed with "Live did not create it"; at whole beats (0, 96, …) it worked. The plan rounds locators to whole beats.

**Speed:** 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s. The Remote Script answers in about 10 ms per command, and notes go in batches of 200.

**The Trial cannot save.** Keep `plan.json`: rebuilding is one command. The plan, the one-shot and the MIDI files are the durable result, not the set.

## Runs

- **The first draft, by hand.** Before `live-build` existed, the PIRAMIT draft was built with a script calling the same tools: the same tracks plus a separate click track, with the snares chosen by hand. It built 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s.
- **`refdissect live-build`, into an empty Live 12.4 set** (2026-10-11): 7 tracks, 6 Drum Rack pads with the identified samples and their transpositions, 549 drum notes, 126 808 notes, 5 stems and 8 locators in 9.7 s. That includes waiting for the browser to list the new 808 file.
  - Checked afterwards: every pad holds its sample, every note is in its clip (`read_clip_notes`), every stem is unwarped and starts at beat 0. That last point needed the start-marker fix above, found on this run.
