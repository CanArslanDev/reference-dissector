# 07 – Building in Ableton Live

`refdissect live-build` turns `plan.json` into an Arrangement in the open Live set. It works through [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp): the MCP server and its Remote Script. The build imports the server and calls its tools directly (`mcp.call_tool`), the same tools an AI client uses.

```bash
uv sync --extra live          # pulls ableton-live-mcp 2.2.0 from GitHub
uv run refdissect live-build "$W/song/refdissect"
```

Live must be running with the AbletonMCP Remote Script selected as a control surface (see ableton-live-mcp's README). The build refuses unless every track in the set is empty. `build_tracks` adds every track in one call; empty tracks already in the set are left as they are.

## What gets built

| track | content |
|---|---|
| 808 (synth) | Simpler with the fitted one-shot, Volume 0 dB, warping off, at the plan's Transpose and Detune (whole numbers, `live.tune_808`). With Glide on, also Glide Mode "Glide", the Glide Time and one voice. An Arrangement clip holds every 808 note at its written pitch and length. |
| Drums (library) | Drum Rack: one pad per identified sound, each a Simpler at Volume 0 dB with its match's Transpose and Detune (whole cents). A pitched copy, or the same file found at another pitch, plays on a pad of its own from note 54 up. An Arrangement clip holds every hit, each note where its hit attacks. |
| other / piano / guitar / vocals (notes) | a MIDI track per tonal stem with `melodic`'s notes, Drift as a placeholder instrument, muted |
| REF drums / bass / other / piano / vocals | the stems as Arrangement audio clips from beat 0, warping off; drums, bass and vocals muted |

Glide is on only for an 808 line with three legato notes or more, and the 808 reader writes none by default ([01](01-pipeline.md#808-notes-bass808transcribe)): no plan of the suite or of PIRAMIT turns it on.

Also set:
- the song tempo, to the fitted grid (136.999 for PIRAMIT; 137.002 before replica mode);
- locators at section starts, rounded to whole beats;
- the view, to the Arrangement with the playhead at 0;
- levels: ours at the RMS of the stems they replace, everything 4 dB down for headroom.

**A/B:**
- mute *808* and *Drums*, unmute *REF drums* and *REF bass* to hear the original;
- solo pairs to compare one part.

## What the build sends

`live.build_spec(plan)` is the whole spec of one `build_tracks` call:
- **808**: `{"sample": "<song> 808 (synth).wav", "warping": false}`. Its settings come from `live.simpler_808`: `"Volume": "0.0 dB"`, `"Transpose"` and `"Detune"` as numbers, and with Glide `"Glide Mode": "Glide"` and `"Glide Time": "<ms> ms"`. The one-shot is first copied into the User Library (`Samples/AbletonMCP Generated/`), named after the song.
- **Each pad**: `{"sample": <file name>, "warping": false, "settings": {"Volume": "0.0 dB", "Transpose": …, "Detune": …}}`.
- **Melodic tracks**: `"instrument": "Drift"`, muted, with their notes.
- **Stems**: audio tracks with one clip each, warping off.
- **Faders**: `volume_db` from the plan's levels, at most +6 dB.

Every sample loads by name through Live's browser. `voices` is not a Simpler parameter but a property of the device, and `build_tracks` does not read it: after the call the build sets it with `lom_set_property`.

ableton-live-mcp's replica-mode branch adds more to `build_tracks`: files loaded by path (`sample_path`), `playback_mode`, `start_marker`, `choke_group` and the time signature. The build uses none of them yet; ableton-live-mcp 2.2.0, which it depends on, ignores `sample_path`.

## ableton-live-mcp tools used

| step | tool or command |
|---|---|
| look at the set | `get_session_overview` |
| tempo, loop | `set_song_properties` |
| every track, device, setting, clip and note | `build_tracks` (ableton-live-mcp 2.1.2+), one call with the spec above |
| the 808's voices | `lom_set_property` (`voices` on the 808 track's Simpler) |
| locators | `place_locators` |
| view | `select_in_live`, `transport` |

Until 2026-10-17 the build called `manage_track`, `add_device`, `browser_load`, `edit_clip_notes` (200 notes per call), `set_track_mixer` and the LOM directly, track by track.

Other ableton-live-mcp tools that fit this work:
- **`separate_stems`**: Demucs, the same model.
- **`audio_to_midi`**: Basic Pitch, for melodies.
- **`find_similar_sounds`**: CLAP search by description or by example.
- **`import_midi_file`**: takes `midi/drums.mid`, `midi/808.mid` and the melodic tracks' files.
- **`compare_to_reference`**: loudness, dynamics, stereo and tonal balance of the mix against a reference, with EQ Eight settings.
- **`render_and_analyze`**, **`capture_stems`**: record the set and analyze it.
- **`generate_audio`**: MusicGen, CC-BY-NC.
- **`agent_changes(undo=N)`**: undoes the last N tool calls exactly.

## Live behaviour measured here

### Live 12.4, while building PIRAMIT (2026-10-11 to 2026-10-17)

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

**Locators:** `place_locators` at fractional beats (0.0685) failed with "Live did not create it"; at whole beats (0, 96, …) it worked. The plan's locators are rounded to whole beats.

**Speed:** 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s. The Remote Script answers in about 10 ms per command, and notes go in batches of 200.

**The Trial cannot save.** Keep `plan.json`: rebuilding is one command. The plan, the one-shot and the MIDI files are the durable result, not the set.

### Live 12.4.6, a battery for replica mode (2026-10-19)

Measured in Live 12.4.6 (Trial) with Remote Script 2.1.0 and ableton-live-mcp's replica-mode `build_tracks`: values read back, and recordings at the audio device's 48 kHz.
- **Transpose keeps whole semitones.** A fraction is truncated toward zero: 11.99 plays as 11.
- **Detune keeps whole cents, −50 to +50.** A number is truncated (11.5 → 11); display text goes to the nearest value ("11.5 ct" → +12). Played pitches follow the stored values within 0.002 cents. The plan therefore writes whole numbers (`pad_pitch`, `tune_808`).
- **Velocity sets the level:** dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over the Volume, "Vol < Vel" as a fraction. It is 35 % on a Simpler made directly (`insert_device`) and 45 % on a pad loaded through the browser. Velocity 70 plays at the Volume; velocity 127 adds 7.9 dB at 35 % and 10.2 dB at 45 %. Volume set as "0.0 dB" lands on −0.005 dB.
- **Defaults depend on how the Simpler was made.** One made directly is Classic with its filter on; a pad loaded through the browser is One-Shot with the filter off. Both start at Volume −12 dB.
- **Snap**, on by default, stops a sample 10–30 ms before its end with a short fade, the same on every hit: a kick lost 28.8 of its 163.5 ms, a hat 9.9 of 71.4.
- **A choke group** cuts the sounding pad with a linear 1 ms fade from the new hit's onset.
- **Glide.** With one voice and Glide Mode "Glide", overlapping notes slide linearly in semitones over exactly the Glide Time (29.95–249.5 ms for 30–250 ms), for 2 to 12 semitones alike, without starting the sample again. Notes 25 ms apart jump.
- **Timing.** A note sounds within 0.04 ms of its time (drum hits of a 5-track build: mean +0.017 ms), and a recording adds no offset of its own.
- **The browser keeps old content.** A pad loaded through the browser by a name Live has loaded before still plays the old file after it is overwritten on disk; a file loaded by path is read again.
- A 5-track `build_tracks` call took 7.7 s.

### What this means for the build (estimates, not measured on a build)

- **Velocity.** The plan's levels and its render take velocity as a linear level (v/127): 0 dB at velocity 127, −11.2 dB at 35. By the law above, a browser-loaded pad plays +10.2 dB and −6.4 dB over its Volume. The plan's drum velocities run from 35 to 127, so in Live its loudest hits would sit about 10 dB over the plan's estimate and its softest about 5 dB. Its 808 notes run from velocity 70 to 127, around 115: about +8 dB at 45 % (+6.2 at 35 %), where the plan assumes −0.9 dB. These numbers come from the law; no build has been captured.
- **Which "Vol < Vel" the 808 gets** (a track's Simpler loaded through the browser) was not measured.
- **Snap.** The build leaves it on, so every pad and the one-shot would end 10–30 ms early.
- **The one-shot's name.** The build names the 808 file after the song, so a rebuild in a Live that loaded the old file may play the old one-shot.

## Not verified in Live

No plan of replica mode has been built and captured yet. Not verified:
- Detune on Drum Rack pads as `build_tracks` sets it (the battery read whole cents back on Simplers and set Detune by display text through `build_tracks`, but no pad with a plan's Detune has been recorded);
- the 808's Detune, Glide Mode and Glide Time sent through `build_tracks`, and its voices set to 1 with `lom_set_property` after it;
- the plan-level pitch, slide and level numbers of the suite, which assume these settings behave as the battery measured them and velocity as a linear level;
- the meter and bar 1: the plan holds no meter, and Live's bar 1 sits at 0 s. A 3/4 beat would be built in 4/4, and full's first bar line, at 0.4 s, lands 400 ms off.

## Runs

- **The first draft, by hand.** Before `live-build` existed, the PIRAMIT draft was built with a script calling the same tools: the same tracks plus a separate click track, with the snares chosen by hand. It built 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s.
- **`refdissect live-build`, into an empty Live 12.4 set** (2026-10-11): 7 tracks, 6 Drum Rack pads with the identified samples and their transpositions, 549 drum notes, 126 808 notes, 5 stems and 8 locators in 9.7 s. That includes waiting for the browser to list the new 808 file.
  - Checked afterwards: every pad holds its sample, every note is in its clip (`read_clip_notes`), every stem is unwarped and starts at beat 0. That last point needed the start-marker fix above, found on this run.
- **`live-build` on `build_tracks`, Live 12.4.2** (2026-10-17): the same plan re-run with the corrected tuning, seven tracks with no problem reported. Read back from Live: the 808's Simpler at +36 st and 0 dB with warping off, the pads at their transpositions (−3, −1, 0 st…), faders and stem mutes as planned, tempo 137, 8 locators. 4.9 s once the 808 file was listed (the first run waits for Live's browser to see it).
- **PIRAMIT with replica mode** (2026-10-22): planned, not built ([02](02-case-study-piramit.md#in-live)).
