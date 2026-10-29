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
| 808 (synth) | Simpler with the one-shot (the fitted voice on the template's own periods, [05](05-808-synthesis.md#in-live)), Volume 0 dB, "Vol < Vel" 35 %, warping off, at the plan's Transpose and Detune (whole numbers, `live.tune_808`). With Glide on, also Glide Mode "Glide", the Glide Time and one voice. An Arrangement clip holds every 808 note at its written pitch and length, each starting so that the one-shot's attack lands on the note's attack. |
| Drums (library) | Drum Rack: one pad per sound the plan gives one on evidence ([01](01-pipeline.md#plan)), each a Simpler at a Volume of its own (at most 0 dB) and "Vol < Vel" 35 % with its match's Transpose and Detune (whole cents). The backbone's kick and snare take the first pads of their roles (36, 38). A pitched copy, or the same file found at another pitch, plays on a pad of its own from note 54 up. An Arrangement clip holds every hit, each note as much before its hit's attack as its pad's sample takes to attack, and a pad plays each strike once. |
| other / piano / guitar / vocals (notes) | a MIDI track per tonal stem with `melodic`'s notes, Drift as a placeholder instrument, muted |
| REF drums / bass / other / piano / vocals | the stems as Arrangement audio clips from beat 0, warping off; drums, bass and vocals muted |

Glide is on only for an 808 line with three legato notes or more, and the 808 reader writes none by default ([01](01-pipeline.md#808-notes-bass808transcribe)): no plan of the suite or of PIRAMIT turns it on.

Also set:
- the song tempo, to the fitted grid (136.999 for PIRAMIT; 137.002 before replica mode);
- locators at section starts, rounded to whole beats;
- the view, to the Arrangement with the playhead at 0;
- levels: each pad at a Volume of its own, read where the stems hold its sound (a kick on drums + bass less the plan's 808), the loudest pad at 0 dB; the drum and 808 tracks at the RMS of the stems they replace, the drum fader stopping where the loudest pad plays as loud as the stem holds its sound; everything 4 dB down for headroom ([01](01-pipeline.md#plan)). No pad Volume under 0 dB has been built or captured in Live.

**A/B:**
- mute *808* and *Drums*, unmute *REF drums* and *REF bass* to hear the original;
- solo pairs to compare one part.

## What the build sends

`live.build_spec(plan)` is the whole spec of one `build_tracks` call:
- **808**: `{"sample": "<song> 808 (synth) <digest>.wav", "warping": false}`. Its settings come from `live.simpler_808`: `"Volume": "0.0 dB"`, `"Transpose"`, `"Detune"` and `"Vol < Vel"` (0.35, raw) as numbers, and with Glide `"Glide Mode": "Glide"` and `"Glide Time": "<ms> ms"`. The one-shot is first copied into the User Library (`Samples/AbletonMCP Generated/`), named after the song and its content: `<digest>` is the first 8 hex digits of the file's SHA-1 (`live.sample_808`). The build refuses a one-shot that changed after the plan named it.
- **Each pad**: `{"sample": <file name>, "warping": false, "settings": {"Volume": "<x.y> dB", "Transpose": …, "Detune": …, "Vol < Vel": 0.35}}`. The Volume is the pad's `volume_db`, at most 0 dB (`render.pad_settings`), as display text with one decimal: "-6.5 dB", never "-0.0 dB"; "0.0 dB" for a pad without one.
- **Melodic tracks**: `"instrument": "Drift"`, muted, with their notes.
- **Stems**: audio tracks with one clip each, warping off.
- **Faders**: `volume_db` from the plan's levels, at most +6 dB.

On the 96 plans of the suite's m4 and m4-ft, the spec passes a check written against ableton-live-mcp 2.2.0's `build_tracks` code (the code read, not run; nothing built in Live): valid JSON with no NaN, unique track names, only keys `build_tracks` reads, pad Volumes from −20.7 to 0.0 dB, the 808's Simpler at "0.0 dB", its Detune from −8 to +14 cents, and Glide on in none.

Every sample loads by name through Live's browser. `voices` is not a Simpler parameter but a property of the device, and `build_tracks` does not read it: after the call the build sets it with `lom_set_property`.

ableton-live-mcp's replica-mode branch adds more to `build_tracks`: files loaded by path (`sample_path`), `playback_mode`, `start_marker`, `choke_group` and the time signature. The build uses none of them yet; ableton-live-mcp 2.2.0, which it depends on, ignores `sample_path`.

## ableton-live-mcp tools used

| step | tool or command |
|---|---|
| look at the set | `get_session_overview` |
| tempo, loop | `set_song_properties` |
| every track, device, setting, clip and note | `build_tracks` (ableton-live-mcp 2.1.2+), one call with the spec above |
| the 808's voices, with Glide on | `lom_set_property` (`voices` on the 808 track's Simpler) |
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
- **Loads samples at −12 dB.** The build sets 0 dB through the display value ("0.0 dB"), so the offline level estimate holds. It now sets each pad to a Volume of its own the same way ("-6.5 dB"); only 0 dB has been read back.
- **Plays a sample at its own pitch on C3 (MIDI 60).** To keep real note numbers, set Transpose to 60 − root; Transpose reaches ±48 semitones.

**Browser:**
- **New files.** A file written into the User Library (here `Samples/AbletonMCP Generated/`) is listed by the browser within a few seconds. Until then `browser_search` does not find it, so the build retries for up to 40 s.
- **Pads.** Samples load onto Drum Rack pads through the browser: select the pad, then `load_item`. A file elsewhere on disk has to be in the User Library or a Places folder first.

**Stems:** `create_audio_clip` places a file at a beat position on an audio track; with `warping` off, the clip plays at its own speed.

**Live moves a new audio clip's start.** Its warp analysis puts the clip's start at the first transient: 21–30 ms on four stems, a few seconds on the `other` stem, whose intro is quiet. The start stays there after warping is turned off, so the stem plays that much early against the notes.
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

- **Velocity.** The plan took velocity as a linear level (v/127): 0 dB at velocity 127, −11.2 dB at 35. By the law above, a browser-loaded pad plays +10.2 dB and −6.4 dB over its Volume. Rendered by the law, with the pads at 45 % and the 808 at 35 % (which "Vol < Vel" a track's Simpler loaded through the browser gets was not measured), m2g's 48 plans sit 9.30 dB from a perfect kit, against the 3.67 dB they scored under v/127. The build now sets "Vol < Vel" to 35 % on every Simpler, and the velocities invert the law ([01](01-pipeline.md#plan)): 3.67 dB again (m2g-vel). These numbers come from the law; no build has been captured, and "Vol < Vel" set through `build_tracks` (a raw value, as the battery set it through the LOM) is not checked in Live.
- **Pad Volumes.** With every pad at 0 dB, m3's plans sat 4.09 dB from a perfect kit (3.67 at m2g-vel): the open hats the pad gate adds played a median 11.4 dB over it. Each pad now has a Volume of its own, −19.7 to 0 dB on the suite (htdemucs_ft −20.7 to 0), and m4's plans sit 1.25 dB from a perfect kit (1.23). That assumes Simpler plays a Volume under 0 dB as `render.py` does, a gain in dB. No Volume under 0 dB has been sent through `build_tracks`, read back or recorded, and how Simpler plays one was not measured.
- **The 808 one-shot.** It now follows its template's own periods and starts each note at its onset (m4). `render.py` plays it from its start, as Simpler is assumed to; the one-shot itself has not been loaded or recorded in Live.
- **Early notes.** A drum note starts as much before its hit's attack as the pad's sample takes to attack, read from the file as `render.py` plays it (from its start, at its Transpose and Detune). In Live that holds only as far as Simpler plays the file the same way, which the battery measured for timing (within 0.04 ms) but not for each pad's sample.
- **Snap.** The build leaves it on, so every pad and the one-shot would end 10–30 ms early.
- **The one-shot's name.** Named after the song alone, a one-shot fitted again reached the User Library under the name Live had loaded, so a rebuild in that Live could play the old one-shot. Its name now carries its content's digest, so a changed one-shot is a new file to Live (not checked in Live).

## Not verified in Live

No plan of replica mode has been built and captured yet. Not verified:
- Detune on Drum Rack pads as `build_tracks` sets it (the battery read whole cents back on Simplers and set Detune by display text through `build_tracks`, but no pad with a plan's Detune has been recorded);
- each pad's Volume under 0 dB (−19.7 to 0 dB on the suite), sent as display text through `build_tracks`;
- the 808 one-shot on its template's periods, and each 808 note starting at its onset;
- the 808's Detune, Glide Mode and Glide Time sent through `build_tracks`, and its voices set to 1 with `lom_set_property` after it;
- the plan-level pitch, slide, timing and level numbers of the suite, which assume these settings behave as the battery measured them, velocities and the notes' early starts included;
- "Vol < Vel" at 35 % on every pad and on the 808, as `build_tracks` sends it;
- the meter and bar 1: the plan holds no meter, and Live's bar 1 sits at 0 s. A 3/4 beat would be built in 4/4, and full's first bar line, at 0.4 s, lands 400 ms off.

## Runs

- **The first draft, by hand.** Before `live-build` existed, the PIRAMIT draft was built with a script calling the same tools: the same tracks plus a separate click track, with the snares chosen by hand. It built 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s.
- **`refdissect live-build`, into an empty Live 12.4 set** (2026-10-11): 7 tracks, 6 Drum Rack pads with the identified samples and their transpositions, 549 drum notes, 126 808 notes, 5 stems and 8 locators in 9.7 s. That includes waiting for the browser to list the new 808 file.
  - Checked afterwards: every pad holds its sample, every note is in its clip (`read_clip_notes`), every stem is unwarped and starts at beat 0. That last point needed the start-marker fix above, found on this run.
- **`live-build` on `build_tracks`, Live 12.4.2** (2026-10-17): the same plan re-run with the corrected tuning, seven tracks with no problem reported. Read back from Live: the 808's Simpler at +36 st and 0 dB with warping off, the pads at their transpositions (−3, −1, 0 st…), faders and stem mutes as planned, tempo 137, 8 locators. 4.9 s once the 808 file was listed (the first run waits for Live's browser to see it).
- **PIRAMIT with replica mode** (2026-10-22): planned, not built ([02](02-case-study-piramit.md#in-live)).
- **PIRAMIT with replica mode's fixes** (2026-10-29, the m3 code): planned, not built. 630 drum notes on 10 pads (607 on 7 on 2026-10-22), 101 808 notes at Transpose +36 and Detune +2, 8 locators. Three sounds changed pad number (two because the backbone is planned first, one because the new hat took its pad by NCC), so a set built by hand from the 2026-10-22 plan would not line up with it.
- **PIRAMIT at the final code** (74931ce, `fit-808` and `plan` again on the same analysis): planned, not built. The same notes and pads, each pad at a Volume of its own, −12.1 to 0 dB under a −12.5 dB drum fader (−20.2 dB, every pad at 0 dB, with the m3 code); the 808 on the one-shot that follows its template's periods, at Transpose +36 and Detune +1, its notes starting at their onsets ([02](02-case-study-piramit.md#in-live)). The plan's tracks pass the format check against ableton-live-mcp 2.2.0's `build_tracks`.
