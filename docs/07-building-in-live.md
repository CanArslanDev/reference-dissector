# 07 – Building in Ableton Live

`refdissect live-build` turns `plan.json` into an Arrangement in the open Live set. It works through [ableton-live-mcp](https://github.com/CanArslanDev/ableton-live-mcp): the MCP server and its Remote Script. The build imports the server and calls its tools directly (`mcp.call_tool`), the same tools an AI client uses.

```bash
uv sync --extra live          # pulls ableton-live-mcp 2.2.0 from GitHub
uv run refdissect live-build "$W/song/refdissect"
```

Live must be running with the AbletonMCP Remote Script selected as a control surface (see ableton-live-mcp's README). The Remote Script serves at most 8 clients at once: with 8 MCP servers of other sessions connected, Live refuses the build's connection, and `live-build` says so. The build refuses unless every track in the set is empty. `build_tracks` adds every track in one call; empty tracks already in the set are left as they are.

## What gets built

| track | content |
|---|---|
| 808 (synth) | Simpler with the one-shot (the fitted voice on the template's own periods, [05](05-808-synthesis.md#in-live)), Volume 0 dB, "Vol < Vel" 35 %, warping off, Snap and its filter off, Ve Release at Live's own 50 ms, at the plan's Transpose and Detune (whole numbers, `live.tune_808`). With Glide on, also Glide Mode "Glide", the Glide Time and one voice. An Arrangement clip holds every 808 note at its written pitch and length, each starting so that the one-shot's attack lands on the note's attack; a note that comes while the voice of its pitch still sounds starts 1 ms earlier, as Live starts it 1 ms late. |
| Drums (library) | Drum Rack: one pad per sound the plan gives one on evidence ([01](01-pipeline.md#plan)), each a Simpler with a copy of its file named by its content, at a Volume of its own (at most 0 dB), "Vol < Vel" 35 % and Snap off, with its match's Transpose and Detune (whole cents). The backbone's kick and snare take the first pads of their roles (36, 38). A pitched copy, or the same file found at another pitch, plays on a pad of its own from note 54 up. An Arrangement clip holds every hit, each note as much before its hit's attack as its pad's sample takes to attack, and 1 ms more where the hit cuts its pad's sounding hit; a pad plays each strike once. |
| other / piano / guitar / vocals (notes) | a MIDI track per tonal stem with `melodic`'s notes, Drift as a placeholder instrument, muted |
| REF drums / bass / other / piano / vocals | the stems as Arrangement audio clips from beat 0, warping off; drums, bass and vocals muted |

Glide is on only for an 808 line with three legato notes or more, and the 808 reader writes none by default ([01](01-pipeline.md#808-notes-bass808transcribe)): no plan of the suite or of PIRAMIT turns it on.

Also set:
- the song tempo, to the fitted grid (136.999 for PIRAMIT; 137.002 before replica mode), and the time signature, to the plan's meter (3/4 for meter34), before any locator;
- locators at section starts, rounded to whole beats; one that Live puts elsewhere is deleted and reported;
- the view, to the Arrangement with the playhead at 0;
- levels: each pad at a Volume of its own, read where the stems hold its sound (a kick on drums + bass less the plan's 808), the loudest pad at 0 dB; the drum and 808 tracks at the RMS of the stems they replace, the drum fader stopping where the loudest pad plays as loud as the stem holds its sound; everything 4 dB down for headroom ([01](01-pipeline.md#plan)). Live holds a pad's Volume a display step under the text sent and plays what it holds within 0.003 dB (below).

After `build_tracks` the build reads back the file each Simpler holds and compares it with the plan's.

**A/B:**
- mute *808* and *Drums*, unmute *REF drums* and *REF bass* to hear the original;
- solo pairs to compare one part.

## What the build sends

`live.build(plan)`, in order:
1. `get_session_overview`: every track must be empty. A connection Live refuses ends here, with the reason (`live.refused`: the Remote Script serves at most 8 clients).
2. `set_song_properties`: the tempo, loop off, and the plan's meter as the time signature (`live.meter_of`; 4/4 for a plan written before plans carried one).
3. The one-shot and every pad's file are copied into the User Library (`Samples/AbletonMCP Generated/`) under names of their content, unless a file there holds that content already.
4. `build_tracks`, with `live.build_spec(plan)`.
5. `place_locators`, then the cue points read back; a cue Live made elsewhere is deleted.
6. The read-back of every Simpler's file.
7. With Glide on, the 808's voices; then the view.

`live.build_spec(plan)` is the whole spec of the one `build_tracks` call:
- **808**: `{"sample": "<song> 808 (synth) <digest>.wav", "warping": false}`. `<digest>` is the first 8 hex digits of the one-shot's SHA-1 (`live.sample_808`), and the build refuses a one-shot that changed after the plan named it. Its settings come from `live.simpler_808`:
  - `"Volume": "0.0 dB"`;
  - `"Transpose"`, `"Detune"` and `"Vol < Vel"` (0.35, raw) as numbers;
  - `"Snap": "Off"`, and `"F On": "Off"`, the filter;
  - `"Ve Release": "50.0 ms"`, Live's own default, sent so that `render.py`'s release holds;
  - with Glide, `"Glide Mode": "Glide"` and `"Glide Time": "<ms> ms"`.
- **Each pad**: `{"sample": "<file name> <digest>.<ext>", "warping": false, "settings": {"Volume": "<x.y> dB", "Transpose": …, "Detune": …, "Vol < Vel": 0.35, "Snap": "Off"}}`.
  - The name is the file's own, its spaces run together, with the first 8 hex digits of its SHA-1 (`live.pad_sample`). Live's browser finds a sample by its name, and other files of a pad's name can come first (below).
  - The build refuses a pad whose file cannot be read.
  - The Volume is the pad's `volume_db`, at most 0 dB (`render.pad_settings`), as display text with one decimal: "-6.5 dB", never "-0.0 dB"; "0.0 dB" for a pad without one.
- **Melodic tracks**: `"instrument": "Drift"`, muted, with their notes.
- **Stems**: audio tracks with one clip each, warping off.
- **Faders**: `volume_db` from the plan's levels, at most +6 dB.

On the 96 plans of the suite's m5 and m5-ft, the spec passes a check written against ableton-live-mcp 2.2.0's `build_tracks` code: only keys `build_tracks` reads, no NaN, each of the 456 pads named after its own file's content, and Snap, F On and Ve Release as above. On m4's 96 plans an earlier check of the same kind found pad Volumes from −20.7 to 0.0 dB, the 808's Detune from −8 to +14 cents, and Glide on in none.

**Read-back.** After `build_tracks` the build reads the file each Simpler holds (`lom_get_object`, its sample's `file_path`) and compares it with the plan's file by the first 8 hex digits of their SHA-1s (`live.held_samples`). A Simpler that holds another file is listed under `wrong_samples` and logged.

**Meter and locators.** The time signature is set before the locators, since Live moves a locator off its grid. After `place_locators` the build reads the cue points back. A cue that is neither asked for nor one the set held before is deleted (`transport`: `jump_to` it, then `toggle_cue`) and listed under `locators_moved`, with the beat asked and the beat Live chose (`live.moved_locators`). A cue's time cannot be set through the LOM.

**When it fails.** A `build_tracks` call that raises ends the build with the track, pad and file it stopped at, and the tracks it left in the set: delete them, or open a new set, before building again (`live.failed_build`).

`voices` is not a Simpler parameter but a property of the device, and `build_tracks` does not read it: after the call the build sets it with `lom_set_property`.

ableton-live-mcp's replica-mode branch adds more to `build_tracks`: files loaded by path (`sample_path`), `playback_mode`, `start_marker`, `choke_group` and the time signature. The build uses none of them yet; ableton-live-mcp 2.2.0, which it depends on, ignores `sample_path`. Loading by path would spare the browser's search: how long the search takes for a content name was not timed.

## ableton-live-mcp tools used

| step | tool or command |
|---|---|
| look at the set, and its cue points after the locators | `get_session_overview` |
| tempo, loop, time signature | `set_song_properties` |
| every track, device, setting, clip and note | `build_tracks` (ableton-live-mcp 2.1.2+), one call with the spec above |
| locators | `place_locators` |
| a locator Live moved | `transport` (`jump_to`, then `toggle_cue`) |
| the file each Simpler holds | `lom_get_object` (its sample's `file_path`) |
| the 808's voices, with Glide on | `lom_set_property` (`voices` on the 808 track's Simpler) |
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
- **Loads samples at −12 dB.** The build sets 0 dB through the display value ("0.0 dB"), so the offline level estimate holds. A pad's own Volume goes the same way ("-6.5 dB"); Live holds it a display step under the text (2026-10-29, below).
- **Plays a sample at its own pitch on C3 (MIDI 60).** To keep real note numbers, set Transpose to 60 − root; Transpose reaches ±48 semitones.

**Browser:**
- **New files.** A file written into the User Library (here `Samples/AbletonMCP Generated/`) is listed by the browser within a few seconds. Until then `browser_search` does not find it, so the build retries for up to 40 s.
- **Pads.** Samples load onto Drum Rack pads through the browser: select the pad, then `load_item`. A file elsewhere on disk has to be in the User Library or a Places folder first. The browser finds a file by its name, and another file of that name can come first (2026-10-29, below).

**Stems:** `create_audio_clip` places a file at a beat position on an audio track; with `warping` off, the clip plays at its own speed.

**Live moves a new audio clip's start.** Its warp analysis puts the clip's start at the first transient: 21–30 ms on four stems, a few seconds on the `other` stem, whose intro is quiet. The start stays there after warping is turned off, so the stem plays that much early against the notes.
- **Fix.** Set the clip's `loop_start` to 0, which carries `start_marker` with it. The build does this and reports each stem's `start_marker` (0) and length.
- **Check.** Unwarped at 137 BPM, PIRAMIT's 126.17 s stems are then 288.09 beats long. Before the fix, `other` was 279.86 beats long.

**Locators:** `place_locators` at fractional beats (0.0685) failed with "Live did not create it"; at whole beats (0, 96, …) it worked. The plan's locators are rounded to whole beats. A whole beat off the Arrangement's grid moves too (2026-10-29, below).

**Speed:** 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s. The Remote Script answers in about 10 ms per command, and notes go in batches of 200.

**The Trial cannot save.** Keep `plan.json`: rebuilding is one command. The plan, the one-shot and the MIDI files are the durable result, not the set.

### Live 12.4.6, a battery for replica mode (2026-10-19)

Measured in Live 12.4.6 (Trial) with Remote Script 2.1.0 and ableton-live-mcp's replica-mode `build_tracks`: values read back, and recordings at the audio device's 48 kHz.
- **Transpose keeps whole semitones.** A fraction is truncated toward zero: 11.99 plays as 11.
- **Detune keeps whole cents, −50 to +50.** A number is truncated (11.5 → 11); display text goes to the nearest value ("11.5 ct" → +12). Played pitches follow the stored values within 0.002 cents. The plan therefore writes whole numbers (`pad_pitch`, `tune_808`).
- **Velocity sets the level:** dB = 0.4 × "Vol < Vel" × (velocity − 70) − 0.096 over the Volume, "Vol < Vel" as a fraction. It is 35 % on a Simpler made directly (`insert_device`) and 45 % on a pad loaded through the browser. Velocity 70 plays 0.096 dB under the Volume; velocity 127 adds 7.9 dB at 35 % and 10.2 dB at 45 %. Volume set as "0.0 dB" lands on −0.005 dB.
- **Defaults depend on how the Simpler was made.** One made directly is Classic with its filter on; a pad loaded through the browser is One-Shot with the filter off. Both start at Volume −12 dB.
- **Snap**, on by default, stops a sample 10–30 ms before its end with a short fade, the same on every hit: a kick lost 28.8 of its 163.5 ms, a hat 9.9 of 71.4.
- **A choke group** cuts the sounding pad with a linear 1 ms fade from the new hit's onset.
- **Glide.** With one voice and Glide Mode "Glide", overlapping notes slide linearly in semitones over exactly the Glide Time (29.95–249.5 ms for 30–250 ms), for 2 to 12 semitones alike, without starting the sample again. Notes 25 ms apart jump.
- **Timing.** A note sounds within 0.04 ms of its time (drum hits of a 5-track build: mean +0.017 ms), and a recording adds no offset of its own.
- **The browser keeps old content.** A pad loaded through the browser by a name Live has loaded before still plays the old file after it is overwritten on disk; a file loaded by path is read again.
- A 5-track `build_tracks` call took 7.7 s.

### Live 12.4.6, eight m4 plans and PIRAMIT's built and recorded (2026-10-29)

The code of 74931ce. Eight plans of the suite's m4 (bass_low-1, glides-1, dynamics-1, full-1, v1orig-3, swing66-1, meter34-1 and drum_pitch-1, htdemucs), two probes and PIRAMIT's plan were built with the calls `live.build` makes and recorded with `capture_stems` at the audio device's 48 kHz, each pad on a copy of the Drums track with the other pads muted (the copies sum to the Drums track within −120 dB). Probe A put one kick file on 7 pads at 0 to −30 dB, played at 7 velocities, and the 808 one-shot at Detune −23 with Glide. `refdissect live-build` itself could not connect: Live's Remote Script serves at most 8 clients, and 8 MCP servers of other sessions were connected ("Connection reset by peer"). The calls went through the check's own session, whose ableton-live-mcp 2.2.0 is the same code.

**Notes and the recording.** Live held every note sent, at its start, pitch and velocity (read back at 480 PPQ on every build). 9 notes at the plan's 0.05-beat floor were shortened to the gap before the next note of their pitch (0.021–0.044 beat). A stem clip recorded in the same pass lined up with its file within 0.0002 samples and played its −4.0 dB fader within 0.004 dB.

**A pad plays one voice.** A pad loaded through the browser is One-Shot, with 3 voices and retrigger on.
- A hit that comes while its pad's hit before still plays fades that one out along a straight line over 1 ms from its note, and starts late: 391 hits on 18 pads, a median 1.000 ms (98 % within 0.95–1.05 ms).
- Which hits do depends on how long Live plays the voice, not on the file: 13 hits that came inside their file but after Snap had ended the voice started on time.
- Where most of a role's hits do this, its timing moves about 1 ms: 28 of dynamics-1's 38 snare hits, 15 of meter34-1's 16, 31 of v1orig-3's 56 kicks.
- At bass_low-1's kick rolls Live played up to 5.6 dB under `render.py`'s render, which summed a pad's hits, and every one of its 66 kick hits within 0.29 dB of a render on one voice.
- Hits clear of the one before sounded 0.005–0.035 ms after `render.py`'s (medians per pad).

**Snap**, on by default, ended pads before their files did: v1orig-3's snare pad 19.1 ms early (16 hits); PIRAMIT's snare file 25.4 and 26.9 ms early on its two pads, its unmatched snare 4.7 ms. On glides-1's kick it started 21.8 ms before the file's end, where the tail lay 51.9 dB under its peak; glides-1 scored the same with Snap off.

**The 808's Simpler**, loaded through the browser, was Classic, with 6 voices, retrigger on, Snap on, Ve Release 50 ms and its filter on (lowpass at 22.0 kHz, "Filt < Key" 100 %).
- **The filter** delayed every note by 0.13–0.46 ms, the lower the note the more (0.46 ms at MIDI 12, 0.13 ms at 36), and took 0.14 dB off the 8th harmonic, 0.55 dB off the 16th and 2.2 dB off the 32nd, against the fundamental. swing66-1 has three notes at MIDI 12 that the plan puts 19.6–19.9 ms from the truth; that delay takes them past 20 ms (its 808 F1@20 0.944 → 0.889 with the measured lags).
- **The release.** After a note's end the voice played −4.0 dB over 0–5 ms, −17.4 dB over 5–20 ms and −45 dB over 20–50 ms, against the note's last 50 ms, and nothing after 50 ms (probe A, isolated notes). It falls 1.683 dB a ms, a straight line in dB within 0.44 dB, and stops at −60 dB, 35.7 ms after the end; −40 dB comes at 23.8 ms. `render.py` faded a note out in 5 ms.
- **The restart.** A note that came while the voice of a note of its pitch still sounded started 0.94–1.05 ms after the other notes of its pitch (110 notes on 3 beats), over the filter's delay.

**Pitch.** The 808 played at `render.py`'s pitch within 0.11 cents on every beat (Detune −8, −1 and +1), within 0.03 cents over whole long notes. Pads at Transpose −1 and Detune +30, or at Detune +25, played the plan's pitch within 0.1 cent.

**Pad Volume.** A Volume sent as "−6.6 dB" was held as −6.605 dB, and from −10 dB on a display step lower ("−14.9 dB" held −14.95). A hit played the held value within 0.003 dB.

**Velocity.** "Vol < Vel" read back 0.35, as sent. A hit played 0.4 × 0.35 × (velocity − 70) − 0.096 dB over its held Volume, from velocity 35 (−4.90 dB against velocity 70) to 127 (+7.98 dB), on the pads and on the 808 (probe A).

**Glide** (probe A). Glide Mode "Glide", Glide Time "120 ms" and Detune −23 read back as sent, and voices went from 6 to 1 with `lom_set_property`. Over a legato chain Live's pitch stayed within 1.04 cents (p90) of `render.py`'s. Notes 25 ms apart started the sample again.

**Simpler loses some highs.** Against its own file at the held Volume, a 48 kHz hat lost 0.14 dB at 4–8 kHz, 0.32 dB at 8–12 kHz and 0.9–2.5 dB above 12 kHz. 44.1 kHz files lost more: about 0.5 dB at 8–12 kHz, 1.6–1.8 dB at 12–16 kHz and 3.1–3.7 dB at 16–18 kHz. A stem clip recorded the same way is flat within 0.02 dB to 12 kHz. `render.py` does not model this.

**The one-shot's digest name works.** A one-shot under a new digest name played the new file; loaded again by its old name after that file was overwritten, Live still played the old content.

**Pads loaded by name.** `build_tracks` (2.2.0) loads a pad's sample through Live's browser by its file name.
- v1orig-3's open-hat pad got an 808 of the same name from another folder: 14.9 dB louder over the whole hit than its own file.
- PIRAMIT's other hat (Transpose +1) got an FX hit of the same name.
- One of PIRAMIT's snares has a one-character name, which so many items match that its file was not among the 20 the search keeps. After 40 s `build_tracks` raised, and the build stopped after the 808 track, leaving an unnamed empty Drum Rack.
- 4 of the 466 pads in the 97 plans of m4, m4-ft and PIRAMIT named a file that other files of other content share.

**Meter and locators.** meter34-1's 3/4 plan was built in 4/4, Live's bar 1 at 0 s. Asked for locators at beats 9, 21 and 33, Live made cues at 8, 20 and 32, and `place_locators` reported "Live did not create it"; with 3/4 set, beat 9 went to 12.

**The check's own scores.** Its plan-level "Live" scores (drum F1, 808 note F1, cents, timing) were the plan's estimate moved by the lags measured on Live's audio, not read off the audio, so "unchanged" followed largely from how they were made; v1orig-3's higher drum F1 came from the wrong file's hits being dropped as silent. Its level scores were read off the recordings: against `render.py`, levels.vs_kit_abs_db moved by −0.25 to +0.13 dB and levels.balance_abs_db by −0.22 to +0.38 dB over the eight beats. The second check read every score off the audio.

### Live 12.4.6, four m5 plans and PIRAMIT's rebuilt and recorded (2026-10-29, second check)

The code of the live/build branch (60b485f), merged as a4f2358: its plans are m5's. Four of the suite's m5 plans (v1orig-3, meter34-1, bass_low-1 and dynamics-1, htdemucs) and PIRAMIT's plan were built with `live.build`'s calls and recorded at 48 kHz as above (the pad copies sum to the Drums track within −123 dB; the stem clip lined up within 0.0002 samples and played −4.0 dB within 0.004 dB). `live-build` was refused again, now with its own message: Live held 8 clients. So the calls went through the check's own session again, and the build's read-back and locator deletion ran as the same commands. PIRAMIT's spec went as 5 `build_tracks` calls in track order: its 1,881 notes did not fit one message from that session.

Each recording was read the way the suite reads a rebuild, and `render.py`'s render of the same plan the same way:
- the drum hits: found on each pad's track by the pad's own sample as a matched filter, with no plan times;
- the 808: note by note against `render.py`'s, and by the pipeline's 808 reader;
- the levels: `evaluate.level_scores` on the recordings.

**Every Simpler holds its plan's file.** All 30 (5 one-shots and 25 pads) hold a file whose SHA-1 is the plan's file's, the three pads that had played other files or none among them. PIRAMIT built whole: the 808 (101 notes), 10 pads (630 notes), 3 muted Drift tracks (1,150 notes), 5 stems and 8 locators. Live held all 2,963 notes at their starts, pitches and velocities; it shortened 2 notes at the 0.05-beat floor.

**Settings read back as sent** (5 one-shots and 12 pads read). Snap and F On off, Ve Release 50.0 ms, "Vol < Vel" 0.35, Transpose and Detune; the 808's Simpler Classic with 6 voices, the pads One-Shot with 3; pad Volumes a display step under the text, as before.

**Drum hits play where `render.py` plays them.**
- All 1,320 planned hits were found (at NCC 0.93 or more), within 0.04 ms of `render.py`'s.
- The 253 that cut their pad's sounding hit started 0.004–0.019 ms after `render.py`'s (medians per pad), which starts them 1 ms late. Against the render before one voice per pad they had been about 1 ms late.
- With Snap off a hit plays its whole file: its end lies within 0.02 ms of `render.py`'s on kicks and snares, where Snap had cut 4.7 to 27 ms.
- Drum F1@20 per role, read off Live's audio, equals `render.py`'s and the suite's on all four beats (v1orig-3 0.654, meter34-1 0.944, bass_low-1 0.721, dynamics-1 0.966). Per-role timing medians lie within 0.03 ms of `render.py`'s.

**The 808 matches `render.py` note by note.** Over 408 notes, Live less `render.py` leaves a residual of −42.7 to −55 dB (medians per beat; the worst note −34.6 dB), and every note starts within 0.11 ms of `render.py`'s, the 186 that restart a note of their pitch among them. With the filter off no delay by pitch is left. Notes that end in a rest fall as `render.py`'s release does: −40 dB 25 ms after the end on PIRAMIT (20 notes) in both, 23–27 ms on the suite's beats.

**Levels differ only by Simpler's top-end loss.** Scored as the suite scores them, on the recordings and on `render.py`'s render:

| | v1orig-3 | meter34-1 | bass_low-1 | dynamics-1 |
|---|---|---|---|---|
| distance from a perfect kit, Live / `render.py` | 0.57 / 0.54 dB | 0.76 / 0.88 dB | 0.63 / 0.51 dB | 2.04 / 2.29 dB |
| balance, Live / `render.py` | 0.57 / 0.17 dB | 0.19 / 0.31 dB | 0.54 / 0.51 dB | 0.48 / 0.29 dB |

Kicks played within 0.004 dB of `render.py` and the 808 within 0.022 dB; snares 0.03–0.16 dB and hats 0.06–0.97 dB under it. v1orig-3's bright 48 kHz hat, 0.97 dB under, sets its balance.

**The 808 reader on a plan's own 808.** It reads Live's 808 as it reads `render.py`'s: note F1@50 0.85 against 0.86 on v1orig-3, the same on the other three. It does not read it as the plan's estimate does. On bass_low-1 it reads the 64 shortest notes (110 ms) 63–68 cents flat in both, note F1@50 0.59, where the estimate gives 1.0. With m4's notes, before the restarts start 1 ms early, it read 6 of the 156 notes more than 50 cents off; on the truth's own 808 stem it misreads 92 of them, so which is right is not settled. The suite's plan-level 808 scores come from the estimate.

**The meter.** meter34-1 was built in 3/4 (read back), Live's bar 1 at 0 s. Two of its five locators landed where planned, at beats 0 and 48. Live made the other three (9, 21 and 33) at 12, 24 and 36, and the build deleted and reported them. A new cue goes to the nearest line of the Arrangement's grid, which follows its zoom (6 beats at the zoom found); zoomed in four steps (`zoom_view`), beat 9 landed exactly. In 4/4, 24 of 24 locators on the other four builds landed where planned.

### What the build does with it

- **Snap is off** on every Simpler, and **the 808's filter** too. Measured on the second check: hits play their whole files, and no 808 note is delayed by its pitch.
- **The 808's release stays at Live's 50 ms.** It is sent explicitly, and `render.py` and the level estimate play it (`render._release`). It was chosen over a short release for two reasons: Live was measured at 50 ms only, and on PIRAMIT's 20 plan notes that end in a rest the bass stem holds a median −7.5 dB over the next 40 ms (against the note's last 20 ms), where Live's release gives −7.9 dB, `render.py`'s old 5 ms fade −10.7 and the same law at 5 ms −18.3 (the last one extrapolated, not measured).
- **One voice per pad, and the 808's restart.** `render_plan`, the level estimate and `estimate_from_plan` play a hit or 808 note that comes while its voice still sounds as Live does: the old voice fades over 1 ms and the new one starts 1 ms late. `make_plan` starts each such note 1 ms early, so Live sounds it where the analysis heard it: on PIRAMIT 98 of the 630 drum notes and 62 of the 101 808 notes.
- **`pad_levels` still reads a pad against its own hits summed.** Read against the hits cut as Live plays them, the suite's plans sat 1.30 dB from a perfect kit instead of 1.20 (htdemucs_ft 1.27 instead of 1.19): a stem keeps a sound's tail under its next hit, as the suite's truth sums them.
- **Pads load by content, and the build checks them.** Each pad's file goes into the User Library under a name of its content, and every Simpler's file is read back.
- **The plan carries its meter**, which the build sets before the locators; a locator Live still moves is deleted, not kept where Live put it.
- **On the suite** (plan and score again, m4 → m5, htdemucs_ft in brackets; [12](12-replica-mode.md#what-each-stage-measured) has the rest): distance from a perfect kit 1.247 → 1.198 dB [1.231 → 1.152], balance 0.552 → 0.512 dB [0.497 → 0.476]; the plan's meter right 0.909 → 0.955 and its downbeat F 0.892 → 0.924; every F1 the same. m4's own plans, played as Live plays them, would sit at 1.259 [1.242] and 0.566 [0.506] dB.

## Not verified in Live

- **`refdissect live-build` end to end.** Both checks found Live serving 8 clients and sent the build's calls through their own session. The package's read-back, locator deletion and failure messages ran as the same commands, not as its code.
- **One `build_tracks` call of PIRAMIT's size**, which went as 5, and **a build that stops half way**: no build of the second check failed, so the failure message has not been seen.
- **How long the browser takes** to find a pad's content name: not timed.
- **htdemucs_ft plans**, **a plan with Glide** (only a probe glided; the restart is not modelled for one voice), and **choke groups**.
- **Locators off the Arrangement's grid.** Live moves them, and the build deletes them. Zoomed in, one probe placed such a locator; that is not in the build. 11 of the 169 locators of the suite's m5 and m5-ft plans lie off Live's bar lines (full and tempo72, whose first bar line is not at 0 s) and are likely to be deleted the same way; they were not built.
- **Bar 1.** Live's bar 1 sits at 0 s: full's first bar line, 0.4 s in, lands 400 ms off it.
- **Simpler's top-end loss** per sample rate and transposition, as a curve. `render.py` does not model it: hats played up to 0.97 dB and snares up to 0.16 dB under `render.py`.
- **PIRAMIT against the song.** It has no truth: Live was measured against `render.py`, and nothing was checked by ear.

## Runs

- **The first draft, by hand.** Before `live-build` existed, the PIRAMIT draft was built with a script calling the same tools: the same tracks plus a separate click track, with the snares chosen by hand. It built 8 tracks, 565 notes, 5 stems and 5 locators in 4.6 s.
- **`refdissect live-build`, into an empty Live 12.4 set** (2026-10-11): 7 tracks, 6 Drum Rack pads with the identified samples and their transpositions, 549 drum notes, 126 808 notes, 5 stems and 8 locators in 9.7 s. That includes waiting for the browser to list the new 808 file.
  - Checked afterwards: every pad holds its sample, every note is in its clip (`read_clip_notes`), every stem is unwarped and starts at beat 0. That last point needed the start-marker fix above, found on this run.
- **`live-build` on `build_tracks`, Live 12.4.2** (2026-10-17): the same plan re-run with the corrected tuning, seven tracks with no problem reported. Read back from Live: the 808's Simpler at +36 st and 0 dB with warping off, the pads at their transpositions (−3, −1, 0 st…), faders and stem mutes as planned, tempo 137, 8 locators. 4.9 s once the 808 file was listed (the first run waits for Live's browser to see it).
- **PIRAMIT with replica mode** (2026-10-22): planned, not built ([02](02-case-study-piramit.md#in-live)).
- **PIRAMIT with replica mode's fixes** (2026-10-29, the m3 code): planned, not built. 630 drum notes on 10 pads (607 on 7 on 2026-10-22), 101 808 notes at Transpose +36 and Detune +2, 8 locators. Three sounds changed pad number (two because the backbone is planned first, one because the new hat took its pad by NCC), so a set built by hand from the 2026-10-22 plan would not line up with it.
- **PIRAMIT at 74931ce** (`fit-808` and `plan` again on the same analysis): the same notes and pads, each pad at a Volume of its own, −12.1 to 0 dB under a −12.5 dB drum fader (−20.2 dB, every pad at 0 dB, with the m3 code); the 808 on the one-shot that follows its template's periods, at Transpose +36 and Detune +1, its notes starting at their onsets ([02](02-case-study-piramit.md#in-live)). The plan's tracks pass the format check against ableton-live-mcp 2.2.0's `build_tracks`.
- **The first live check** (2026-10-29, the code of 74931ce, Live 12.4.6): eight m4 plans of the suite, two probes and PIRAMIT's plan, built with `live.build`'s calls and recorded (above). PIRAMIT's build stopped at a pad whose file the browser did not list; its drums were built again without that pad and recorded, and its melodic tracks and stems were not built.
- **PIRAMIT at the live/build merge** (a4f2358, `plan` again on the same analysis): 4/4 over 292 beats and the 8 locators as before; 98 of the 630 drum notes and 62 of the 101 808 notes start 1 ms earlier; the kick's pad at −8.1 dB (−8.5) under a −12.6 dB drum fader (−12.5).
- **The second live check** (2026-10-29, the code of 60b485f, Live 12.4.6): four m5 plans and PIRAMIT's plan of a4f2358, built and recorded (above). PIRAMIT's plan built whole, in 5 `build_tracks` calls, and every Simpler held its plan's file.
