# Auto-Fisher — Islands Reborn

A **Windows Python prototype** that recognizes fishing objects from screen pixels and clicks to cast/reel. Version 3: cleaner fish candidates and a faster local bite-check path, retaining the daylight/night lighting and rod improvements from version 2. No game-memory access, injection, network manipulation, or anti-cheat bypass. Use only where the game/server rules permit automation.

**Preview is the default. Mouse input requires `--enable-clicks` AND pressing F8 while the selected game window is focused.** This is a source-code release, not a prebuilt EXE. Python and dependency installation are required.

## What changed in version 3

- **Cleaner fish candidates:** require a filled, locally darker body rather than
  an oval outline alone. Reject avatar-edge candidates, long thin shapes, and
  distant candidates outside the cast search area. A calibrated pond rectangle
  further reduces scenery false positives. These are still heuristic detections.
- **Independent bite checks:** full-scene recognition runs in a background
  worker at a target 10 scans/second. The fresh-frame tackle/bubble loop targets
  60 FPS without waiting for each scan. Only the newest pending frame is kept.
- **Freshness safeguards:** cached fish positions are not treated as new
  observations; stale/out-of-order scene results cannot authorize actions.
  Current pixels check the visible line before extending tackle evidence.
- **Less display overhead:** overlay redraws are capped at 10 FPS by default.
- **Frame-rate-aware bubble background:** the background adaptation keeps the
  same time constant as polling increases; two-frame bite confirmation remains.
- **Timing logs:** achieved loop FPS, capture duration, scene-scan duration,
  result age, and decision duration; dispatched clicks also log local dispatch
  and click-call durations. These do not measure server response or network ping.

This reduces avoidable **local processing delay**, not Roblox/server/network lag.
The 60 FPS setting is a target, not a guaranteed reaction time. Preview first.
For troubleshooting, these commands use the virtual environment created by setup:

```powershell
# Reduce overlay overhead without changing the input safeguards
.venv\Scripts\python main.py --enable-clicks --no-view
# Return to a blocking scene loop for comparison
.venv\Scripts\python main.py --no-click --sync-vision --fps 30
```

If CPU load is high, try `--vision-fps 5` or `--fps 30`; slower scene updates
can also delay fish readiness. The log is `logs/session.jsonl`. If bites still
miss, keep the log and a short recording showing the game's FPS/ping statistics.
Local timings can identify local stalls but cannot prove a network cause.

## Retained version-2 lighting and rod improvements

- **Automatic lighting compensation:** a neutral reference from the supplied
  white avatar estimates bounded exposure and BGR color gains under warm lamps
  and lower light. No correction is made without a usable reference.
- **Stronger rod recognition:** a narrow brown-color channel plus local Lab
  chroma ridges, searched only near the avatar's hands. Pole/string junction
  geometry refines overextended Hough tips.
- **Dark string recognition:** gradient line segments recover strings that
  share the nighttime pond's hue. Only tightly collinear, short-gap fragments
  are joined when fish cover the string.
- **Occlusion-safe bite anchors:** a previously observed float is not moved
  up to a shortened string fragment. Fish proximity still does not reel by itself.
- **Dim-bubble sensitivity:** bounded local contrast, brightness and vote thresholds adapt in darker tackle ROIs. Daylight keeps the previous thresholds.
- **Exposure-change veto:** uniform local brightness jumps relearn the bubble
  background rather than authorize a reel.
- **Late-flash guard:** a brightness burst after the cast disappears cannot trigger a reel without recent tackle evidence.
- **Conservative cast readiness:** requires a nearly upright pole; uncertain
  or leaning-pole detections do not cast blindly.

Automatic lighting correction is on by default. The overlay and JSONL logs
show scene brightness, applied gains and whether correction was used.
`--no-adaptive-lighting` disables the correction for troubleshooting; saved
profiles also accept `"adaptive_lighting": false`.

The old fixed "night" color overrides are no longer used. Dim or warm frames
are corrected in pixel space before recognizing objects; bubble timing is
still measured from raw gameplay pixels with a local temporal background.

## Quick start on Windows

1. Install **64-bit Python 3.11 or 3.12**.
2. Extract the entire ZIP to a folder you can write to.
3. Double-click `setup_windows.bat` to create an isolated `.venv` and install dependencies.
4. Open the game, equip the fishing rod, and stand beside a pond. Keep the camera still and use a similar view to the supplied recording. The avatar should be near the middle of the window and the water/tackle clearly visible.
5. Run `preview_windows.bat`. Verify the avatar box, brown pole, magenta line endpoint and yellow visible-bobber marker. Blue tint is **candidate water segmentation**, not a guarantee of fishable terrain.
6. Once the preview is satisfactory, close it and run `start_windows.bat`. Focus the game and press **F8** to arm. Press **F8** again to pause or **F9** to stop globally.

Moving the mouse to a screen corner triggers PyAutoGUI's failsafe at the next input action. F9 is the primary emergency stop; it does not require the overlay to be focused. Q/Esc quits when the overlay is focused. T saves a screenshot. There are no force-cast/reel keys that bypass dry-run.

## CLI equivalents

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py --no-click
.venv\Scripts\python main.py --enable-clicks
```

Options:

```powershell
# Find the correct game window title
.venv\Scripts\python main.py --list-windows
# Saved calibration profile
.venv\Scripts\python main.py --config profile.json --enable-clicks
# Specify a cast location as fractions of the window, not screen pixels
.venv\Scripts\python main.py --enable-clicks --aim 0.65,0.70
# No overlay, still F8/F9 controlled
.venv\Scripts\python main.py --enable-clicks --no-view
```

Default aiming is toward candidate water beside the avatar. If no such target is found, the program **pauses without clicking**; calibrate the pond or specify `--aim`. Reeling clicks at the latched tackle position. Mouse input moves the cursor; do not try to use the same mouse for other tasks while armed.

## Recognition and fishing loop

- **Player:** cool-white/gray avatar mask with size, upright shape and central-position checks. This is tuned to the supplied white avatar, not a universal avatar classifier.
- **Water:** HSV segmentation with ground/UI exclusion; an optional manually selected pond constraint reduces rock/sand confusion.
- **Fish:** dark silhouettes with filled-body contrast, aspect, avatar-exclusion and water/cast-area checks. Scenery/lighting can still confuse them; a pond profile helps.
- **Rod:** brown line attached to the detected avatar, not the longest random scene line.
- **Fishing line:** color/geometry tracing from the pole tip. Stops at the supported endpoint instead of extrapolating dark water edges.
- **Bobber:** compact pale float near that endpoint. Shaded bobbers need not be pure white. If a fish hides it, the **line endpoint is explicitly an estimate**, not a visible-bobber detection.
- **Bite:** a short-lived, compact brightness increase around the fixed tackle ROI, with edge evidence and a nearby fish that has settled. Uses consecutive confirmation frames. Fish arrival/proximity alone does not reel.

The loop casts once after stable upright-rod readiness, ignores the casting splash while settling, watches the tackle and immediately requests one reel click when confirmed, then waits through cooldown. Starting with a visible cast already out does **not** send a cast click. A timeout with the line still visible pauses instead of blindly toggling/recasting.

All detection distances use a 1280-wide working frame. Capture coordinates are converted back to the actual game window for mouse input. The fresh-frame/bite loop targets 60 FPS, full-scene recognition 10 FPS, and overlay redraws 10 FPS. Achieved rates depend on hardware; these are not guaranteed reaction times or catch rates.

## Calibration

`profiles/recording-daytime.json` and `profiles/recording-night.json` include **view-specific pond rectangles**
for their corresponding recordings, excluding the left rock wall. Try one only with a
matching camera view; otherwise select your own pond rectangle:

```powershell
.venv\Scripts\python main.py --no-click --config profiles/recording-daytime.json
```


Static screenshot inspection works without mouse/desktop-control dependencies:

```powershell
.venv\Scripts\python calibrate.py --test "fisher/assets/samples/fish3.png"
.venv\Scripts\python calibrate.py --capture --view
```

Constrain the pond to visible fishable water (GUI selection, Enter accepts):

```powershell
.venv\Scripts\python calibrate.py --capture --select-pond --save-config profile.json
.venv\Scripts\python main.py --no-click --config profile.json
```

Tune screenshot HSV thresholds and save a profile with S:

```powershell
.venv\Scripts\python calibrate.py --test calibration-source.png --config profile.json --tune
```

The bubble slider changes **sensitivity only**. A static screenshot cannot validate a temporal bite event; check the saved setting using video replay. Raising `bite_relative_jump` or `bite_confirm_frames` reduces sensitivity and can miss brief bites; lowering them can cause false reels.

Optional bobber template: choose a tight crop with visible texture/contrast and no mouse pointer or UI:

```powershell
.venv\Scripts\python calibrate.py --test calibration-source.png --crop-template bobber
```

The template is stored at `fisher/assets/templates/bobber.png`, near-endpoint gated, and resolution-normalized. No usable template is bundled by default: many recording frames show an occluded float. Template matching alone never authorizes a click.

A pond rectangle is camera-specific. Select it again after a major camera change; the software does not navigate the player, switch equipment, locate new ponds, or correct the camera.

## Offline replay — no clicks on any platform

```powershell
python -m pip install -r requirements-replay.txt
python replay.py "20261014-0549-57.1097707.mp4" --overlay --output replay-output
python replay.py gameplay.mp4 --config profile.json --output tuned-replay
python -m unittest discover -s tests -v
```

The supplied videos are not duplicated inside this download; use your original recordings. Replay observes the recorded casts; it cannot simulate how the game would respond to hypothetical clicks. It exports `frames.jsonl`, `summary.json`, event screenshots, and optionally `overlay.mp4`. A predicted reel is an **attempt**, not a confirmed catch.

### Fast temporal recheck

After a full detector replay, bite/state settings can be rechecked against
its cached geometry without re-running the expensive vision pass:

```powershell
python scripts/recheck_temporal.py gameplay.mp4 replay-output/frames.jsonl --output temporal-review --overlay
```

Use the **same video** as the cache. The review overlay displays cached fish
centres (not bounding boxes), player/rod/line/float markers and the current
bite ROI. It intentionally omits water masks, which are not cached. It is an
offline observer, not a live-input or catch-success test.

## Safety and limitations

- No clicks unless explicitly enabled, armed, in bounds, and the verified game window is foreground. Does not automatically focus another app.
- Focus loss and window move/resize pause live operation and discard stale temporal history. F8 must be used to re-arm. Closing/minimizing the game stops capture; no silent desktop fallback.
- Full-monitor and manual-region capture are preview-only. Multiple matching windows require a specific title or focusing the intended game before launch.
- Camera movement within the game is not reliably distinguished from every bite animation. Large local motion resets the bite background, but subtle camera/lighting changes can still fool it. Pause before moving the camera.
- Supported reference scenes: the supplied white avatar, daytime and warm-lantern/night pond, equipped reddish-brown rod, stable third-person view. Extreme darkness, colored avatars, other environments, reflections, overlapping fish and UI changes still need calibration and may remain unsupported. Lighting correction uses the avatar as a neutral reference; it is not a universal day/night classifier.
- Live input has **not** been tested in an actual Windows game session here. Included results are offline replay and automated logic tests, not a certified working bot or proof of catch success. Stay in preview until your detections are correct.
- Logs count cast/reel attempts only; no inventory/catch-success verification is implemented.

See `validation/v3/VALIDATION.md` for current test results. Version-1 and version-2 reports are retained as historical validation, not fresh tests of the version-3 live loop.
