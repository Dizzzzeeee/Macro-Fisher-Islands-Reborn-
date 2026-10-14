# Validation scope

## Automated checks

28 tests passed (see `unit-tests.txt`): deterministic state transitions,
no-player/no-rod/no-water cast suppression, existing-line handling,
timeout pausing, one reel per event, moving-fish gating, synthetic bubble
confirmation, missing/shifted anchor reset, large-motion veto, opt-in/focus/
bounds/cooldown input guards, profile validation, and sample-frame regressions.

The input-guard tests use a mocked mouse. **No real clicks were sent.**

## Supplied gameplay recording

- Original input: `20261014-0549-57.1097707.mp4`, 2548 x 1332, 30 FPS,
  approximately 45.33 seconds.
- Replay processed **1360 frames** at 1280 x 669 working resolution.
- Default thresholds, no pond constraint, observer-only mode.
- Processing including decoding, overlays and video writing:
  **112.19 seconds** in this shared Linux environment.
  This is slower than real time here; no 30-FPS live-performance guarantee
  is made. Actual capture/input latency depends on the user's Windows PC.
- A separate 30-call detector timing on one frame averaged approximately
  45 ms per call during concurrent replay work; not an idle-hardware benchmark.

Predicted events:

- 11.5667s: predicted `reel`.
- 24.9333s: predicted `reel`.

Event screenshots are included. The shaded, occluded float can disappear
from the visible-bobber detector; the line endpoint then serves as a latched
**estimated anchor**. Two consecutive local brightness-change votes were
used for each reel prediction. The same recording was used for calibration
and validation, so these results are not an independent accuracy benchmark.

### Raw detector availability, not accuracy

| Detector | Frames with a result |
| --- | --- |
| player | 1360 |
| rod | 1317 |
| fish | 1223 |
| bobber | 969 |
| line_end | 1110 |

These counts do not prove that each detection is correct. Ground-truth labels,
false-positive/false-negative rates, catch success, and other camera/lighting
conditions have not been independently evaluated. The late part of the clip
also contains tackle/camera/pose changes; do not infer perfect coverage from
the two predicted events.

## Visual review

`detection-preview.jpg` shows the default pixel segmentation and demonstrates
that rock/sand can match candidate-water colors. `pond-constrained-preview.jpg`
uses `profiles/recording-daytime.json` to limit the search to the recorded pond.
That preset is camera-specific and must be recalibrated for another view.

The separately provided annotated MP4 corresponds to this default-profile
replay. It is a **read-only review artifact**, not a live gameplay test.

## Environment and limits

- Python 3.13.14, OpenCV 5.0.0, NumPy 2.5.3.
- Linux used for image processing, video replay and tests.
- Windows live capture, keyboard hooks, focus/DPI behavior and real game
  input are implemented but **not exercised on a Windows game session here**.
- No compiled Windows EXE is provided. Use the included Python setup scripts.
- Verify the overlay and run only where automation is permitted.
