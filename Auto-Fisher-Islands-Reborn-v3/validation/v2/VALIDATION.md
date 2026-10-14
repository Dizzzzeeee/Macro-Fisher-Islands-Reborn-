# Version 2 validation — rod and changing illumination

## Scope

The original 45.33-second daylight recording and the new 138.70-second
sunset/warm-lantern/night recording were used for tuning and offline testing.
This is **not** an independent accuracy benchmark or proof of catch success.
Live Windows capture/input has not been exercised here. No real clicks were sent.

## Automated tests

**40 tests passed**, covering the prior state/input safeguards plus full warm
avatar/pole recovery, night float/endpoint recovery, gain bounds, unchanged
pixels without a neutral reference or with correction disabled, simulated
exposure/color shifts, dim bubble flashes, exposure-jump vetoes, verified
float anchoring during occlusion, leaning-pole cast suppression and late-flash
suppression after tackle disappears. See `unit-tests.txt`.

## Detector availability on the new recording

One frame per second was sampled across the full clip: **139 frames**.
The comparison uses the actual preceding delivered detector, preserved
before editing, and the final v2 vision pipeline.

| Detector | v1 frames with a result | v2 frames with a result |
| --- | ---: | ---: |
| player | 98 | 139 |
| rod | 3 | 129 |
| line | 0 | 123 |
| bobber | 0 | 86 |
| fish | 78 | 119 |

These are availability counts, **not correctness/accuracy scores**.
A result may still be wrong, and some misses occur during changes in pose,
casting/reeling, occlusion or camera movement. No ground-truth annotations
were created for every frame. The raw results are in `rod-comparison.json`.

## Final temporal checks

The vision pass ran across both entire recordings. Final bubble/state logic
was rechecked on the same raw video frames using the saved detector geometry:

- Daylight: 1360 frames at 30 FPS. Predicted reel times:
  11.5667s.
- Night: 1387 sampled frames at 10 FPS over the full 138.7-second
  recording. Predicted reel times:
  4.00s, 13.30s, 26.80s, 56.40s, 75.70s.
- A separate 30-FPS check of the first five seconds of the new recording
  predicted a reel near 3.90 seconds while the line was still visible.

Predictions are **attempts, not confirmed catches**, and are not an exhaustive
list of the user's successful manual catches. Weak or occluded bites can still
be missed. The 10-FPS night review cannot establish a 30-FPS live reaction-time
or throughput guarantee.

The earlier daylight prediction near 24.93 seconds is now suppressed: the
cast is no longer recently detected at that instant. The guard prioritizes
avoiding a late click over claiming a catch from an unverified brightness
flash. This can also suppress genuine cues when tackle recognition drops out.

`night-current/` and `day-current/` contain the **final** temporal results and
matching event screenshots. `night-final/` and `day-final/` retain full detector
geometry caches; their earlier temporal actions predate the last bite-safety
refinement and should not be treated as final decisions.

## Annotated review

The separately provided `Auto-Fisher-v2-night-replay.mp4` corresponds to the
final 10-FPS temporal recheck. It uses real cached player/pole/line/float
coordinates and **fish-centre markers**, not invented fish bounding boxes.
Water masks are not cached and are intentionally omitted from this review.

The raw game image is shown. Lighting correction is applied only internally
for object recognition; the caption reports measured brightness and the
bounded BGR gains. A green ROI marks the latched bite area.

`night-preview.jpg` shows the full detector with the night camera pond preset.
That preset is view-specific; reselect the pond when the camera changes.

## Environment / remaining limitations

- Python 3.13.14, OpenCV 5.0.0, NumPy 2.5.3, Linux.
- Actual live FPS/latency depends on the user's Windows hardware and capture.
- The white/gray supplied avatar supplies the neutral reference. Colored
  avatars, extreme darkness, unrelated ponds or different game visuals are
  not universally supported.
- HSV candidate water can still include scenery; the pond constraint helps.
- Detection can fail during fast motion, overlapping fish and cast animations.
- Fish availability and predicted reels do not verify inventory/catch success.
- Preview first. Enable clicking only after calibration and where permitted.
