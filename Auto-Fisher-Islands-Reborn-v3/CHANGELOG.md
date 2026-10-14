# Version 3 — fish filtering and local reel latency

- Filled-body/local-contrast fish validation; avatar-edge, thin-shape and
  cast-distance exclusions. Optional pond calibration remains supported.
- Latest-frame background scene worker (target 10Hz); independent fresh-frame
  bubble/tackle checks (target 60Hz), overlay cap (10Hz).
- Bounded queue, generation reset, stale/out-of-order result guard and distinct
  observation timestamps. Cached fish cannot fabricate stationary-fish history.
- Small-ROI current-line support check; stale cache alone cannot extend tackle evidence.
- Time-normalized temporal bubble background; consecutive confirmation retained.
- Local stage timing, achieved loop FPS and click dispatch timing in JSONL logs.
- Pending F8/F9 requests checked again immediately before mouse dispatch.
- `--sync-vision` fallback; `--vision-fps` and `--overlay-fps` controls.

No network-lag compensation, server-response measurement, or live Windows
catch-success verification is claimed. See validation/v3/VALIDATION.md.

# Version 2 — warm-light and rod detection update

Uses both user-provided gameplay recordings. Source-code release, not an EXE.

## Detection

- Avatar-referenced exposure/white-balance correction with bounded lookup tables.
- Warm-neutral full-avatar seeding instead of a feet-only cool-white mask.
- Removed abrupt fixed night rod-saturation overrides.
- Hand-anchored rod ROI, HSV/color and local chroma-ridge channels.
- Pole/string-junction tip refinement and tightly gated gradient string segments.
- Short-gap, collinear string reconstruction across fish occlusion.
- Relative-darkness fish search near traced tackle, no whole-screen fallback.
- Verified bobber anchors retained while the visible line is shortened by fish.
- Bounded local contrast/vote adaptation for dim bubble flashes.
- Uniform exposure-jump veto for bubble detection.
- Late-flash suppression after the cast disappears; no post-catch reel requests.

## Safety and tooling

- Nearly upright-pole requirement for an automatic cast.
- Existing opt-in, foreground-window, F8/F9, no-desktop-fallback safeguards retained.
- Passive replay recognizes longer recorded line disappearances as manual resets.
- Lighting diagnostics in overlays and replay logs; optional correction-off flag.
- Night camera preset and additional warm-light/occlusion regression fixtures.
- Cache-based temporal recheck tool and clearly labelled centre-only replay overlay.
- ROI-only pole processing, BGR lookup tables and downsampled scene statistics.

Live Windows input and real catch success remain unverified here. Recordings
used to tune the detector are not independent accuracy benchmarks.
