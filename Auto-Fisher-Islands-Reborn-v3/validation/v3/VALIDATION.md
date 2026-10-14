# Version-3 validation — fish filtering and local loop timing

## Automated tests

65 tests passed (`python -m unittest discover -s tests -v`). Compile checks and
`main.py --help` also passed. Full test output is in `unit-tests.txt`.

Added coverage: filled versus outline-only fish, thin/distant/avatar-edge
rejection, current-line evidence, missing-line/fish-only rejection, latest-frame
queue bounds and frame ownership, nonblocking result polls, generation reset,
worker failure reporting, stale/out-of-order scene results, distinct stationary
fish observations, cached tackle timestamps, pending stop requests, and equal
background adaptation time constants at 30/60 FPS.

## Recorded-scene spot checks

Seven daylight/night fixtures were checked. `fish-fixtures.json` records the
v2/v3 candidate centres and line/rod checks. Examples:

- `gameplay_09`: four candidates reduced to the two visible fish; avatar-side
  and scenery candidates were rejected.
- `night_040`: rock-wall candidate removed; tackle-area fish retained.
- All seven fixtures retained rod evidence. The uncast `gameplay_00` correctly
  lacked current string support; six cast fixtures retained it.

These examples are not a labelled precision/recall benchmark and do not prove
that every fish/scenery case is handled. Pond calibration remains useful.
Preview images show current geometry and candidate-water segmentation.

## Short raw-video bite regressions

Both checks used the final v3 scene detector and raw video pixels at 30 FPS,
with no mouse input and default configuration:

| Recording section | Frames | Predicted reel |
| --- | --- | --- |
| Night, first 5 seconds | 151 | 3.9000s |
| Daylight, first 12 seconds | 361 | 11.5667s |

A separate deterministic simulation published scene geometry at 10Hz after an
artificial 80ms delay while checking tackle pixels at 60Hz using duplicated
30FPS recording frames. It retained the expected single predicted reel in each
section (night 3.8833s, daylight 11.5500s). This tests cached-result behaviour,
not live throughput. Those sub-frame time differences arise from the simulated
sampling schedule and are **not evidence of measured faster catch performance**.
See `short-replays.json`.

To reproduce with the original recordings:

```powershell
python scripts/check_v3_replays.py --day "20261014-0549-57.1097707.mp4" --night "20261014-0752-08.7823154.mp4"
```

The full 45/139-second recordings were not freshly replayed end-to-end with v3.
The v2 full-recording reports remain historical. No new video was provided for
this update; the supplied earlier recordings/fixtures were reused.

## Live latency and limits

The new live loop separates scene scans from fresh tackle/bubble checks, keeps
only the latest pending frame, and throttles display redraws. Targets are
60Hz capture/checks, 10Hz scene scans, and 10Hz overlay; they are not guaranteed
rates. The existing two-frame bite confirmation and settled-fish gate remain.

Live Windows capture, keyboard hooks, mouse dispatch, achieved loop rates and
real catch success were not tested here. The user reported the preceding v2
working; that is not a live test of v3. Preview v3 before arming input.

JSONL action/periodic timing includes capture duration, scene scan duration,
scene result age, decision duration, and achieved loop FPS. Dispatch timing is
local capture-to-dispatch and click-call time, not bite-to-server-response time.
No game ping/server delay is measured or corrected. Missed weak flashes,
lighting changes, unexpected avatars and camera movement remain possible.
