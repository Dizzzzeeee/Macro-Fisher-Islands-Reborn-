# Optional bobber template

Use `calibrate.py --test screenshot.png --crop-template bobber` to create
`bobber.png` here. Choose a tight, textured crop (3..100 pixels in the
1280-wide working image), not a flat-color patch or a mouse pointer.
The float must still be near a traced fishing-line endpoint; a template
match alone never authorizes a click. Fish/bubble template classifiers
are not implemented. Main detection uses color, shape and temporal changes.
