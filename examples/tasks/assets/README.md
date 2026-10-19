# Demo assets

## `airport_aerial.jpg`

An overhead view of San Diego International Airport (1280×606). It is the
default image for all inference examples: detection, segmentation, pose,
depth, classification and the optional YOLO examples. `oriented_detection.py`
uses it because the GTR OBB checkpoint is trained on DOTA aerial imagery.

- **Source**: Wikimedia Commons,
  [Aerial photograph of San Diego airport 1.jpg](https://commons.wikimedia.org/wiki/File:Aerial_photograph_of_San_Diego_airport_1.jpg)
- **Author**: Mertbiol
- **License**: CC0 1.0 (public domain dedication)
- **sha256**: `774292b56e6db311e8fd6e327966433406d498e80739be71afb8bf58c04c187a`
- **Shipped in**: the sdist (`MANIFEST.in`); not installed in the wheel.
- **Usage**:

  ```bash
  python examples/tasks/detection.py
  python examples/tasks/oriented_detection.py
  ```

  `oriented_detection.py` loads `gtrobb_s_dota` (MIT) at threshold 0.3 and
  detects parked aircraft as `plane` rotated boxes.

The image is CC0 and may be redistributed under the CC0 terms (attribution
appreciated, not required). Pass `--image PATH` to any example to use
task-specific imagery instead.
