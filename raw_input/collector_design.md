## Mouse and view logging design (high-level)

- Target: PC FPS or aim trainer where raw mouse input and view angles can be logged.
- Key signals:
  - Mouse deltas: counts (dx, dy) with timestamps.
  - View changes: yaw/pitch per frame.
  - Context: weapon, FOV, sensitivity, accel settings, map, etc.
- Constraints:
  - Must respect game ToS and anti-cheat systems.
  - Prefer official APIs or external aim trainers that expose telemetry.

This repo includes a minimal `raw_input/logger.py` prototype that just records mouse deltas and timestamps
as a starting point; it does not integrate with a game.