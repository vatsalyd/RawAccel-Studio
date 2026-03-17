"""
Desktop mouse logger — captures raw mouse deltas at high resolution.

Run alongside games to record mouse movement data for ML analysis.

Usage:
    python -m collector.logger --duration 60 --dpi 800 --sens 0.5
    python -m collector.logger --duration 0   # until Ctrl+C
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List

try:
    import mouse
except ImportError:
    mouse = None


@dataclass
class MouseSample:
    t: float          # seconds since session start
    dx: int           # x delta (counts)
    dy: int           # y delta (counts)
    buttons: int = 0  # bitmask: 1=left, 2=right, 4=middle
    event_type: str = "move"


class DesktopMouseLogger:
    """
    High-resolution mouse logger using the `mouse` library.
    Timestamps use time.perf_counter() for microsecond precision.
    """

    def __init__(self, dpi: int = 800, sensitivity: float = 0.5):
        if mouse is None:
            raise RuntimeError(
                "Install the mouse library: pip install mouse"
            )
        self.samples: List[MouseSample] = []
        self.metadata = {"dpi": dpi, "sensitivity": sensitivity}
        self._last_x = None
        self._last_y = None
        self._start = 0.0
        self._running = False

    def _dispatch(self, event):
        if not self._running:
            return
        name = type(event).__name__
        if name == "MoveEvent":
            self._on_move(event)
        elif name == "ButtonEvent":
            self._on_click(event)

    def _on_move(self, event):
        if not hasattr(event, "x"):
            return
        t = time.perf_counter() - self._start
        x, y = event.x, event.y
        if self._last_x is None:
            self._last_x, self._last_y = x, y
            return
        dx, dy = x - self._last_x, y - self._last_y
        self._last_x, self._last_y = x, y
        if dx == 0 and dy == 0:
            return
        self.samples.append(MouseSample(t=round(t, 6), dx=dx, dy=dy))

    def _on_click(self, event):
        t = time.perf_counter() - self._start
        btn_map = {"left": 1, "right": 2, "middle": 4}
        btn = btn_map.get(getattr(event, "button", "left"), 1)
        etype = "click" if getattr(event, "event_type", "") == "down" else "release"
        self.samples.append(MouseSample(t=round(t, 6), dx=0, dy=0, buttons=btn, event_type=etype))

    def record(self, duration: float = 10.0):
        """Record for `duration` seconds (0 = until Ctrl+C)."""
        self.samples.clear()
        self._last_x = None
        self._running = True
        self._start = time.perf_counter()

        mouse.hook(self._dispatch)
        print(f"🔴 Recording... {'Ctrl+C to stop' if duration <= 0 else f'{duration}s'}")

        try:
            if duration > 0:
                start = time.perf_counter()
                while time.perf_counter() - start < duration:
                    time.sleep(0.001)
            else:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("⏹  Stopped")
        finally:
            self._running = False
            mouse.unhook_all()

        moves = [s for s in self.samples if s.event_type == "move"]
        clicks = len(self.samples) - len(moves)
        dur = self.samples[-1].t if self.samples else 0
        print(f"✅ {len(moves)} moves, {clicks} clicks, {dur:.1f}s")

    def save(self, output_dir: str = "data/raw") -> str:
        os.makedirs(output_dir, exist_ok=True)
        ts = int(time.time())
        path = os.path.join(output_dir, f"session_{ts}.json")
        data = {
            "source": "desktop_logger",
            "metadata": self.metadata,
            "samples": [asdict(s) for s in self.samples],
            "recorded_at": ts,
        }
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"💾 Saved → {path}")
        return path


def main():
    parser = argparse.ArgumentParser(description="🖱️ RawAccel Studio — Mouse Logger")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--dpi", type=int, default=800)
    parser.add_argument("--sens", type=float, default=0.5)
    args = parser.parse_args()

    logger = DesktopMouseLogger(dpi=args.dpi, sensitivity=args.sens)
    logger.record(duration=args.duration)
    if logger.samples:
        logger.save(output_dir=args.output)


if __name__ == "__main__":
    main()
