"""
Desktop mouse logger — captures raw mouse deltas at high resolution.

Works alongside games (runs in background, captures system-wide mouse events).
Records dx, dy, buttons, and high-resolution timestamps for later analysis.

Usage:
    # Record 60 seconds of mouse data while you play Valorant:
    python -m raw_input.logger --duration 60 --output data/recorded_sessions/

    # Record until you press Ctrl+C:
    python -m raw_input.logger --duration 0

    # Quick 10-second test:
    python -m raw_input.logger
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

try:
    import mouse
except ImportError:
    mouse = None

try:
    import win32api
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


@dataclass
class MouseSample:
    """A single mouse event sample."""
    t: float            # timestamp (seconds since session start)
    dx: int             # raw x delta (counts)
    dy: int             # raw y delta (counts)
    buttons: int = 0    # bitmask: 1=left, 2=right, 4=middle
    event_type: str = "move"  # move, click, release


@dataclass
class RecordingSession:
    """Complete recording session with metadata."""
    samples: List[MouseSample] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    start_time: float = 0.0

    def summary(self) -> dict:
        if not self.samples:
            return {"num_samples": 0}

        durations = []
        speeds = []
        for i in range(1, len(self.samples)):
            dt = self.samples[i].t - self.samples[i - 1].t
            if dt > 0:
                durations.append(dt)
                dx = self.samples[i].dx
                dy = self.samples[i].dy
                speed = (dx**2 + dy**2) ** 0.5 / dt
                speeds.append(speed)

        clicks = sum(1 for s in self.samples if s.event_type == "click")

        return {
            "num_samples": len(self.samples),
            "duration_s": self.samples[-1].t if self.samples else 0,
            "avg_poll_rate_hz": 1.0 / (sum(durations) / len(durations)) if durations else 0,
            "avg_speed": sum(speeds) / len(speeds) if speeds else 0,
            "max_speed": max(speeds) if speeds else 0,
            "num_clicks": clicks,
        }


class DesktopMouseLogger:
    """
    High-resolution mouse logger that captures system-wide mouse events.

    Uses the `mouse` library for cross-platform event capture.
    Timestamps use time.perf_counter() for microsecond precision.
    """

    def __init__(self, dpi: int = 800, sensitivity: float = 0.5):
        if mouse is None:
            raise RuntimeError(
                "The `mouse` library is required. Install with:\n"
                "  pip install mouse"
            )

        self.session = RecordingSession()
        self.session.metadata = {
            "dpi": dpi,
            "sensitivity": sensitivity,
            "platform": sys.platform,
        }
        self._last_x = None
        self._last_y = None
        self._start_perf = 0.0
        self._running = False

    def _on_move(self, event):
        """Handle mouse move events from the `mouse` library."""
        if not self._running:
            return

        if hasattr(event, 'x') and hasattr(event, 'y'):
            x, y = event.x, event.y
        else:
            return

        t = time.perf_counter() - self._start_perf

        if self._last_x is None:
            self._last_x, self._last_y = x, y
            return

        dx = x - self._last_x
        dy = y - self._last_y
        self._last_x, self._last_y = x, y

        if dx == 0 and dy == 0:
            return

        self.session.samples.append(MouseSample(
            t=round(t, 6),
            dx=dx,
            dy=dy,
            event_type="move",
        ))

    def _on_click(self, event):
        """Handle mouse click events."""
        if not self._running:
            return

        t = time.perf_counter() - self._start_perf
        button_map = {"left": 1, "right": 2, "middle": 4}
        btn = button_map.get(getattr(event, 'button', 'left'), 1)
        event_type = "click" if getattr(event, 'event_type', '') == 'down' else "release"

        self.session.samples.append(MouseSample(
            t=round(t, 6),
            dx=0, dy=0,
            buttons=btn,
            event_type=event_type,
        ))

    def record(self, duration_sec: float = 10.0):
        """
        Record mouse events for `duration_sec` seconds.
        If duration_sec <= 0, record until Ctrl+C.
        """
        self.session.samples.clear()
        self._last_x = None
        self._last_y = None
        self._running = True
        self._start_perf = time.perf_counter()
        self.session.start_time = time.time()

        mouse.hook(self._dispatch)

        print(f"🔴 Recording mouse data...")
        if duration_sec > 0:
            print(f"   Duration: {duration_sec}s | Press Ctrl+C to stop early")
        else:
            print(f"   Press Ctrl+C to stop")

        try:
            if duration_sec > 0:
                start = time.perf_counter()
                while time.perf_counter() - start < duration_sec:
                    time.sleep(0.001)  # 1ms sleep for responsiveness
                    # Progress indicator every 5 seconds
                    elapsed = time.perf_counter() - start
                    if int(elapsed) % 5 == 0 and int(elapsed) > 0:
                        remaining = duration_sec - elapsed
                        if remaining > 0 and abs(elapsed - int(elapsed)) < 0.002:
                            print(f"   ⏱  {int(remaining)}s remaining | {len(self.session.samples)} samples")
            else:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n   ⏹  Stopped by user")
        finally:
            self._running = False
            mouse.unhook_all()

        summary = self.session.summary()
        print(f"\n✅ Recording complete!")
        print(f"   Samples: {summary['num_samples']}")
        print(f"   Duration: {summary['duration_s']:.1f}s")
        print(f"   Avg poll rate: {summary['avg_poll_rate_hz']:.0f} Hz")
        print(f"   Avg speed: {summary['avg_speed']:.0f} counts/s")
        print(f"   Clicks: {summary['num_clicks']}")

    def _dispatch(self, event):
        """Route events to the correct handler."""
        event_type = type(event).__name__
        if event_type == 'MoveEvent':
            self._on_move(event)
        elif event_type in ('ButtonEvent',):
            self._on_click(event)

    def save(self, output_dir: str = "data/recorded_sessions") -> str:
        """Save recording to a JSON file. Returns the filepath."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filename = f"desktop_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        data = {
            "source": "desktop_logger",
            "metadata": self.session.metadata,
            "summary": self.session.summary(),
            "samples": [asdict(s) for s in self.session.samples],
            "recorded_at": timestamp,
        }

        with open(filepath, "w") as f:
            json.dump(data, f)

        print(f"💾 Saved to {filepath} ({len(self.session.samples)} samples)")
        return filepath

    def to_numpy(self):
        """Convert to numpy arrays for analysis."""
        import numpy as np
        samples = self.session.samples
        return {
            "t": np.array([s.t for s in samples], dtype=np.float64),
            "dx": np.array([s.dx for s in samples], dtype=np.int32),
            "dy": np.array([s.dy for s in samples], dtype=np.int32),
            "buttons": np.array([s.buttons for s in samples], dtype=np.int32),
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🖱️  RawAccel Studio — Desktop Mouse Logger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Record 60s while playing Valorant:\n"
            "    python -m raw_input.logger --duration 60\n\n"
            "  Record until Ctrl+C:\n"
            "    python -m raw_input.logger --duration 0\n\n"
            "  Specify your DPI and sensitivity:\n"
            "    python -m raw_input.logger --dpi 1600 --sens 0.3 --duration 30"
        ),
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Recording duration in seconds (0 = until Ctrl+C)")
    parser.add_argument("--output", type=str, default="data/recorded_sessions",
                        help="Output directory for saved sessions")
    parser.add_argument("--dpi", type=int, default=800,
                        help="Your mouse DPI (stored as metadata)")
    parser.add_argument("--sens", type=float, default=0.5,
                        help="Your in-game sensitivity (stored as metadata)")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save the recording (just print summary)")
    args = parser.parse_args()

    logger = DesktopMouseLogger(dpi=args.dpi, sensitivity=args.sens)

    print("=" * 50)
    print("  🖱️  RawAccel Studio — Mouse Logger")
    print("=" * 50)
    print(f"  DPI: {args.dpi}  |  Sensitivity: {args.sens}")
    print()

    logger.record(duration_sec=args.duration)

    if not args.no_save and logger.session.samples:
        logger.save(output_dir=args.output)


if __name__ == "__main__":
    main()