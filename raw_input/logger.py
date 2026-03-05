import time
from dataclasses import dataclass
from typing import List

try:
    import mouse  # `pip install mouse` (Windows only)
except ImportError:
    mouse = None


@dataclass
class MouseEvent:
    t: float
    dx: int
    dy: int


class SimpleMouseLogger:
    def __init__(self):
        self.events: List[MouseEvent] = []
        self.last_x = None
        self.last_y = None

    def _on_move(self, x, y):
        t = time.time()
        if self.last_x is None:
            self.last_x, self.last_y = x, y
            return
        dx = x - self.last_x
        dy = y - self.last_y
        self.last_x, self.last_y = x, y
        self.events.append(MouseEvent(t=t, dx=dx, dy=dy))

    def run(self, duration_sec: float = 10.0):
        if mouse is None:
            raise RuntimeError("The `mouse` library is not installed.")
        self.events.clear()
        self.last_x = None
        self.last_y = None
        mouse.on_move(self._on_move)
        start = time.time()
        try:
            while time.time() - start < duration_sec:
                time.sleep(0.01)
        finally:
            mouse.unhook_all()

    def to_numpy(self):
        import numpy as np

        t = np.array([e.t for e in self.events], dtype=float)
        dx = np.array([e.dx for e in self.events], dtype=int)
        dy = np.array([e.dy for e in self.events], dtype=int)
        return t, dx, dy