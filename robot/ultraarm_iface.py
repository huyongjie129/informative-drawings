"""Interface to the Elephant Robotics UltraArm P340.

Two implementations:
  - MockUltraArm: simulates motion using the trapezoidal motion model. Used for
    local dry-runs, unit tests, and CI without hardware.
  - RealUltraArm: wraps pymycobot.ultraArm; only imported when constructed,
    so the rest of the codebase can be used without `pymycobot` installed.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from common.motion_model import (
    MotionParams,
    trapezoidal_time,
)


@dataclass
class MoveResult:
    """Outcome of a single commanded move."""
    start: np.ndarray
    end: np.ndarray
    distance: float
    elapsed: float
    pen_down: bool


class UltraArmInterface(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def home(self) -> None: ...
    @abstractmethod
    def pen_up(self) -> None: ...
    @abstractmethod
    def pen_down(self) -> None: ...
    @abstractmethod
    def move_to(self, x: float, y: float) -> MoveResult:
        """Pen-up rapid travel to (x, y). Returns measured timing."""
    @abstractmethod
    def draw_to(self, x: float, y: float) -> MoveResult:
        """Pen-down drawing move to (x, y). Returns measured timing."""
    @abstractmethod
    def get_position(self) -> np.ndarray: ...
    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.close()


class MockUltraArm(UltraArmInterface):
    """Simulated arm. Returns timings derived from a known MotionParams.

    Use `noise_std` > 0 to add gaussian noise to simulated timings — this lets
    `measure_motion.py` exercise its fitting routine on imperfect data.
    """

    def __init__(self,
                 params: MotionParams | None = None,
                 noise_std: float = 0.0,
                 seed: int | None = 0):
        self._params = params or MotionParams.default()
        self._pos = np.zeros(2)
        self._pen_down = False
        self._rng = np.random.default_rng(seed)
        self._noise_std = noise_std

    @property
    def params(self) -> MotionParams:
        return self._params

    def connect(self) -> None:
        pass

    def home(self) -> None:
        self._pos = np.zeros(2)
        self._pen_down = False

    def pen_up(self) -> None:
        self._pen_down = False

    def pen_down(self) -> None:
        self._pen_down = True

    def _noisy(self, t: float) -> float:
        if self._noise_std <= 0:
            return t
        return max(0.0, t + float(self._rng.normal(0.0, self._noise_std * t)))

    def move_to(self, x: float, y: float) -> MoveResult:
        target = np.array([float(x), float(y)])
        d = float(np.linalg.norm(target - self._pos))
        t_clean = float(trapezoidal_time(d, self._params.v_travel, self._params.accel))
        elapsed = self._noisy(t_clean)
        result = MoveResult(self._pos.copy(), target.copy(), d, elapsed, pen_down=False)
        self._pos = target
        return result

    def draw_to(self, x: float, y: float) -> MoveResult:
        target = np.array([float(x), float(y)])
        d = float(np.linalg.norm(target - self._pos))
        t_clean = d / self._params.v_draw
        elapsed = self._noisy(t_clean)
        result = MoveResult(self._pos.copy(), target.copy(), d, elapsed, pen_down=True)
        self._pos = target
        return result

    def get_position(self) -> np.ndarray:
        return self._pos.copy()

    def close(self) -> None:
        pass


class RealUltraArm(UltraArmInterface):
    """Wraps pymycobot.ultraArm. Lazy import so the dep is optional.

    The Python API exposes coordinate moves and direct gcode; we use coordinate
    moves and time them on the host side. For very small moves the round-trip
    serial latency dominates — that's exactly what we want to characterize.

    See: https://docs.elephantrobotics.com/docs/ultraArm/3-HowToUseultraArm/2-SoftwareControl/4-Python/2-PythonAPI.html
    """

    def __init__(self,
                 port: str,
                 baudrate: int = 115200,
                 z_paper: float = 0.0,
                 z_clear: float = 10.0,
                 draw_speed_mm_min: int = 1800,
                 travel_speed_mm_min: int = 4800):
        self._port = port
        self._baudrate = baudrate
        self._z_paper = z_paper
        self._z_clear = z_clear
        self._draw_speed = draw_speed_mm_min   # mm/min for gcode-style speed arg
        self._travel_speed = travel_speed_mm_min
        self._arm = None
        self._pen_down = False
        self._pos = np.zeros(2)

    def connect(self) -> None:
        try:
            from pymycobot.ultraArm import ultraArm  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "pymycobot is required for RealUltraArm. "
                "Install with: pip install pymycobot"
            ) from e
        self._arm = ultraArm(self._port, self._baudrate)
        self._arm.go_zero()
        self._pos = np.zeros(2)
        self._pen_down = False

    def home(self) -> None:
        assert self._arm is not None
        self._arm.go_zero()
        self._pos = np.zeros(2)

    def pen_up(self) -> None:
        assert self._arm is not None
        # Lift Z to the clearance height. Keeps XY constant.
        x, y = self._pos
        self._arm.set_coords([float(x), float(y), self._z_clear], self._travel_speed)
        self._pen_down = False

    def pen_down(self) -> None:
        assert self._arm is not None
        x, y = self._pos
        self._arm.set_coords([float(x), float(y), self._z_paper], self._travel_speed)
        self._pen_down = True

    def _timed_move(self, x: float, y: float, speed: int, pen_down: bool) -> MoveResult:
        assert self._arm is not None
        target = np.array([float(x), float(y)])
        d = float(np.linalg.norm(target - self._pos))
        z = self._z_paper if pen_down else self._z_clear
        start_time = time.perf_counter()
        self._arm.set_coords([float(x), float(y), z], speed)
        # set_coords in pymycobot is blocking for ultraArm; if not on your build,
        # add a wait_done() / polling loop here.
        elapsed = time.perf_counter() - start_time
        result = MoveResult(self._pos.copy(), target.copy(), d, elapsed, pen_down=pen_down)
        self._pos = target
        return result

    def move_to(self, x: float, y: float) -> MoveResult:
        return self._timed_move(x, y, self._travel_speed, pen_down=False)

    def draw_to(self, x: float, y: float) -> MoveResult:
        return self._timed_move(x, y, self._draw_speed, pen_down=True)

    def get_position(self) -> np.ndarray:
        return self._pos.copy()

    def close(self) -> None:
        if self._arm is not None:
            try:
                self._arm.close()
            except Exception:
                pass
            self._arm = None
