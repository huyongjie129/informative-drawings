"""Trapezoidal velocity-profile time model for UltraArm P340.

Used by both the calibration script (to fit measured times → params) and the
TSP solver (to compute edge costs from stroke endpoints).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


@dataclass
class MotionParams:
    v_draw: float       # mm/s, pen-down drawing speed
    v_travel: float     # mm/s, pen-up rapid speed
    accel: float        # mm/s^2, acceleration for travel moves
    t_pen_toggle: float = 0.3  # seconds, per pen up+down round trip

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MotionParams":
        return cls(**json.loads(Path(path).read_text()))

    @classmethod
    def default(cls) -> "MotionParams":
        # Educated guesses; replace with calibrated values before claiming numbers.
        return cls(v_draw=30.0, v_travel=80.0, accel=200.0, t_pen_toggle=0.3)


def trapezoidal_time(distance: np.ndarray | float,
                     v_max: float,
                     accel: float) -> np.ndarray | float:
    """Time to move `distance` mm under a symmetric trapezoidal velocity profile.

    Long enough to reach cruise: t = d/v_max + v_max/a
    Too short (triangular):      t = 2*sqrt(d/a)

    Both branches are smooth in their domain. The switch happens at
    d_critical = v_max**2 / a (distance at which the triangle peaks at v_max).
    """
    d = np.asarray(distance, dtype=np.float64)
    d_critical = v_max * v_max / accel
    long_branch = d / v_max + v_max / accel
    short_branch = 2.0 * np.sqrt(np.maximum(d, 0.0) / accel)
    return np.where(d >= d_critical, long_branch, short_branch)


def draw_time(distance: np.ndarray | float, params: MotionParams) -> np.ndarray | float:
    """Time to draw a segment of given length at constant draw speed."""
    return np.asarray(distance, dtype=np.float64) / params.v_draw


def travel_time(distance: np.ndarray | float, params: MotionParams) -> np.ndarray | float:
    """Pen-up travel time, includes pen toggle overhead at each transition."""
    return trapezoidal_time(distance, params.v_travel, params.accel) + params.t_pen_toggle


def fit_trapezoidal(distances: np.ndarray,
                    observed_times: np.ndarray,
                    v_init: float = 80.0,
                    a_init: float = 200.0) -> tuple[float, float]:
    """Fit (v_max, accel) from observed move times via least squares.

    Returns (v_max_mm_s, accel_mm_s2).
    """
    distances = np.asarray(distances, dtype=np.float64)
    observed_times = np.asarray(observed_times, dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        v, a = params
        return trapezoidal_time(distances, v, a) - observed_times

    result = least_squares(
        residuals,
        x0=np.array([v_init, a_init]),
        bounds=([1.0, 1.0], [1000.0, 10000.0]),
    )
    v_fit, a_fit = result.x
    return float(v_fit), float(a_fit)


def fit_draw_speed(distances: np.ndarray, observed_times: np.ndarray) -> float:
    """Fit a single draw speed (no accel) for pen-down moves.

    Drawing moves are typically slow and dominated by the steady-state speed.
    """
    distances = np.asarray(distances, dtype=np.float64)
    observed_times = np.asarray(observed_times, dtype=np.float64)
    # closed-form least squares for t = d / v  ->  v = sum(d^2) / sum(d*t)
    numerator = float(np.sum(distances * distances))
    denominator = float(np.sum(distances * observed_times))
    if denominator <= 0:
        raise ValueError("invalid observed times for draw-speed fit")
    return numerator / denominator
