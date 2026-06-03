"""Stroke data structure: a single connected pen-down path as a 2D polyline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.motion_model import MotionParams, draw_time, travel_time


@dataclass
class Stroke:
    """A single connected drawing path, represented as a polyline.

    `points` is shape (N, 2) with N >= 2 — the robot draws straight segments
    between consecutive points at v_draw with the pen down.
    """

    points: np.ndarray

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Stroke.points must be (N,2), got {self.points.shape}")
        if len(self.points) < 2:
            raise ValueError("Stroke needs at least 2 points")

    @property
    def start(self) -> np.ndarray:
        return self.points[0]

    @property
    def end(self) -> np.ndarray:
        return self.points[-1]

    def arc_length(self) -> float:
        """Total length of the polyline in mm (same units as `points`)."""
        diffs = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def reversed(self) -> "Stroke":
        return Stroke(self.points[::-1].copy())

    def draw_time(self, params: MotionParams) -> float:
        return float(draw_time(self.arc_length(), params))


def total_time(strokes: list[Stroke],
               directions: list[bool],
               params: MotionParams,
               start_pos: np.ndarray | None = None) -> float:
    """Total drawing time for an ordered, directed list of strokes.

    `directions[i]` = False means draw strokes[i] forward (start->end);
    True means reverse it. The robot is at `start_pos` (defaults to origin)
    before the first stroke.
    """
    if len(strokes) != len(directions):
        raise ValueError("strokes and directions must have the same length")

    pos = np.zeros(2) if start_pos is None else np.asarray(start_pos, dtype=np.float64)
    total = 0.0

    for stroke, reverse in zip(strokes, directions):
        s = stroke.reversed() if reverse else stroke
        # Pen-up move from current position to stroke start
        d = float(np.linalg.norm(s.start - pos))
        total += float(travel_time(d, params))
        # Pen-down drawing along the stroke
        total += s.draw_time(params)
        pos = s.end

    return total


def default_order_time(strokes: list[Stroke],
                       params: MotionParams,
                       start_pos: np.ndarray | None = None) -> float:
    """Drawing time in the strokes' original order, all forward direction."""
    return total_time(strokes, [False] * len(strokes), params, start_pos)
