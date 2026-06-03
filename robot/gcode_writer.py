"""Emit gcode for an ordered list of strokes.

Uses the simple servo-pen convention: M3 = pen down, M5 = pen up.
G0 = rapid travel (pen up), G1 = controlled feedrate (pen down drawing).
"""

from __future__ import annotations

from io import StringIO

import numpy as np

from common.stroke import Stroke


def emit_gcode(strokes: list[Stroke],
               directions: list[bool],
               draw_feed_mm_min: int = 1800,
               travel_feed_mm_min: int = 4800,
               start_pos: np.ndarray | None = None) -> str:
    """Serialize strokes to a gcode string.

    Feed rates are in mm/min (standard gcode units), independent of the motion
    model used for time estimation.
    """
    if len(strokes) != len(directions):
        raise ValueError("strokes and directions must have the same length")

    buf = StringIO()
    buf.write("; draw_robot Phase 1 baseline output\n")
    buf.write(f"; {len(strokes)} strokes\n")
    buf.write("G21 ; mm\n")
    buf.write("G90 ; absolute\n")
    buf.write(f"F{travel_feed_mm_min}\n")
    buf.write("M5 ; pen up\n")

    pos = np.zeros(2) if start_pos is None else np.asarray(start_pos, dtype=np.float64)

    for stroke_idx, (stroke, reverse) in enumerate(zip(strokes, directions)):
        s = stroke.reversed() if reverse else stroke
        # Travel to start of stroke (pen up).
        if not np.allclose(s.start, pos):
            buf.write(f"G0 X{s.start[0]:.4f} Y{s.start[1]:.4f}\n")
        buf.write("M3 ; pen down\n")
        buf.write(f"F{draw_feed_mm_min}\n")
        # Draw each subsequent point.
        for pt in s.points[1:]:
            buf.write(f"G1 X{pt[0]:.4f} Y{pt[1]:.4f}\n")
        buf.write("M5 ; pen up\n")
        buf.write(f"F{travel_feed_mm_min}\n")
        pos = s.end

    buf.write("G0 X0 Y0 ; return home\n")
    return buf.getvalue()
