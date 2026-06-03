"""Parse an SVG file into a list of Stroke polylines.

vtracer (the upstream stage) emits SVG with `path` elements; each `path` may
contain multiple subpaths (separated by `M`/`m` commands). Each subpath is one
disconnected stroke from the robot's perspective — pen must lift between them.
"""

from __future__ import annotations

import numpy as np
from svgpathtools import parse_path, svg2paths

from common.stroke import Stroke


def discretize_segment(seg, samples_per_unit: float = 0.5,
                       min_samples: int = 2) -> list[tuple[float, float]]:
    """Sample a single svgpathtools segment into discrete points.

    Lines need 2 points; curves need enough to follow the curvature visually
    without exploding the gcode size.
    """
    length = seg.length(error=1e-3)
    n = max(min_samples, int(round(length * samples_per_unit)))
    ts = np.linspace(0.0, 1.0, n)
    return [(p.real, p.imag) for p in (seg.point(t) for t in ts)]


def path_to_strokes(path,
                    samples_per_unit: float = 0.5) -> list[Stroke]:
    """One svgpathtools Path → one or more Strokes (one per connected subpath).

    Adjacent segments whose endpoints coincide belong to the same stroke; a
    discontinuity (M command in the original SVG) starts a new stroke.
    """
    if len(path) == 0:
        return []

    strokes: list[Stroke] = []
    current_points: list[tuple[float, float]] = []
    last_end = None

    for seg in path:
        seg_start = (seg.start.real, seg.start.imag)
        if last_end is not None and not _points_close(last_end, seg_start):
            # discontinuity → flush current stroke
            if len(current_points) >= 2:
                strokes.append(Stroke(np.array(current_points)))
            current_points = [seg_start]
        elif not current_points:
            current_points = [seg_start]

        pts = discretize_segment(seg, samples_per_unit=samples_per_unit)
        # Avoid duplicating the seg.start point we already have
        current_points.extend(pts[1:])
        last_end = (seg.end.real, seg.end.imag)

    if len(current_points) >= 2:
        strokes.append(Stroke(np.array(current_points)))
    return strokes


def _points_close(a: tuple[float, float], b: tuple[float, float],
                  tol: float = 1e-3) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def load_strokes(svg_path: str,
                 samples_per_unit: float = 0.5) -> list[Stroke]:
    """Load all strokes from every path element in an SVG file."""
    paths, _attrs = svg2paths(svg_path)
    strokes: list[Stroke] = []
    for path in paths:
        strokes.extend(path_to_strokes(path, samples_per_unit=samples_per_unit))
    return strokes


def load_strokes_from_d_string(d: str, samples_per_unit: float = 0.5) -> list[Stroke]:
    """Helper for tests: parse a single SVG `d` string into strokes."""
    return path_to_strokes(parse_path(d), samples_per_unit=samples_per_unit)
