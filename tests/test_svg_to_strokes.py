"""Tests for SVG → strokes parsing."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.svg_to_strokes import load_strokes_from_d_string


class TestSubpathSplitting:
    def test_single_line(self):
        strokes = load_strokes_from_d_string("M 0 0 L 10 0")
        assert len(strokes) == 1
        assert np.allclose(strokes[0].start, [0, 0])
        assert np.allclose(strokes[0].end, [10, 0])

    def test_two_disconnected_subpaths(self):
        # M starts a new subpath ⇒ two strokes
        strokes = load_strokes_from_d_string("M 0 0 L 10 0 M 20 0 L 30 0")
        assert len(strokes) == 2
        assert np.allclose(strokes[0].end, [10, 0])
        assert np.allclose(strokes[1].start, [20, 0])

    def test_continuous_polyline_is_one_stroke(self):
        strokes = load_strokes_from_d_string("M 0 0 L 10 0 L 10 10 L 0 10 Z")
        assert len(strokes) == 1
        # Z closes back to start (0, 0)
        assert np.allclose(strokes[0].end, [0, 0])
        assert strokes[0].arc_length() == pytest.approx(40.0, rel=0.01)


class TestCurveSampling:
    def test_curve_has_many_samples(self):
        # quadratic Bezier curve
        strokes = load_strokes_from_d_string("M 0 0 Q 50 100 100 0")
        assert len(strokes) == 1
        assert len(strokes[0].points) > 5  # should be sampled densely
        # arc length should be roughly the parabola length, not the chord
        assert strokes[0].arc_length() > 100.0
