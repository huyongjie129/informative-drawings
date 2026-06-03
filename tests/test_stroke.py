"""Tests for the Stroke data structure and time accounting."""

from __future__ import annotations

import numpy as np
import pytest

from common.motion_model import MotionParams
from common.stroke import Stroke, default_order_time, total_time


def make_params():
    return MotionParams(v_draw=30.0, v_travel=80.0, accel=200.0, t_pen_toggle=0.3)


class TestStroke:
    def test_simple_line_arc_length(self):
        s = Stroke([[0, 0], [3, 4]])
        assert s.arc_length() == pytest.approx(5.0)

    def test_polyline_arc_length(self):
        s = Stroke([[0, 0], [1, 0], [1, 1], [2, 1]])
        assert s.arc_length() == pytest.approx(3.0)

    def test_endpoints(self):
        s = Stroke([[0, 0], [5, 5], [10, 0]])
        assert np.allclose(s.start, [0, 0])
        assert np.allclose(s.end, [10, 0])

    def test_reversed(self):
        s = Stroke([[0, 0], [1, 2], [3, 4]])
        r = s.reversed()
        assert np.allclose(r.start, [3, 4])
        assert np.allclose(r.end, [0, 0])
        # arc length is invariant
        assert r.arc_length() == pytest.approx(s.arc_length())

    def test_rejects_single_point(self):
        with pytest.raises(ValueError):
            Stroke([[0, 0]])

    def test_rejects_3d(self):
        with pytest.raises(ValueError):
            Stroke([[0, 0, 0], [1, 1, 1]])


class TestTotalTime:
    def test_ordering_changes_total_time(self):
        params = make_params()
        # two strokes: one short stroke at origin, one short stroke far away
        s_near = Stroke([[0, 0], [1, 0]])
        s_far = Stroke([[100, 100], [101, 100]])

        # default order [near, far]: travel 0 + draw + travel ~141 + draw
        t_near_first = total_time([s_near, s_far], [False, False], params)
        # reversed order [far, near]: travel 141 + draw + travel ~141 + draw
        t_far_first = total_time([s_far, s_near], [False, False], params)

        # near-first should be cheaper because origin is at [0,0]
        assert t_near_first < t_far_first

    def test_direction_can_save_travel(self):
        params = make_params()
        # Origin at [0,0]. Stroke goes from [10,0] to [0,0].
        # Forward: travel to [10,0] then draw to [0,0]; pos ends at [0,0].
        # Reversed: travel to [0,0] (zero distance) then draw to [10,0]; pos ends at [10,0].
        s = Stroke([[10, 0], [0, 0]])
        t_forward = total_time([s], [False], params, start_pos=np.array([0.0, 0.0]))
        t_reversed = total_time([s], [True], params, start_pos=np.array([0.0, 0.0]))
        # Reversed avoids 10mm travel
        assert t_reversed < t_forward

    def test_default_order_helper(self):
        params = make_params()
        s1 = Stroke([[0, 0], [1, 1]])
        s2 = Stroke([[5, 5], [6, 6]])
        assert default_order_time([s1, s2], params) == total_time(
            [s1, s2], [False, False], params
        )

    def test_start_position_affects_first_travel(self):
        params = make_params()
        s = Stroke([[10, 0], [11, 0]])
        t_from_origin = total_time([s], [False], params, start_pos=np.array([0.0, 0.0]))
        t_from_nearby = total_time([s], [False], params, start_pos=np.array([10.0, 0.0]))
        assert t_from_nearby < t_from_origin
