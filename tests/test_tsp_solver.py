"""Tests for the TSP solver (greedy backend, always available)."""

from __future__ import annotations

import numpy as np
import pytest

from baseline.tsp_solver import solve, solve_greedy
from common.motion_model import MotionParams
from common.stroke import Stroke, default_order_time, total_time


def make_params():
    return MotionParams(v_draw=30.0, v_travel=80.0, accel=200.0, t_pen_toggle=0.3)


class TestGreedySolver:
    def test_empty_input(self):
        params = make_params()
        order, dirs = solve_greedy([], params)
        assert order == []
        assert dirs == []

    def test_single_stroke(self):
        params = make_params()
        s = Stroke([[5, 5], [10, 10]])
        order, dirs = solve_greedy([s], params)
        assert order == [0]
        assert len(dirs) == 1

    def test_picks_nearest_first(self):
        params = make_params()
        s_far = Stroke([[100, 100], [110, 100]])
        s_near = Stroke([[5, 0], [10, 0]])
        # Default order would draw far first; greedy should reorder.
        order, dirs = solve_greedy([s_far, s_near], params)
        assert order[0] == 1  # near stroke first

    def test_chooses_direction(self):
        params = make_params()
        # Stroke from [10,0] to [0,0]. Reversing avoids 10mm travel from origin.
        s = Stroke([[10, 0], [0, 0]])
        order, dirs = solve_greedy([s], params, start_pos=np.zeros(2))
        # Reversed: start [0,0] is at origin
        assert dirs == [True]

    def test_always_better_than_default(self):
        """On random sets of fragments, optimization must not be worse than default."""
        rng = np.random.default_rng(0)
        params = make_params()
        for trial in range(5):
            strokes = []
            for _ in range(15):
                a = rng.uniform(0, 100, size=2)
                b = a + rng.uniform(-5, 5, size=2)
                strokes.append(Stroke(np.array([a, b])))
            t_default = default_order_time(strokes, params)
            order, dirs = solve_greedy(strokes, params)
            optimized = [strokes[i] for i in order]
            t_optimized = total_time(optimized, dirs, params)
            assert t_optimized <= t_default + 1e-6


class TestTopLevelSolve:
    def test_fallback_to_greedy(self):
        params = make_params()
        s = Stroke([[5, 0], [10, 0]])
        order, dirs = solve([s], params, solver="greedy")
        assert order == [0]


class TestImprovementOnFragmented:
    """The headline metric — does reordering actually save substantial time?"""

    def test_significant_speedup_on_grid_fragments(self):
        """Many short fragments scattered in a grid: random order is wasteful."""
        params = make_params()
        # 5x5 grid of tiny strokes, in row-major (raster scan) order.
        # This is roughly what vtracer's default order looks like.
        strokes = []
        for row in range(5):
            for col in range(5):
                base = np.array([col * 20.0, row * 20.0])
                strokes.append(Stroke(np.array([base, base + [3, 0]])))

        t_default = default_order_time(strokes, params)

        # Now shuffle to be adversarially bad
        rng = np.random.default_rng(42)
        perm = list(range(len(strokes)))
        rng.shuffle(perm)
        shuffled = [strokes[i] for i in perm]
        t_shuffled = default_order_time(shuffled, params)

        order, dirs = solve_greedy(shuffled, params)
        optimized = [shuffled[i] for i in order]
        t_optimized = total_time(optimized, dirs, params)

        # Optimization should at least match (and usually beat) the row-major order
        assert t_optimized <= t_shuffled
        # And should bring shuffled close to or better than the natural raster order
        assert t_optimized < t_shuffled * 0.7  # at least 30% improvement
