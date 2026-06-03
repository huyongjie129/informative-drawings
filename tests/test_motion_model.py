"""Tests for the trapezoidal motion model and parameter fitting."""

from __future__ import annotations

import numpy as np
import pytest

from common.motion_model import (
    MotionParams,
    draw_time,
    fit_draw_speed,
    fit_trapezoidal,
    trapezoidal_time,
    travel_time,
)


class TestTrapezoidalTime:
    def test_long_distance_reaches_cruise(self):
        v_max, a = 80.0, 200.0
        d_critical = v_max * v_max / a   # 32 mm
        d = 200.0  # well beyond cruise threshold
        t = trapezoidal_time(d, v_max, a)
        expected = d / v_max + v_max / a
        assert t == pytest.approx(expected)

    def test_short_distance_triangular(self):
        v_max, a = 80.0, 200.0
        d = 5.0  # below cruise threshold 32 mm
        t = trapezoidal_time(d, v_max, a)
        expected = 2.0 * np.sqrt(d / a)
        assert t == pytest.approx(expected)

    def test_continuity_at_critical_distance(self):
        v_max, a = 80.0, 200.0
        d_crit = v_max * v_max / a
        long_t = d_crit / v_max + v_max / a       # 0.4 + 0.4 = 0.8
        tri_t = 2.0 * np.sqrt(d_crit / a)         # 2*sqrt(0.16) = 0.8
        assert long_t == pytest.approx(tri_t)
        # the function should be continuous at the switch
        t_just_below = trapezoidal_time(d_crit - 1e-6, v_max, a)
        t_just_above = trapezoidal_time(d_crit + 1e-6, v_max, a)
        assert t_just_below == pytest.approx(t_just_above, abs=1e-3)

    def test_zero_distance_zero_time(self):
        assert trapezoidal_time(0.0, 80.0, 200.0) == pytest.approx(0.0)

    def test_vectorized(self):
        v_max, a = 80.0, 200.0
        d = np.array([1.0, 10.0, 50.0, 200.0])
        t = trapezoidal_time(d, v_max, a)
        assert t.shape == (4,)
        for di, ti in zip(d, t):
            assert ti == pytest.approx(trapezoidal_time(float(di), v_max, a))


class TestParamFitting:
    def test_recover_known_params(self):
        v_true, a_true = 75.0, 250.0
        distances = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0])
        times = trapezoidal_time(distances, v_true, a_true)
        v_fit, a_fit = fit_trapezoidal(distances, times)
        assert v_fit == pytest.approx(v_true, rel=1e-3)
        assert a_fit == pytest.approx(a_true, rel=1e-3)

    def test_fitting_robust_to_noise(self):
        v_true, a_true = 80.0, 200.0
        distances = np.array([5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
        rng = np.random.default_rng(42)
        clean = trapezoidal_time(distances, v_true, a_true)
        noisy = clean + rng.normal(0, 0.02 * clean, size=clean.shape)
        v_fit, a_fit = fit_trapezoidal(distances, noisy)
        assert v_fit == pytest.approx(v_true, rel=0.1)
        assert a_fit == pytest.approx(a_true, rel=0.2)

    def test_draw_speed_fit(self):
        v_true = 35.0
        distances = np.array([5.0, 10.0, 20.0, 50.0, 100.0])
        times = distances / v_true
        v_fit = fit_draw_speed(distances, times)
        assert v_fit == pytest.approx(v_true, rel=1e-6)


class TestMotionParams:
    def test_default_sane(self):
        p = MotionParams.default()
        assert p.v_draw > 0
        assert p.v_travel > p.v_draw
        assert p.accel > 0
        assert 0 <= p.t_pen_toggle <= 2.0

    def test_save_load_roundtrip(self, tmp_path):
        p = MotionParams(v_draw=33.0, v_travel=77.0, accel=222.0, t_pen_toggle=0.4)
        path = tmp_path / "params.json"
        p.save(path)
        loaded = MotionParams.load(path)
        assert loaded == p

    def test_travel_includes_pen_toggle(self):
        p = MotionParams(v_draw=30.0, v_travel=80.0, accel=200.0, t_pen_toggle=0.5)
        d = 100.0
        pure_travel = trapezoidal_time(d, p.v_travel, p.accel)
        assert travel_time(d, p) == pytest.approx(pure_travel + 0.5)

    def test_draw_time_uses_v_draw(self):
        p = MotionParams(v_draw=25.0, v_travel=80.0, accel=200.0)
        assert draw_time(100.0, p) == pytest.approx(4.0)
