"""Phase 1 calibration: measure UltraArm P340 motion params from real moves.

Runs a fixed sequence of pen-up and pen-down moves of varying distances, logs
each move's actual elapsed time, and fits (v_travel, accel) for travel moves
and v_draw for drawing moves. Writes MotionParams to JSON.

Local dry-run:
    python -m calibration.measure_motion --mock --output motion_params.json

Real hardware:
    python -m calibration.measure_motion --port /dev/ttyUSB0 --output motion_params.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from common.motion_model import (
    MotionParams,
    fit_draw_speed,
    fit_trapezoidal,
)
from robot.ultraarm_iface import (
    MockUltraArm,
    MoveResult,
    RealUltraArm,
    UltraArmInterface,
)

# Distances (mm) sampled across short-triangular and long-cruise regimes.
DEFAULT_DISTANCES = (2.0, 5.0, 10.0, 20.0, 40.0, 75.0, 120.0, 200.0)
DEFAULT_REPEATS = 3


@dataclass
class MotionSample:
    distance: float
    elapsed: float
    pen_down: bool


def _move_pair(arm: UltraArmInterface,
               a: tuple[float, float],
               b: tuple[float, float],
               pen_down: bool) -> MoveResult:
    """Move arm to point a (rapid), then to point b. Return the b move's timing."""
    arm.move_to(*a)
    if pen_down:
        arm.pen_down()
        result = arm.draw_to(*b)
        arm.pen_up()
    else:
        result = arm.move_to(*b)
    return result


def collect_samples(arm: UltraArmInterface,
                    distances: tuple[float, ...] = DEFAULT_DISTANCES,
                    repeats: int = DEFAULT_REPEATS) -> list[MotionSample]:
    """Drive the arm through a calibration sequence and collect timing samples."""
    samples: list[MotionSample] = []
    arm.home()
    # Centre the calibration motions inside the work area.
    base = np.array([50.0, 50.0])

    for trial in range(repeats):
        for d in distances:
            for pen_down in (False, True):
                # alternate axis to average over different joints
                if (trial + int(pen_down)) % 2 == 0:
                    a = (float(base[0]), float(base[1]))
                    b = (float(base[0] + d), float(base[1]))
                else:
                    a = (float(base[0]), float(base[1]))
                    b = (float(base[0]), float(base[1] + d))
                result = _move_pair(arm, a, b, pen_down=pen_down)
                samples.append(MotionSample(d, result.elapsed, pen_down))

    return samples


def fit_from_samples(samples: list[MotionSample]) -> MotionParams:
    """Fit a MotionParams from collected samples."""
    travel = [s for s in samples if not s.pen_down]
    draw = [s for s in samples if s.pen_down]
    if not travel or not draw:
        raise ValueError("need both pen-up and pen-down samples")

    travel_d = np.array([s.distance for s in travel])
    travel_t = np.array([s.elapsed for s in travel])
    draw_d = np.array([s.distance for s in draw])
    draw_t = np.array([s.elapsed for s in draw])

    v_travel, accel = fit_trapezoidal(travel_d, travel_t)
    v_draw = fit_draw_speed(draw_d, draw_t)

    return MotionParams(
        v_draw=v_draw,
        v_travel=v_travel,
        accel=accel,
        t_pen_toggle=0.3,  # not yet calibrated; nominal value
    )


def summarize(samples: list[MotionSample], params: MotionParams) -> dict:
    """Per-distance mean/std of observed times vs the fit model's prediction."""
    from common.motion_model import draw_time, trapezoidal_time

    grouped: dict[tuple[float, bool], list[float]] = {}
    for s in samples:
        grouped.setdefault((s.distance, s.pen_down), []).append(s.elapsed)

    rows = []
    for (d, pen_down), ts in sorted(grouped.items()):
        ts_arr = np.array(ts)
        if pen_down:
            predicted = float(draw_time(d, params))
        else:
            predicted = float(trapezoidal_time(d, params.v_travel, params.accel))
        rows.append({
            "distance_mm": d,
            "pen_down": pen_down,
            "n": len(ts),
            "mean_s": float(ts_arr.mean()),
            "std_s": float(ts_arr.std()),
            "predicted_s": predicted,
        })
    return {"params": params.__dict__, "samples": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                        help="Use MockUltraArm with known parameters (for dry-run testing)")
    parser.add_argument("--mock-noise", type=float, default=0.05,
                        help="Mock timing noise (relative std). Default: 0.05 (5%%)")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port for real UltraArm (e.g. /dev/ttyUSB0)")
    parser.add_argument("--output", type=str, default="motion_params.json")
    parser.add_argument("--report", type=str, default=None,
                        help="Optional path to write a JSON report with per-distance stats")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    if args.mock and args.port:
        parser.error("--mock and --port are mutually exclusive")
    if not args.mock and not args.port:
        parser.error("specify --mock for dry-run or --port for real hardware")

    if args.mock:
        true_params = MotionParams(
            v_draw=32.0,
            v_travel=85.0,
            accel=230.0,
            t_pen_toggle=0.3,
        )
        arm: UltraArmInterface = MockUltraArm(
            params=true_params,
            noise_std=args.mock_noise,
        )
        print(f"[mock] true params: {true_params}")
    else:
        arm = RealUltraArm(port=args.port)

    with arm:
        samples = collect_samples(arm, repeats=args.repeats)

    fitted = fit_from_samples(samples)
    fitted.save(args.output)

    print(f"\nFitted parameters → {args.output}")
    print(f"  v_draw   = {fitted.v_draw:.2f} mm/s")
    print(f"  v_travel = {fitted.v_travel:.2f} mm/s")
    print(f"  accel    = {fitted.accel:.2f} mm/s²")
    print(f"  pen_toggle = {fitted.t_pen_toggle:.3f} s")

    if args.mock:
        print(f"\n[mock] error vs ground truth:")
        print(f"  v_draw   {100 * abs(fitted.v_draw - true_params.v_draw) / true_params.v_draw:.2f}%")
        print(f"  v_travel {100 * abs(fitted.v_travel - true_params.v_travel) / true_params.v_travel:.2f}%")
        print(f"  accel    {100 * abs(fitted.accel - true_params.accel) / true_params.accel:.2f}%")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(summarize(samples, fitted), f, indent=2)
        print(f"\nReport → {args.report}")


if __name__ == "__main__":
    main()
