"""Phase 1 end-to-end baseline: SVG → strokes → TSP-reorder → gcode + report.

Reports the % improvement of optimized drawing time vs the SVG's default
ordering, using the calibrated motion model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baseline.svg_to_strokes import load_strokes
from baseline.tsp_solver import solve
from common.motion_model import MotionParams
from common.stroke import Stroke, default_order_time, total_time
from robot.gcode_writer import emit_gcode


def fmt_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    sec = s - 60 * m
    return f"{m}m{sec:.1f}s"


def time_breakdown(strokes: list[Stroke],
                   order: list[int],
                   directions: list[bool],
                   params: MotionParams,
                   start_pos: np.ndarray) -> dict:
    """Decompose total time into draw, travel, and pen-toggle components."""
    ordered = [strokes[i] for i in order]
    total = total_time(ordered, directions, params, start_pos)
    draw_t = sum(s.draw_time(params) for s in ordered)
    pen_toggles = len(ordered) * params.t_pen_toggle
    pure_travel = total - draw_t - pen_toggles
    return {
        "total_s": total,
        "draw_s": draw_t,
        "travel_s": max(0.0, pure_travel),
        "pen_toggle_s": pen_toggles,
        "travel_fraction": max(0.0, pure_travel) / total if total > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 baseline: TSP reorder strokes from an SVG"
    )
    parser.add_argument("--svg", type=str, required=True,
                        help="Input SVG file (typically vtracer output)")
    parser.add_argument("--motion-params", type=str, required=True,
                        help="motion_params.json from calibration")
    parser.add_argument("--output", type=str, default=None,
                        help="Output gcode file (default: <svg>.reordered.gcode)")
    parser.add_argument("--solver", choices=("auto", "or-tools", "greedy"),
                        default="auto")
    parser.add_argument("--time-limit", type=int, default=10,
                        help="OR-Tools time limit in seconds")
    parser.add_argument("--samples-per-unit", type=float, default=0.5,
                        help="Curve sampling density (points per mm)")
    parser.add_argument("--report", type=str, default=None,
                        help="Optional path to write a JSON comparison report")
    parser.add_argument("--draw-feed", type=int, default=1800,
                        help="Drawing feedrate in mm/min for the emitted gcode")
    parser.add_argument("--travel-feed", type=int, default=4800,
                        help="Travel feedrate in mm/min for the emitted gcode")
    args = parser.parse_args()

    svg_path = Path(args.svg)
    if not svg_path.exists():
        parser.error(f"SVG not found: {svg_path}")
    params = MotionParams.load(args.motion_params)
    output_path = Path(args.output) if args.output else svg_path.with_suffix(".reordered.gcode")

    print(f"Loading {svg_path} ...")
    strokes = load_strokes(str(svg_path), samples_per_unit=args.samples_per_unit)
    print(f"  Parsed {len(strokes)} strokes")
    print(f"  Total arc length: {sum(s.arc_length() for s in strokes):.1f} mm")

    if not strokes:
        parser.error("No strokes found in SVG")

    start_pos = np.zeros(2)

    # Baseline: default order, all forward
    default_order = list(range(len(strokes)))
    default_dirs = [False] * len(strokes)
    bd = time_breakdown(strokes, default_order, default_dirs, params, start_pos)
    print(f"\n[default order]  total={fmt_seconds(bd['total_s'])}  "
          f"draw={fmt_seconds(bd['draw_s'])}  "
          f"travel={fmt_seconds(bd['travel_s'])} ({bd['travel_fraction']*100:.1f}%)")

    # Optimized
    print(f"\nSolving TSP (solver={args.solver}) ...")
    order, dirs = solve(strokes, params, start_pos=start_pos,
                        solver=args.solver, time_limit_s=args.time_limit)
    bo = time_breakdown(strokes, order, dirs, params, start_pos)
    print(f"[optimized]      total={fmt_seconds(bo['total_s'])}  "
          f"draw={fmt_seconds(bo['draw_s'])}  "
          f"travel={fmt_seconds(bo['travel_s'])} ({bo['travel_fraction']*100:.1f}%)")

    improvement = (bd["total_s"] - bo["total_s"]) / bd["total_s"]
    travel_savings = (bd["travel_s"] - bo["travel_s"]) / max(bd["travel_s"], 1e-9)
    print(f"\nTotal time:  -{improvement*100:.1f}%")
    print(f"Travel time: -{travel_savings*100:.1f}%")

    # Emit gcode
    ordered = [strokes[i] for i in order]
    gcode = emit_gcode(ordered, dirs,
                       draw_feed_mm_min=args.draw_feed,
                       travel_feed_mm_min=args.travel_feed,
                       start_pos=start_pos)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(gcode)
    print(f"\nGcode → {output_path}")

    if args.report:
        report = {
            "svg": str(svg_path),
            "n_strokes": len(strokes),
            "total_arc_length_mm": sum(s.arc_length() for s in strokes),
            "params": params.__dict__,
            "default": bd,
            "optimized": bo,
            "improvement_total_pct": improvement * 100,
            "improvement_travel_pct": travel_savings * 100,
        }
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"Report → {args.report}")


if __name__ == "__main__":
    main()
