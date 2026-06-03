"""TSP solver for stroke ordering with direction choice.

Models the problem as a 2N-node asymmetric TSP:
  - Each stroke i contributes two nodes: A_i (its start) and B_i (its end).
  - Within-stroke edges (A_i ↔ B_i) have very negative cost ⇒ the optimal
    tour is forced to use them as required "drawing" edges.
  - Inter-stroke edges carry the pen-up travel time between endpoints.
  - A virtual depot at start_pos (default origin) anchors the tour so we
    optimize the open-route cost (depot → ... → back to depot, where the
    closing leg is free).

Returns (order, directions): a permutation of stroke indices and a bool per
stroke indicating whether to draw it reversed (B_i → A_i).

Solver chain: try OR-Tools, fall back to greedy nearest-neighbor + 2-opt.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from common.motion_model import MotionParams, travel_time
from common.stroke import Stroke

# How much to scale floating-point times into integer costs for OR-Tools.
COST_SCALE = 1000


def _endpoint_matrix(strokes: Sequence[Stroke],
                     start_pos: np.ndarray) -> np.ndarray:
    """Return (2N+1, 2) array of endpoints; row 0 is the depot."""
    pts = [start_pos]
    for s in strokes:
        pts.append(s.start)
        pts.append(s.end)
    return np.array(pts, dtype=np.float64)


def _pairwise_travel(endpoints: np.ndarray, params: MotionParams) -> np.ndarray:
    diffs = endpoints[:, None, :] - endpoints[None, :, :]
    dist = np.linalg.norm(diffs, axis=-1)
    times = np.asarray(travel_time(dist, params))
    # Self-loops have zero distance; travel_time adds pen_toggle. Zero them out.
    np.fill_diagonal(times, 0.0)
    return times


def _build_cost_matrix(strokes: Sequence[Stroke],
                       params: MotionParams,
                       start_pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Build the 2N+1 cost matrix. Returns (matrix, within_stroke_subsidy_total)."""
    n = len(strokes)
    endpoints = _endpoint_matrix(strokes, start_pos)
    travel = _pairwise_travel(endpoints, params)

    # Subsidy: any within-stroke edge gets cost = -SUBSIDY so the solver is
    # forced to include all N of them. SUBSIDY must exceed the worst-case
    # extra cost incurred by skipping a required edge.
    subsidy = float(travel.max() * 2 + 1.0)
    total_subsidy = subsidy * n

    cost = travel.copy()
    for i in range(n):
        a = 1 + 2 * i
        b = 2 + 2 * i
        cost[a, b] = -subsidy
        cost[b, a] = -subsidy
        # Depot to/from any stroke endpoint: just travel time (no draw cost yet)

    return cost, total_subsidy


def _decode_tour(tour: Sequence[int], n_strokes: int) -> tuple[list[int], list[bool]]:
    """Convert a node-sequence tour into (stroke order, reversed flags).

    The tour starts at node 0 (depot) and ends back at node 0. Between, it
    visits A/B node pairs for each stroke in some order. For stroke i:
      - if A_i is visited before B_i ⇒ draw forward (reversed=False)
      - if B_i is visited before A_i ⇒ draw reversed (reversed=True)
    """
    order: list[int] = []
    directions: list[bool] = []
    # Trim leading/trailing depot if present
    inner = [x for x in tour if x != 0]
    seen = set()
    for node in inner:
        idx = (node - 1) // 2  # which stroke
        if idx in seen:
            continue  # second endpoint, already accounted for
        seen.add(idx)
        # Is this the A (start) node or B (end) node?
        is_a = (node == 1 + 2 * idx)
        order.append(idx)
        directions.append(not is_a)  # entering at B means we draw reversed
    return order, directions


def solve_or_tools(strokes: Sequence[Stroke],
                   params: MotionParams,
                   start_pos: np.ndarray | None = None,
                   time_limit_s: int = 10) -> tuple[list[int], list[bool]]:
    """OR-Tools backend. Raises ImportError if ortools is missing."""
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    if start_pos is None:
        start_pos = np.zeros(2)

    cost, _subsidy = _build_cost_matrix(strokes, params, start_pos)
    n_nodes = cost.shape[0]

    manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    int_cost = (cost * COST_SCALE).round().astype(np.int64)

    def cost_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(int_cost[i, j])

    transit_idx = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    params_or = pywrapcp.DefaultRoutingSearchParameters()
    params_or.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params_or.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params_or.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params_or)
    if solution is None:
        raise RuntimeError("OR-Tools failed to find a solution")

    tour = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    tour.append(manager.IndexToNode(index))

    return _decode_tour(tour, len(strokes))


def solve_greedy(strokes: Sequence[Stroke],
                 params: MotionParams,
                 start_pos: np.ndarray | None = None,
                 two_opt_passes: int = 3) -> tuple[list[int], list[bool]]:
    """Greedy nearest-neighbor with direction choice, refined by 2-opt."""
    if start_pos is None:
        start_pos = np.zeros(2)

    n = len(strokes)
    remaining = set(range(n))
    order: list[int] = []
    directions: list[bool] = []
    pos = np.asarray(start_pos, dtype=np.float64)

    # Nearest neighbor (with direction)
    while remaining:
        best_idx, best_dir, best_t = None, None, float("inf")
        for i in remaining:
            for rev in (False, True):
                start = strokes[i].end if rev else strokes[i].start
                t = float(travel_time(float(np.linalg.norm(start - pos)), params))
                if t < best_t:
                    best_t = t
                    best_idx = i
                    best_dir = rev
        assert best_idx is not None
        order.append(best_idx)
        directions.append(bool(best_dir))
        s = strokes[best_idx].reversed() if best_dir else strokes[best_idx]
        pos = s.end
        remaining.discard(best_idx)

    # 2-opt-ish improvement: try swapping adjacent strokes and try flipping
    # direction of each stroke; keep change if total time drops.
    for _ in range(two_opt_passes):
        improved = False
        for i in range(n):
            cur_dir = directions[i]
            directions[i] = not cur_dir
            t_new = _eval(strokes, order, directions, params, start_pos)
            directions[i] = cur_dir
            t_old = _eval(strokes, order, directions, params, start_pos)
            if t_new < t_old:
                directions[i] = not cur_dir
                improved = True
        for i in range(n - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            directions[i], directions[i + 1] = directions[i + 1], directions[i]
            t_new = _eval(strokes, order, directions, params, start_pos)
            order[i], order[i + 1] = order[i + 1], order[i]
            directions[i], directions[i + 1] = directions[i + 1], directions[i]
            t_old = _eval(strokes, order, directions, params, start_pos)
            if t_new < t_old:
                order[i], order[i + 1] = order[i + 1], order[i]
                directions[i], directions[i + 1] = directions[i + 1], directions[i]
                improved = True
        if not improved:
            break

    return order, directions


def _eval(strokes: Sequence[Stroke],
          order: Sequence[int],
          directions: Sequence[bool],
          params: MotionParams,
          start_pos: np.ndarray) -> float:
    from common.stroke import total_time
    ordered = [strokes[i] for i in order]
    return total_time(ordered, list(directions), params, start_pos)


def solve(strokes: Sequence[Stroke],
          params: MotionParams,
          start_pos: np.ndarray | None = None,
          solver: str = "auto",
          time_limit_s: int = 10) -> tuple[list[int], list[bool]]:
    """Top-level entry; tries OR-Tools then falls back to greedy."""
    if not strokes:
        return [], []
    if solver in ("auto", "or-tools"):
        try:
            return solve_or_tools(strokes, params, start_pos, time_limit_s)
        except ImportError:
            if solver == "or-tools":
                raise
        except Exception as e:
            if solver == "or-tools":
                raise
            print(f"[tsp_solver] OR-Tools failed ({e}); falling back to greedy")
    if solver in ("auto", "greedy"):
        return solve_greedy(strokes, params, start_pos)
    raise ValueError(f"Unknown solver: {solver}")
