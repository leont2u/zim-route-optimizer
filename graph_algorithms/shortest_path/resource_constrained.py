import time
import heapq
from typing import Dict, List, Tuple
from .models import PathResult


def resource_constrained_shortest_path(graph, start: str, end: str, max_cost: float = float('inf'), cost_matrix: Dict[Tuple[str, str], float] = None):
    """
    Label-setting multi-criteria shortest path that tracks (distance, cost).

    Args:
        graph: ZimbabweGraph instance with get_neighbors and get_weight
        start, end: city ids
        max_cost: maximum allowed cost (budget). If inf, no budget restriction.
        cost_matrix: optional dict (from,to)->cost if different metric is needed

    Returns PathResult where distance is minimal distance under cost constraint or -1 if infeasible.
    """
    start_time = time.time()
    cities = graph.get_all_cities()
    if start not in cities or end not in cities:
        raise ValueError("Start or end city not in graph")

    # Each label is (distance, cost, city, prev_city)
    # We'll store best labels per city and prune dominated labels.
    labels: Dict[str, List[Tuple[float, float, str]]] = {c: [] for c in cities}

    # priority queue ordered by distance then cost
    pq: List[Tuple[float, float, str, str]] = []  # (dist, cost, city, prev)
    heapq.heappush(pq, (0.0, 0.0, start, None))
    labels[start].append((0.0, 0.0, None))

    prev_map: Dict[Tuple[str, float, float], Tuple[str, float, float]] = {}
    nodes_visited = 0

    best_label_for_end = None

    while pq:
        dist_u, cost_u, u, prev = heapq.heappop(pq)
        nodes_visited += 1

        # Early exit if this label is dominated by a better known label for u
        dominated = False
        for (ld, lc, _) in labels[u]:
            if ld <= dist_u and lc <= cost_u and (ld < dist_u or lc < cost_u):
                dominated = True
                break
        if dominated:
            continue

        if u == end:
            best_label_for_end = (dist_u, cost_u)
            break

        for v in graph.get_neighbors(u):
            edge_dist = graph.get_weight(u, v)
            edge_cost = cost_matrix.get((u, v), edge_dist) if cost_matrix else edge_dist
            new_dist = dist_u + edge_dist
            new_cost = cost_u + edge_cost
            if new_cost > max_cost:
                continue

            # Check if label is dominated at v
            skip = False
            for (ld, lc, _) in labels[v]:
                if ld <= new_dist and lc <= new_cost:
                    skip = True
                    break
            if skip:
                continue

            # Otherwise add label and push
            labels[v].append((new_dist, new_cost, u))
            prev_map[(v, new_dist, new_cost)] = (u, dist_u, cost_u)
            heapq.heappush(pq, (new_dist, new_cost, v, u))

    if not best_label_for_end:
        execution_time = time.time() - start_time
        return PathResult(distance=-1, path=[], algorithm="Resource-Constrained Dijkstra", execution_time=execution_time, nodes_visited=nodes_visited)

    # Reconstruct path from prev_map using the best label we recorded for end
    dist_end, cost_end = best_label_for_end
    # Find the matching label entry in labels[end]
    chosen = None
    for (ld, lc, p) in labels[end]:
        if abs(ld - dist_end) < 1e-9 and abs(lc - cost_end) < 1e-9:
            chosen = (ld, lc, p)
            break

    path = [end]
    cur = end
    cur_ld, cur_lc = dist_end, cost_end
    while True:
        key = (cur, cur_ld, cur_lc)
        if key not in prev_map:
            break
        prev_entry = prev_map[key]
        prev_city, prev_ld, prev_lc = prev_entry
        path.append(prev_city)
        cur, cur_ld, cur_lc = prev_city, prev_ld, prev_lc
        if prev_city == start:
            break

    path.reverse()
    execution_time = time.time() - start_time
    return PathResult(distance=dist_end, path=path, algorithm="Resource-Constrained Dijkstra", execution_time=execution_time, nodes_visited=nodes_visited)
