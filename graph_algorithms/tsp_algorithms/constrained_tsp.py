# tsp/constrained_tsp.py
import time
from typing import Optional, Dict, Set, List
from .types import TSPResult
from .base import BaseTSP
from ..shortest_path.dijkstra import DijkstraAlgorithm
from .held_karp import HeldKarpAlgorithm


class ConstrainedTSP(BaseTSP):
    """TSP with constraints like mandatory stops, budget and time.

    Implementation approach:
    - If mandatory stops are provided, build a reduced complete graph whose
      vertices are {start} ∪ mandatory_stops. Edge weights are shortest-path
      distances computed by Dijkstra on the full graph.
    - Solve the reduced TSP exactly with Held-Karp (reasonable when the
      number of mandatory stops is small). Expand each reduced edge into the
      full city-level path using the Dijkstra path results.
    - Finally validate budget/time constraints against the expanded tour.
    """

    def __init__(self, graph, constraints: Optional[Dict] = None):
        super().__init__(graph)
        self.constraints = constraints or {}
        self.max_budget = self.constraints.get("max_budget", float("inf"))
        self.max_time = self.constraints.get("max_time", float("inf"))
        self.time_windows = self.constraints.get("time_windows", {})
        self.mandatory: Set[str] = set(self.constraints.get("mandatory_cities", set()))

    def _build_reduced_graph(self, nodes: List[str]):
        """Return a lightweight graph object (get_all_cities, get_weight) and
        a dict of shortest path sequences between node pairs.
        """
        dijkstra = DijkstraAlgorithm(self.graph)
        n = len(nodes)
        # distances[u][v] float, paths[(u,v)] = list of cities from u->v
        distances = {u: {} for u in nodes}
        paths = {}
        for u in nodes:
            for v in nodes:
                if u == v:
                    distances[u][v] = 0.0
                    paths[(u, v)] = [u]
                    continue
                res = dijkstra.find_shortest_path(u, v)
                if res.distance < 0:
                    # unreachable
                    distances[u][v] = float('inf')
                    paths[(u, v)] = []
                else:
                    distances[u][v] = res.distance
                    paths[(u, v)] = res.path

        class SimpleGraph:
            def __init__(self, nodes, distances):
                self._nodes = list(nodes)
                self._dist = distances

            def get_all_cities(self):
                return list(self._nodes)

            def get_weight(self, a, b):
                return self._dist.get(a, {}).get(b, float('inf'))

        return SimpleGraph(nodes, distances), paths

    def solve_constrained_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        start_time = time.time()
        start_city = start_city or (self.cities[0] if self.cities else None)
        if not start_city:
            return TSPResult(tour=[], total_cost=-1, algorithm="Constrained TSP", execution_time=0, nodes_visited=0)

        # If no mandatory constraints, fallback to a greedy constrained solver
        if not self.mandatory:
            # simple greedy respecting max_budget
            tour = [start_city]
            total_cost = 0.0
            nodes_visited = 0
            unvisited = set(self.cities) - {start_city}
            while unvisited:
                nodes_visited += 1
                feasible = []
                for city in unvisited:
                    cost = self.graph.get_weight(tour[-1], city)
                    if total_cost + cost > self.max_budget:
                        continue
                    feasible.append((city, cost))
                if not feasible:
                    break
                best_city, best_cost = min(feasible, key=lambda x: x[1])
                tour.append(best_city)
                total_cost += best_cost
                unvisited.remove(best_city)

            # close tour
            if tour and tour[0] == start_city and tour[-1] != start_city:
                return_cost = self.graph.get_weight(tour[-1], start_city)
                total_cost += return_cost
                tour.append(start_city)

            if total_cost > self.max_budget:
                return TSPResult(tour=[], total_cost=-1, algorithm="Constrained TSP", execution_time=time.time() - start_time, nodes_visited=nodes_visited)

            return TSPResult(
                tour=tour,
                total_cost=total_cost,
                algorithm="Constrained TSP (greedy)",
                execution_time=time.time() - start_time,
                nodes_visited=nodes_visited,
            )

        # Build reduced node set: start + mandatory (preserve start first)
        reduced_nodes = [start_city] + [c for c in self.mandatory if c != start_city]

        # compute pairwise shortest paths among reduced nodes
        reduced_graph, paths = self._build_reduced_graph(reduced_nodes)

        # Check connectivity
        for u in reduced_nodes:
            for v in reduced_nodes:
                if u == v: continue
                if reduced_graph.get_weight(u, v) == float('inf'):
                    # unreachable required pair
                    return TSPResult(tour=[], total_cost=-1, algorithm="Constrained TSP", execution_time=time.time() - start_time, nodes_visited=0)

        # Solve Held-Karp on the reduced graph
        solver = HeldKarpAlgorithm(reduced_graph)
        reduced_result = solver.solve_tsp(start_city=start_city, constraints=self.constraints)
        reduced_tour = reduced_result.tour

        # Expand reduced tour into full-city-level tour using paths
        full_tour: List[str] = []
        total_cost = 0.0
        nodes_visited = reduced_result.nodes_visited
        for i in range(len(reduced_tour) - 1):
            a, b = reduced_tour[i], reduced_tour[i+1]
            segment = paths.get((a, b), [])
            if not segment:
                return TSPResult(tour=[], total_cost=-1, algorithm="Constrained TSP", execution_time=time.time() - start_time, nodes_visited=nodes_visited)
            # avoid duplicating intermediate nodes
            if full_tour and full_tour[-1] == segment[0]:
                full_tour.extend(segment[1:])
            else:
                full_tour.extend(segment)
            total_cost += reduced_graph.get_weight(a, b)

        # Validate budget/time after expansion
        if total_cost > self.max_budget or total_cost != total_cost:  # NaN guard
            return TSPResult(tour=[], total_cost=-1, algorithm="Constrained TSP", execution_time=time.time() - start_time, nodes_visited=nodes_visited)

        return TSPResult(
            tour=full_tour,
            total_cost=total_cost,
            algorithm="Constrained TSP (reduced Held-Karp)",
            execution_time=time.time() - start_time,
            nodes_visited=nodes_visited,
        )
