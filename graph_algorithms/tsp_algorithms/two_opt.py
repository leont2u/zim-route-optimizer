# tsp/two_opt.py
import time
from typing import Optional
from .types import TSPResult
from .base import BaseTSP
from .nearest_neighbor import NearestNeighborTSP

class TwoOptTSP(BaseTSP):
    """Two-opt improvement heuristic for TSP."""

    def calculate_tour_cost(self, tour):
        total = 0
        for i in range(len(tour)):
            c1, c2 = tour[i], tour[(i+1) % len(tour)]
            total += self.graph.get_weight(c1, c2)
        return total

    def solve_tsp(self, start_city: Optional[str] = None, max_iterations: int = 1000) -> TSPResult:
        start_time = time.time()
        start_city = start_city or self.cities[0]
        nn_result = NearestNeighborTSP(self.graph).solve_tsp(start_city)
        tour, best_cost = nn_result.tour.copy(), nn_result.total_cost
        improved, iteration, nodes_visited = True, 0, nn_result.nodes_visited

        while improved and iteration < max_iterations:
            improved, iteration = False, iteration + 1
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    nodes_visited += 1
                    old = (self.graph.get_weight(tour[i-1], tour[i]) +
                           self.graph.get_weight(tour[j], tour[(j+1) % len(tour)]))
                    new = (self.graph.get_weight(tour[i-1], tour[j]) +
                           self.graph.get_weight(tour[i], tour[(j+1) % len(tour)]))
                    if new < old:
                        tour[i:j+1] = reversed(tour[i:j+1])
                        best_cost = self.calculate_tour_cost(tour)
                        improved = True
                        break
                if improved:
                    break

        return TSPResult(
            tour=tour,
            total_cost=best_cost,
            algorithm="Two-Opt",
            execution_time=time.time() - start_time,
            nodes_visited=nodes_visited
        )
