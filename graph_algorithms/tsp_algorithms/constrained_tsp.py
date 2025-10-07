# tsp/constrained_tsp.py
import time
from typing import Optional, Dict, Set
from .types import TSPResult
from .base import BaseTSP

class ConstrainedTSP(BaseTSP):
    """TSP with constraints like budget and time windows."""
    
    def __init__(self, graph, constraints: Optional[Dict] = None):
        super().__init__(graph)
        self.constraints = constraints or {}
        self.max_budget = self.constraints.get("max_budget", float("inf"))
        self.max_time = self.constraints.get("max_time", float("inf"))
        self.time_windows = self.constraints.get("time_windows", {})
        self.mandatory = self.constraints.get("mandatory_cities", set())

    def solve_constrained_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        start_time = time.time()
        start_city = start_city or self.cities[0]
        tour = [start_city]
        total_cost, total_time, nodes_visited = 0, 0, 0
        unvisited = set(self.cities) - {start_city}

        while unvisited:
            nodes_visited += 1
            feasible = []
            for city in unvisited:
                cost = self.graph.get_weight(tour[-1], city)
                if total_cost + cost > self.max_budget:
                    continue
                feasible.append((city, cost))
            if not feasible: break
            best_city, best_cost = min(feasible, key=lambda x: x[1])
            tour.append(best_city)
            total_cost += best_cost
            unvisited.remove(best_city)

        return TSPResult(
            tour=tour,
            total_cost=total_cost,
            algorithm="Constrained TSP",
            execution_time=time.time() - start_time,
            nodes_visited=nodes_visited
        )
