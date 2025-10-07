# tsp/nearest_neighbor.py
import time
from typing import Optional
from .types import TSPResult
from .base import BaseTSP
from ..shortest_path.dijkstra import DijkstraAlgorithm

class NearestNeighborTSP(BaseTSP):
    """Nearest Neighbor heuristic for TSP."""
    
    def __init__(self, graph):
        super().__init__(graph)
        self.dijkstra = DijkstraAlgorithm(graph)

    def solve_tsp(self, start_city: Optional[str] = None, constraints: dict = None) -> TSPResult:
        start_time = time.time()
        constraints = constraints or {}
        max_budget = constraints.get('max_budget', float('inf'))
        mandatory = set(constraints.get('mandatory_cities', []))
        start_city = start_city or self.cities[0]
        tour = [start_city]
        unvisited = set(self.cities) - {start_city}
        total_cost, nodes_visited = 0, 0
        current = start_city

        # First, ensure mandatory nodes are visited (greedy among mandatory)
        remaining_mandatory = set(mandatory) & set(unvisited)
        while remaining_mandatory:
            nodes_visited += 1
            best_city, best_cost = None, float('inf')
            for city in remaining_mandatory:
                cost = self.graph.get_weight(current, city)
                if cost <= 0:
                    try:
                        cost = self.dijkstra.find_shortest_path(current, city).distance
                    except:
                        cost = float('inf')
                # if mandatory city itself is unreachable or would violate budget, we still try to pick it (to surface infeasibility later)
                if cost < best_cost:
                    best_city, best_cost = city, cost

            if not best_city:
                break
            tour.append(best_city)
            total_cost += best_cost
            if best_city in unvisited:
                unvisited.remove(best_city)
            remaining_mandatory.discard(best_city)
            current = best_city

        # Then visit remaining nodes with nearest-neighbor heuristic while respecting budget
        while unvisited:
            best_city, best_cost = None, float('inf')
            nodes_visited += 1
            for city in unvisited:
                cost = self.graph.get_weight(current, city)
                if cost <= 0:
                    try:
                        cost = self.dijkstra.find_shortest_path(current, city).distance
                    except:
                        continue
                # If selecting city would immediately violate budget, skip it
                if total_cost + cost > max_budget:
                    continue
                if cost < best_cost:
                    best_city, best_cost = city, cost

            if not best_city:
                break
            tour.append(best_city)
            total_cost += best_cost
            unvisited.remove(best_city)
            current = best_city

        total_cost += self.graph.get_weight(current, start_city)
        # close the tour by returning to the start city
        if not tour or tour[0] != start_city:
            tour.insert(0, start_city)
        if tour[-1] != start_city:
            tour.append(start_city)

        return TSPResult(
            tour=tour,
            total_cost=total_cost,
            algorithm="Nearest Neighbor",
            execution_time=time.time() - start_time,
            nodes_visited=nodes_visited
        )
