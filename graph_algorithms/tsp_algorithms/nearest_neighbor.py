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

    def solve_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        start_time = time.time()
        start_city = start_city or self.cities[0]
        tour, unvisited = [start_city], set(self.cities) - {start_city}
        total_cost, nodes_visited = 0, 0
        current = start_city

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
                if cost < best_cost:
                    best_city, best_cost = city, cost

            if not best_city: break
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
