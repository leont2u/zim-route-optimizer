# algorithms/dijkstra.py
import heapq
import time
from typing import Dict
from .models import PathResult

class DijkstraAlgorithm:
    """
    Implementation of Dijkstra's shortest path algorithm.
    Time Complexity: O((V + E) log V)
    """

    def __init__(self, graph):
        self.graph = graph
        self.cities = graph.get_all_cities()

    def find_shortest_path(self, start: str, end: str) -> PathResult:
        start_time = time.time()
        
        if start not in self.cities or end not in self.cities:
            raise ValueError(f"City not found: {start} or {end}")
        
        if start == end:
            return PathResult(0, [start], "Dijkstra", 0, 1)

        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0

        pq = [(0, start)]
        visited = set()
        nodes_visited = 0

        while pq:
            current_dist, current_city = heapq.heappop(pq)
            if current_city in visited:
                continue
            visited.add(current_city)
            nodes_visited += 1

            if current_city == end:
                break

            for neighbor in self.graph.get_neighbors(current_city):
                edge_weight = self.graph.get_weight(current_city, neighbor)
                new_dist = current_dist + edge_weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_city
                    heapq.heappush(pq, (new_dist, neighbor))

        path = []
        if distances[end] != float('inf'):
            current = end
            while current:
                path.append(current)
                current = previous[current]
            path.reverse()

        execution_time = time.time() - start_time
        return PathResult(
            distance=distances[end] if distances[end] != float('inf') else -1,
            path=path,
            algorithm="Dijkstra",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )
