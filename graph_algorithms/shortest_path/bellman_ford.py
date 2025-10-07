# algorithms/bellman_ford.py
import time
from typing import Dict
from .models import PathResult

class BellmanFordAlgorithm:
    """Implementation of Bellman-Ford shortest path algorithm."""

    def __init__(self, graph):
        self.graph = graph
        self.cities = graph.get_all_cities()

    def find_shortest_path(self, start: str, end: str) -> PathResult:
        start_time = time.time()

        if start not in self.cities or end not in self.cities:
            raise ValueError(f"City not found: {start} or {end}")

        if start == end:
            return PathResult(0, [start], "Bellman-Ford", 0, 1)

        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0
        nodes_visited = 0

        for _ in range(len(self.cities) - 1):
            for city in self.cities:
                nodes_visited += 1
                for neighbor in self.graph.get_neighbors(city):
                    edge_weight = self.graph.get_weight(city, neighbor)
                    if distances[city] + edge_weight < distances[neighbor]:
                        distances[neighbor] = distances[city] + edge_weight
                        previous[neighbor] = city

        for city in self.cities:
            for neighbor in self.graph.get_neighbors(city):
                edge_weight = self.graph.get_weight(city, neighbor)
                if distances[city] + edge_weight < distances[neighbor]:
                    return PathResult(-1, [], "Bellman-Ford", time.time() - start_time, nodes_visited)

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
            algorithm="Bellman-Ford",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )
