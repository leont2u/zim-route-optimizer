from .dijkstra import DijkstraAlgorithm
from .bellman_ford import BellmanFordAlgorithm
from .models import PathResult
import random

class ShortestPathComparator:
    """Compare and benchmark shortest path algorithms."""

    def __init__(self, graph):
        self.graph = graph
        self.dijkstra = DijkstraAlgorithm(graph)
        self.bellman_ford = BellmanFordAlgorithm(graph)

    def compare_algorithms(self, start: str, end: str):
        results = {}
        try:
            results["Dijkstra"] = self.dijkstra.find_shortest_path(start, end)
        except Exception as e:
            results["Dijkstra"] = PathResult(-1, [], "Dijkstra", 0, 0)
            print(f"Dijkstra error: {e}")

        try:
            results["Bellman-Ford"] = self.bellman_ford.find_shortest_path(start, end)
        except Exception as e:
            results["Bellman-Ford"] = PathResult(-1, [], "Bellman-Ford", 0, 0)
            print(f"Bellman-Ford error: {e}")

        return results

    def benchmark_algorithms(self, iterations: int = 50):
        cities = self.graph.get_all_cities()
        stats = {
            "Dijkstra": {"total_time": 0, "total_nodes": 0, "successful_runs": 0},
            "Bellman-Ford": {"total_time": 0, "total_nodes": 0, "successful_runs": 0},
        }

        for _ in range(iterations):
            start = random.choice(cities)
            end = random.choice([c for c in cities if c != start])
            for algo, instance in [
                ("Dijkstra", self.dijkstra),
                ("Bellman-Ford", self.bellman_ford),
            ]:
                try:
                    result = instance.find_shortest_path(start, end)
                    if result.distance >= 0:
                        stats[algo]["total_time"] += result.execution_time
                        stats[algo]["total_nodes"] += result.nodes_visited
                        stats[algo]["successful_runs"] += 1
                except:
                    pass

        for algo in stats:
            runs = stats[algo]["successful_runs"]
            stats[algo]["avg_time"] = stats[algo]["total_time"] / runs if runs else 0
            stats[algo]["avg_nodes"] = stats[algo]["total_nodes"] / runs if runs else 0

        return stats
