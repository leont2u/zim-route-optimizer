"""
Shortest Path Algorithms Implementation
Contains Dijkstra's algorithm and Bellman-Ford algorithm for finding shortest paths.
"""

import heapq
import sys
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import time

@dataclass
class PathResult:
    """Result of a shortest path algorithm."""
    distance: float
    path: List[str]
    algorithm: str
    execution_time: float
    nodes_visited: int

class DijkstraAlgorithm:
    """
    Implementation of Dijkstra's shortest path algorithm.
    Time Complexity: O((V + E) log V) with binary heap
    Space Complexity: O(V)
    """
    
    def __init__(self, graph):
        """
        Initialize Dijkstra's algorithm with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
        self.city_to_index = {city: i for i, city in enumerate(self.cities)}
        self.index_to_city = {i: city for i, city in enumerate(self.cities)}
    
    def find_shortest_path(self, start: str, end: str) -> PathResult:
        """
        Find shortest path from start to end city.
        
        Args:
            start: Starting city name
            end: Destination city name
            
        Returns:
            PathResult object with distance, path, and metadata
        """
        start_time = time.time()
        
        if start not in self.cities or end not in self.cities:
            raise ValueError(f"City not found: {start} or {end}")
        
        if start == end:
            return PathResult(0, [start], "Dijkstra", 0, 1)
        
        # Initialize distances and previous nodes
        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0
        
        # Priority queue: (distance, city)
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
            
            # Check all neighbors
            for neighbor in self.graph.get_neighbors(current_city):
                if neighbor in visited:
                    continue
                
                edge_weight = self.graph.get_weight(current_city, neighbor)
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_city
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct path
        path = []
        if distances[end] != float('inf'):
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
        
        execution_time = time.time() - start_time
        
        return PathResult(
            distance=distances[end] if distances[end] != float('inf') else -1,
            path=path if path else [],
            algorithm="Dijkstra",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )
    
    def find_all_shortest_paths(self, start: str) -> Dict[str, PathResult]:
        """
        Find shortest paths from start to all other cities.
        
        Args:
            start: Starting city name
            
        Returns:
            Dictionary mapping destination cities to PathResult objects
        """
        start_time = time.time()
        
        if start not in self.cities:
            raise ValueError(f"City not found: {start}")
        
        # Initialize distances and previous nodes
        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0
        
        # Priority queue: (distance, city)
        pq = [(0, start)]
        visited = set()
        nodes_visited = 0
        
        while pq:
            current_dist, current_city = heapq.heappop(pq)
            
            if current_city in visited:
                continue
                
            visited.add(current_city)
            nodes_visited += 1
            
            # Check all neighbors
            for neighbor in self.graph.get_neighbors(current_city):
                if neighbor in visited:
                    continue
                
                edge_weight = self.graph.get_weight(current_city, neighbor)
                new_dist = current_dist + edge_weight
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_city
                    heapq.heappush(pq, (new_dist, neighbor))
        
        # Reconstruct paths for all destinations
        results = {}
        execution_time = time.time() - start_time
        
        for end_city in self.cities:
            if end_city == start:
                results[end_city] = PathResult(0, [start], "Dijkstra", execution_time, nodes_visited)
                continue
            
            path = []
            if distances[end_city] != float('inf'):
                current = end_city
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
            
            results[end_city] = PathResult(
                distance=distances[end_city] if distances[end_city] != float('inf') else -1,
                path=path if path else [],
                algorithm="Dijkstra",
                execution_time=execution_time,
                nodes_visited=nodes_visited
            )
        
        return results

class BellmanFordAlgorithm:
    """
    Implementation of Bellman-Ford shortest path algorithm.
    Time Complexity: O(VE)
    Space Complexity: O(V)
    Can handle negative weights and detect negative cycles.
    """
    
    def __init__(self, graph):
        """
        Initialize Bellman-Ford algorithm with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
    
    def find_shortest_path(self, start: str, end: str) -> PathResult:
        """
        Find shortest path from start to end city using Bellman-Ford.
        
        Args:
            start: Starting city name
            end: Destination city name
            
        Returns:
            PathResult object with distance, path, and metadata
        """
        start_time = time.time()
        
        if start not in self.cities or end not in self.cities:
            raise ValueError(f"City not found: {start} or {end}")
        
        if start == end:
            return PathResult(0, [start], "Bellman-Ford", 0, 1)
        
        # Initialize distances and previous nodes
        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0
        
        nodes_visited = 0
        
        # Relax edges V-1 times
        for _ in range(len(self.cities) - 1):
            for city in self.cities:
                nodes_visited += 1
                for neighbor in self.graph.get_neighbors(city):
                    edge_weight = self.graph.get_weight(city, neighbor)
                    if distances[city] + edge_weight < distances[neighbor]:
                        distances[neighbor] = distances[city] + edge_weight
                        previous[neighbor] = city
        
        # Check for negative cycles
        for city in self.cities:
            for neighbor in self.graph.get_neighbors(city):
                edge_weight = self.graph.get_weight(city, neighbor)
                if distances[city] + edge_weight < distances[neighbor]:
                    # Negative cycle detected
                    return PathResult(-1, [], "Bellman-Ford", time.time() - start_time, nodes_visited)
        
        # Reconstruct path
        path = []
        if distances[end] != float('inf'):
            current = end
            while current is not None:
                path.append(current)
                current = previous[current]
            path.reverse()
        
        execution_time = time.time() - start_time
        
        return PathResult(
            distance=distances[end] if distances[end] != float('inf') else -1,
            path=path if path else [],
            algorithm="Bellman-Ford",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )
    
    def find_all_shortest_paths(self, start: str) -> Dict[str, PathResult]:
        """
        Find shortest paths from start to all other cities using Bellman-Ford.
        
        Args:
            start: Starting city name
            
        Returns:
            Dictionary mapping destination cities to PathResult objects
        """
        start_time = time.time()
        
        if start not in self.cities:
            raise ValueError(f"City not found: {start}")
        
        # Initialize distances and previous nodes
        distances = {city: float('inf') for city in self.cities}
        previous = {city: None for city in self.cities}
        distances[start] = 0
        
        nodes_visited = 0
        
        # Relax edges V-1 times
        for _ in range(len(self.cities) - 1):
            for city in self.cities:
                nodes_visited += 1
                for neighbor in self.graph.get_neighbors(city):
                    edge_weight = self.graph.get_weight(city, neighbor)
                    if distances[city] + edge_weight < distances[neighbor]:
                        distances[neighbor] = distances[city] + edge_weight
                        previous[neighbor] = city
        
        # Check for negative cycles
        has_negative_cycle = False
        for city in self.cities:
            for neighbor in self.graph.get_neighbors(city):
                edge_weight = self.graph.get_weight(city, neighbor)
                if distances[city] + edge_weight < distances[neighbor]:
                    has_negative_cycle = True
                    break
            if has_negative_cycle:
                break
        
        execution_time = time.time() - start_time
        
        # Reconstruct paths for all destinations
        results = {}
        for end_city in self.cities:
            if has_negative_cycle:
                results[end_city] = PathResult(-1, [], "Bellman-Ford", execution_time, nodes_visited)
                continue
            
            if end_city == start:
                results[end_city] = PathResult(0, [start], "Bellman-Ford", execution_time, nodes_visited)
                continue
            
            path = []
            if distances[end_city] != float('inf'):
                current = end_city
                while current is not None:
                    path.append(current)
                    current = previous[current]
                path.reverse()
            
            results[end_city] = PathResult(
                distance=distances[end_city] if distances[end_city] != float('inf') else -1,
                path=path if path else [],
                algorithm="Bellman-Ford",
                execution_time=execution_time,
                nodes_visited=nodes_visited
            )
        
        return results

class ShortestPathComparator:
    """
    Compare different shortest path algorithms.
    """
    
    def __init__(self, graph):
        """
        Initialize comparator with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.dijkstra = DijkstraAlgorithm(graph)
        self.bellman_ford = BellmanFordAlgorithm(graph)
    
    def compare_algorithms(self, start: str, end: str) -> Dict[str, PathResult]:
        """
        Compare Dijkstra and Bellman-Ford algorithms.
        
        Args:
            start: Starting city name
            end: Destination city name
            
        Returns:
            Dictionary with algorithm results
        """
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
    
    def benchmark_algorithms(self, iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Benchmark algorithms with multiple random city pairs.
        
        Args:
            iterations: Number of test iterations
            
        Returns:
            Dictionary with performance statistics
        """
        import random
        
        cities = self.graph.get_all_cities()
        stats = {
            "Dijkstra": {"total_time": 0, "total_nodes": 0, "successful_runs": 0},
            "Bellman-Ford": {"total_time": 0, "total_nodes": 0, "successful_runs": 0}
        }
        
        for _ in range(iterations):
            start = random.choice(cities)
            end = random.choice([c for c in cities if c != start])
            
            # Test Dijkstra
            try:
                result = self.dijkstra.find_shortest_path(start, end)
                if result.distance >= 0:
                    stats["Dijkstra"]["total_time"] += result.execution_time
                    stats["Dijkstra"]["total_nodes"] += result.nodes_visited
                    stats["Dijkstra"]["successful_runs"] += 1
            except:
                pass
            
            # Test Bellman-Ford
            try:
                result = self.bellman_ford.find_shortest_path(start, end)
                if result.distance >= 0:
                    stats["Bellman-Ford"]["total_time"] += result.execution_time
                    stats["Bellman-Ford"]["total_nodes"] += result.nodes_visited
                    stats["Bellman-Ford"]["successful_runs"] += 1
            except:
                pass
        
        # Calculate averages
        for algorithm in stats:
            if stats[algorithm]["successful_runs"] > 0:
                stats[algorithm]["avg_time"] = stats[algorithm]["total_time"] / stats[algorithm]["successful_runs"]
                stats[algorithm]["avg_nodes"] = stats[algorithm]["total_nodes"] / stats[algorithm]["successful_runs"]
            else:
                stats[algorithm]["avg_time"] = 0
                stats[algorithm]["avg_nodes"] = 0
        
        return stats

if __name__ == "__main__":
    # Test the algorithms
    from data.zimbabwe_cities import ZimbabweGraph
    
    graph = ZimbabweGraph("distance")
    comparator = ShortestPathComparator(graph)
    
    # Test single path
    print("Testing shortest path from Harare to Victoria Falls:")
    results = comparator.compare_algorithms("Harare", "Victoria Falls")
    
    for algo, result in results.items():
        print(f"\n{algo}:")
        print(f"  Distance: {result.distance}")
        print(f"  Path: {' -> '.join(result.path)}")
        print(f"  Execution time: {result.execution_time:.6f}s")
        print(f"  Nodes visited: {result.nodes_visited}")
    
    # Benchmark
    print("\nBenchmarking algorithms...")
    stats = comparator.benchmark_algorithms(50)
    print("\nPerformance Statistics:")
    for algo, stat in stats.items():
        print(f"\n{algo}:")
        print(f"  Average execution time: {stat['avg_time']:.6f}s")
        print(f"  Average nodes visited: {stat['avg_nodes']:.1f}")
        print(f"  Successful runs: {stat['successful_runs']}")
