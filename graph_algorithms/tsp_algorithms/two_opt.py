# tsp/two_opt.py
import time
from typing import Optional
from .types import TSPResult
from .base import BaseTSP
from .nearest_neighbor import NearestNeighborTSP
from ..shortest_path.dijkstra import DijkstraAlgorithm
from typing  import List

class TwoOptTSP:
    """
    Two-opt improvement heuristic for TSP.
    Time Complexity: O(n²) per iteration
    Space Complexity: O(n)
    """
    
    def __init__(self, graph):
        """
        Initialize two-opt algorithm with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
        self.dijkstra = DijkstraAlgorithm(graph)
    
    def _distance(self, city_a: str, city_b: str) -> float:
        """Return effective distance between two cities using direct edge or shortest path.
        Returns infinity if no path exists.
        """
        direct_cost = self.graph.get_weight(city_a, city_b)
        if direct_cost > 0:
            return direct_cost
        try:
            path_result = self.dijkstra.find_shortest_path(city_a, city_b)
            return path_result.distance
        except Exception:
            return float('inf')

    def calculate_tour_cost(self, tour: List[str]) -> float:
        """Calculate total cost of a tour."""
        if len(tour) < 2:
            return 0
        
        total_cost = 0
        for i in range(len(tour)):
            current_city = tour[i]
            next_city = tour[(i + 1) % len(tour)]
            
            # Try direct connection first
            direct_cost = self.graph.get_weight(current_city, next_city)
            if direct_cost > 0:
                total_cost += direct_cost
            else:
                # Use shortest path if no direct connection
                try:
                    path_result = self.dijkstra.find_shortest_path(current_city, next_city)
                    total_cost += path_result.distance
                except:
                    # If no path exists, return infinity
                    return float('inf')
        
        return total_cost
    
    def solve_tsp(self, start_city: Optional[str] = None, max_iterations: int = 1000, constraints: dict = None) -> TSPResult:
        """
        Solve TSP using two-opt improvement heuristic.
        
        Args:
            start_city: Starting city (if None, uses first city)
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            TSPResult object with tour, cost, and metadata
        """
        start_time = time.time()
        
        if start_city is None:
            start_city = self.cities[0]
        
        if start_city not in self.cities:
            raise ValueError(f"City not found: {start_city}")
        
        # Start with nearest neighbor solution (pass constraints through)
        nn_solver = NearestNeighborTSP(self.graph)
        initial_result = nn_solver.solve_tsp(start_city, constraints=constraints)

        tour = initial_result.tour.copy()
        best_cost = initial_result.total_cost
        nodes_visited = initial_result.nodes_visited
        constraints = constraints or {}
        max_budget = constraints.get('max_budget', float('inf'))
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            best_improvement = 0
            best_i, best_j = -1, -1
            
            # Try all possible 2-opt swaps
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    # Skip if j is adjacent to i (would create invalid tour)
                    if j == i + 1:
                        continue
                    
                    nodes_visited += 1
                    
                    # Calculate the cost difference of the 2-opt swap
                    # Remove edges (i-1, i) and (j, j+1)
                    # Add edges (i-1, j) and (i, j+1)
                    a, b = tour[i - 1], tour[i]
                    c, d = tour[j], tour[(j + 1) % len(tour)]
                    
                    old_cost = self._distance(a, b) + self._distance(c, d)
                    new_cost = self._distance(a, c) + self._distance(b, d)
                    
                    improvement = old_cost - new_cost
                    
                    # Keep track of the best improvement
                    if improvement > best_improvement:
                        # Quick budget check: estimate new cost and only consider if within budget
                        # Compute candidate new cost by applying swap to a copy (cheap for small tours)
                        # Note: calculate_tour_cost is O(n), but this is safe for moderate n or small iteration count
                        candidate_tour = tour.copy()
                        candidate_tour[ i: j+1 ] = candidate_tour[ i: j+1 ][::-1]
                        candidate_cost = self.calculate_tour_cost(candidate_tour)
                        if candidate_cost <= max_budget:
                            best_improvement = improvement
                            best_i, best_j = i, j
            
            # Apply the best improvement if it exists
            if best_improvement > 1e-12:  # Use small epsilon for numerical stability
                # Perform the 2-opt swap: reverse the segment from i to j
                tour[best_i:best_j+1] = tour[best_i:best_j+1][::-1]
                best_cost = self.calculate_tour_cost(tour)
                improved = True
        
        execution_time = time.time() - start_time
        
        return TSPResult(
            tour=tour,
            total_cost=best_cost,
            algorithm="Two-Opt",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )