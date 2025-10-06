"""
Traveling Salesman Problem (TSP) Implementation
Contains Held-Karp algorithm for exact TSP solution and heuristic approaches.
"""

import itertools
import time
import random
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from .shortest_path import DijkstraAlgorithm

@dataclass
class TSPResult:
    """Result of a TSP algorithm."""
    tour: List[str]
    total_cost: float
    algorithm: str
    execution_time: float
    nodes_visited: int

class HeldKarpAlgorithm:
    """
    Implementation of Held-Karp algorithm for exact TSP solution.
    Time Complexity: O(n²2ⁿ)
    Space Complexity: O(n2ⁿ)
    """
    
    def __init__(self, graph):
        """
        Initialize Held-Karp algorithm with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
        self.n = len(self.cities)
        self.city_to_index = {city: i for i, city in enumerate(self.cities)}
        self.index_to_city = {i: city for i, city in enumerate(self.cities)}
        
        # Precompute distance matrix
        self.dist_matrix = np.zeros((self.n, self.n))
        for i, city1 in enumerate(self.cities):
            for j, city2 in enumerate(self.cities):
                self.dist_matrix[i][j] = self.graph.get_weight(city1, city2)
    
    def solve_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        """
        Solve TSP using Held-Karp algorithm.
        
        Args:
            start_city: Starting city (if None, uses first city)
            
        Returns:
            TSPResult object with tour, cost, and metadata
        """
        start_time = time.time()
        
        if start_city is None:
            start_city = self.cities[0]
        
        if start_city not in self.cities:
            raise ValueError(f"City not found: {start_city}")
        
        start_idx = self.city_to_index[start_city]
        
        # DP table: dp[mask][i] = minimum cost to visit all cities in mask ending at city i
        dp = {}
        parent = {}
        
        # Initialize base case: visiting only the starting city
        dp[(1 << start_idx, start_idx)] = 0
        
        nodes_visited = 0
        
        # Fill DP table
        for subset_size in range(2, self.n + 1):
            for subset in itertools.combinations(range(self.n), subset_size):
                if start_idx not in subset:
                    continue
                
                subset_mask = sum(1 << i for i in subset)
                
                # Try all possible last cities in the subset
                for last_city in subset:
                    if last_city == start_idx:
                        continue
                    
                    nodes_visited += 1
                    
                    # Try all possible second-to-last cities
                    prev_mask = subset_mask ^ (1 << last_city)
                    
                    for prev_city in subset:
                        if prev_city == last_city or prev_city == start_idx:
                            continue
                        
                        if (prev_mask, prev_city) in dp:
                            cost = dp[(prev_mask, prev_city)] + self.dist_matrix[prev_city][last_city]
                            
                            if (subset_mask, last_city) not in dp or cost < dp[(subset_mask, last_city)]:
                                dp[(subset_mask, last_city)] = cost
                                parent[(subset_mask, last_city)] = prev_city
        
        # Find minimum cost tour
        full_mask = (1 << self.n) - 1
        min_cost = float('inf')
        last_city = -1
        
        for i in range(self.n):
            if i != start_idx and (full_mask, i) in dp:
                cost = dp[(full_mask, i)] + self.dist_matrix[i][start_idx]
                if cost < min_cost:
                    min_cost = cost
                    last_city = i
        
        # Reconstruct tour
        tour = []
        if last_city != -1:
            current_mask = full_mask
            current_city = last_city
            
            while current_city != start_idx:
                tour.append(self.index_to_city[current_city])
                prev_city = parent[(current_mask, current_city)]
                current_mask ^= (1 << current_city)
                current_city = prev_city
            
            tour.append(self.index_to_city[start_idx])
            tour.reverse()
        
        execution_time = time.time() - start_time
        
        return TSPResult(
            tour=tour,
            total_cost=min_cost if min_cost != float('inf') else -1,
            algorithm="Held-Karp",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )

class NearestNeighborTSP:
    """
    Greedy nearest neighbor heuristic for TSP.
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    
    def __init__(self, graph):
        """
        Initialize nearest neighbor algorithm with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
        self.dijkstra = DijkstraAlgorithm(graph)
    
    def solve_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        """
        Solve TSP using nearest neighbor heuristic.
        
        Args:
            start_city: Starting city (if None, uses first city)
            
        Returns:
            TSPResult object with tour, cost, and metadata
        """
        start_time = time.time()
        
        if start_city is None:
            start_city = self.cities[0]
        
        if start_city not in self.cities:
            raise ValueError(f"City not found: {start_city}")
        
        tour = [start_city]
        unvisited = set(self.cities) - {start_city}
        current_city = start_city
        total_cost = 0
        nodes_visited = 0
        
        while unvisited:
            nodes_visited += 1
            
            # Find nearest unvisited city using shortest path if needed
            best_city = None
            best_cost = float('inf')
            
            for city in unvisited:
                # Try direct connection first
                direct_cost = self.graph.get_weight(current_city, city)
                if direct_cost > 0:
                    cost = direct_cost
                else:
                    # Use shortest path if no direct connection
                    try:
                        path_result = self.dijkstra.find_shortest_path(current_city, city)
                        cost = path_result.distance
                    except:
                        # If no path exists, skip this city
                        continue
                
                if cost < best_cost:
                    best_cost = cost
                    best_city = city
            
            if best_city is None:
                # If no reachable cities, break the tour
                break
            
            # Add to tour
            tour.append(best_city)
            total_cost += best_cost
            
            # Update state
            unvisited.remove(best_city)
            current_city = best_city
        
        # Return to start
        return_cost = self.graph.get_weight(current_city, start_city)
        if return_cost <= 0:
            # Use shortest path to return to start
            try:
                return_result = self.dijkstra.find_shortest_path(current_city, start_city)
                total_cost += return_result.distance
            except:
                # If no path back to start, we can't complete the tour
                pass
        else:
            total_cost += return_cost
        
        execution_time = time.time() - start_time
        
        return TSPResult(
            tour=tour,
            total_cost=total_cost,
            algorithm="Nearest Neighbor",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )

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
    
    def solve_tsp(self, start_city: Optional[str] = None, max_iterations: int = 1000) -> TSPResult:
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
        
        # Start with nearest neighbor solution
        nn_solver = NearestNeighborTSP(self.graph)
        initial_result = nn_solver.solve_tsp(start_city)
        
        tour = initial_result.tour.copy()
        best_cost = initial_result.total_cost
        nodes_visited = initial_result.nodes_visited
        
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, len(tour) - 1):
                for j in range(i + 1, len(tour)):
                    nodes_visited += 1
                    
                    # Calculate cost of two-opt swap
                    old_cost = (self.graph.get_weight(tour[i-1], tour[i]) + 
                              self.graph.get_weight(tour[j], tour[(j+1) % len(tour)]))
                    
                    new_cost = (self.graph.get_weight(tour[i-1], tour[j]) + 
                              self.graph.get_weight(tour[i], tour[(j+1) % len(tour)]))
                    
                    if new_cost < old_cost:
                        # Perform two-opt swap
                        tour[i:j+1] = tour[i:j+1][::-1]
                        best_cost = self.calculate_tour_cost(tour)
                        improved = True
                        break
                
                if improved:
                    break
        
        execution_time = time.time() - start_time
        
        return TSPResult(
            tour=tour,
            total_cost=best_cost,
            algorithm="Two-Opt",
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )

class ConstrainedTSP:
    """
    TSP with constraints like time windows and budgets.
    Uses dynamic programming with constraint states.
    """
    
    def __init__(self, graph, constraints: Optional[Dict] = None):
        """
        Initialize constrained TSP solver.
        
        Args:
            graph: ZimbabweGraph instance
            constraints: Dictionary with constraint parameters
        """
        self.graph = graph
        self.cities = graph.get_all_cities()
        self.constraints = constraints or {}
        
        # Default constraints
        self.max_budget = self.constraints.get('max_budget', float('inf'))
        self.max_time = self.constraints.get('max_time', float('inf'))
        self.time_windows = self.constraints.get('time_windows', {})
        self.mandatory_cities = self.constraints.get('mandatory_cities', set())
        
        # Add realistic time windows for demonstration
        if not self.time_windows:
            self.time_windows = {
                'Harare': (0, 24),      # Always available
                'Bulawayo': (0, 24),    # Always available
                'Mutare': (0, 24),      # Always available
                'Gweru': (0, 24),       # Always available
                'Victoria Falls': (0, 24)  # Always available
            }
    
    def solve_constrained_tsp(self, start_city: Optional[str] = None) -> TSPResult:
        """
        Solve constrained TSP using constraint-aware greedy approach.
        Demonstrates how constraints affect route optimization.
        
        Args:
            start_city: Starting city
            
        Returns:
            TSPResult object with tour, cost, and metadata
        """
        start_time = time.time()
        
        if start_city is None:
            start_city = self.cities[0]
        
        if start_city not in self.cities:
            raise ValueError(f"City not found: {start_city}")
        
        # Start with the starting city
        tour = [start_city]
        unvisited = set(self.cities) - {start_city}
        current_city = start_city
        total_cost = 0
        total_time = 0
        nodes_visited = 0
        
        # Track constraint violations for demonstration
        constraint_violations = []
        mandatory_remaining = self.mandatory_cities.copy()
        
        # First, visit mandatory cities if specified
        if mandatory_remaining:
            for mandatory_city in list(mandatory_remaining):
                if mandatory_city in unvisited:
                    cost_to_city = self.graph.get_weight(current_city, mandatory_city)
                    
                    if cost_to_city > 0:  # Only if directly connected
                        time_to_city = cost_to_city / 60  # 60 km/h average speed
                        
                        # Check if we can visit this mandatory city within constraints
                        if (total_cost + cost_to_city <= self.max_budget and 
                            total_time + time_to_city <= self.max_time):
                            
                            # Check time window if applicable
                            if mandatory_city in self.time_windows:
                                arrival_time = total_time + time_to_city
                                window_start, window_end = self.time_windows[mandatory_city]
                                if not (window_start <= arrival_time <= window_end):
                                    constraint_violations.append(f"{mandatory_city}: Time window missed")
                                    continue
                            
                            # Visit the mandatory city
                            tour.append(mandatory_city)
                            total_cost += cost_to_city
                            total_time += time_to_city
                            unvisited.remove(mandatory_city)
                            mandatory_remaining.remove(mandatory_city)
                            current_city = mandatory_city
                            nodes_visited += 1
        
        # Now visit remaining cities using constraint-aware greedy approach
        while unvisited:
            nodes_visited += 1
            
            best_city = None
            best_score = float('inf')
            
            # Find the best next city considering constraints
            for city in unvisited:
                cost_to_city = self.graph.get_weight(current_city, city)
                
                # Skip cities that are not directly connected
                if cost_to_city <= 0:
                    continue
                
                time_to_city = cost_to_city / 60  # 60 km/h average speed
                new_time = total_time + time_to_city
                new_cost = total_cost + cost_to_city
                
                # Check constraints
                constraints_met = True
                violation_reason = ""
                
                # Budget constraint
                if new_cost > self.max_budget:
                    constraints_met = False
                    violation_reason = f"Budget exceeded: {new_cost:.1f} > {self.max_budget}"
                
                # Time constraint
                elif new_time > self.max_time:
                    constraints_met = False
                    violation_reason = f"Time exceeded: {new_time:.1f}h > {self.max_time}h"
                
                # Time window constraint
                elif city in self.time_windows:
                    window_start, window_end = self.time_windows[city]
                    if not (window_start <= new_time <= window_end):
                        constraints_met = False
                        violation_reason = f"Time window missed: {new_time:.1f}h not in [{window_start}, {window_end}]"
                
                if not constraints_met:
                    constraint_violations.append(f"{city}: {violation_reason}")
                    continue
                
                # Calculate score (lower is better)
                # Prioritize mandatory cities and shorter distances
                score = cost_to_city
                if city in mandatory_remaining:
                    score *= 0.3  # Strongly prioritize mandatory cities
                
                if score < best_score:
                    best_score = score
                    best_city = city
            
            # If no valid city found, break
            if best_city is None:
                break
            
            # Visit the best city
            cost_to_city = self.graph.get_weight(current_city, best_city)
            time_to_city = cost_to_city / 60
            
            tour.append(best_city)
            total_cost += cost_to_city
            total_time += time_to_city
            
            unvisited.remove(best_city)
            if best_city in mandatory_remaining:
                mandatory_remaining.remove(best_city)
            
            current_city = best_city
        
        # Try to return to start if possible
        if current_city != start_city:
            cost_home = self.graph.get_weight(current_city, start_city)
            if cost_home > 0:
                time_home = cost_home / 60
                
                # Check if we can return within constraints
                if (total_cost + cost_home <= self.max_budget and 
                    total_time + time_home <= self.max_time):
                    total_cost += cost_home
                    total_time += time_home
                    # Note: We don't add start_city to tour again to avoid duplication
        
        execution_time = time.time() - start_time
        
        # Add constraint information to result
        algorithm_name = "Constrained TSP"
        if constraint_violations:
            algorithm_name += f" (Constraints: {len(constraint_violations)} violations)"
        
        return TSPResult(
            tour=tour,
            total_cost=total_cost,
            algorithm=algorithm_name,
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )

class TSPComparator:
    """
    Compare different TSP algorithms.
    """
    
    def __init__(self, graph):
        """
        Initialize TSP comparator with a graph.
        
        Args:
            graph: ZimbabweGraph instance
        """
        self.graph = graph
        self.held_karp = HeldKarpAlgorithm(graph)
        self.nearest_neighbor = NearestNeighborTSP(graph)
        self.two_opt = TwoOptTSP(graph)
    
    def compare_algorithms(self, start_city: Optional[str] = None) -> Dict[str, TSPResult]:
        """
        Compare different TSP algorithms.
        
        Args:
            start_city: Starting city
            
        Returns:
            Dictionary with algorithm results
        """
        results = {}
        
        # Test Held-Karp (only for small instances)
        if len(self.graph.get_all_cities()) <= 12:
            try:
                results["Held-Karp"] = self.held_karp.solve_tsp(start_city)
            except Exception as e:
                print(f"Held-Karp error: {e}")
                results["Held-Karp"] = TSPResult([], -1, "Held-Karp", 0, 0)
        else:
            print("Held-Karp skipped: too many cities for exact solution")
        
        # Test Nearest Neighbor
        try:
            results["Nearest Neighbor"] = self.nearest_neighbor.solve_tsp(start_city)
        except Exception as e:
            print(f"Nearest Neighbor error: {e}")
            results["Nearest Neighbor"] = TSPResult([], -1, "Nearest Neighbor", 0, 0)
        
        # Test Two-Opt
        try:
            results["Two-Opt"] = self.two_opt.solve_tsp(start_city)
        except Exception as e:
            print(f"Two-Opt error: {e}")
            results["Two-Opt"] = TSPResult([], -1, "Two-Opt", 0, 0)
        
        return results
    
    def benchmark_algorithms(self, iterations: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Benchmark TSP algorithms.
        
        Args:
            iterations: Number of test iterations
            
        Returns:
            Dictionary with performance statistics
        """
        cities = self.graph.get_all_cities()
        stats = {
            "Nearest Neighbor": {"total_time": 0, "total_cost": 0, "successful_runs": 0},
            "Two-Opt": {"total_time": 0, "total_cost": 0, "successful_runs": 0}
        }
        
        for _ in range(iterations):
            start_city = random.choice(cities)
            
            # Test Nearest Neighbor
            try:
                result = self.nearest_neighbor.solve_tsp(start_city)
                if result.total_cost > 0:
                    stats["Nearest Neighbor"]["total_time"] += result.execution_time
                    stats["Nearest Neighbor"]["total_cost"] += result.total_cost
                    stats["Nearest Neighbor"]["successful_runs"] += 1
            except:
                pass
            
            # Test Two-Opt
            try:
                result = self.two_opt.solve_tsp(start_city)
                if result.total_cost > 0:
                    stats["Two-Opt"]["total_time"] += result.execution_time
                    stats["Two-Opt"]["total_cost"] += result.total_cost
                    stats["Two-Opt"]["successful_runs"] += 1
            except:
                pass
        
        # Calculate averages
        for algorithm in stats:
            if stats[algorithm]["successful_runs"] > 0:
                stats[algorithm]["avg_time"] = stats[algorithm]["total_time"] / stats[algorithm]["successful_runs"]
                stats[algorithm]["avg_cost"] = stats[algorithm]["total_cost"] / stats[algorithm]["successful_runs"]
            else:
                stats[algorithm]["avg_time"] = 0
                stats[algorithm]["avg_cost"] = 0
        
        return stats

if __name__ == "__main__":
    # Test the TSP algorithms
    from data.zimbabwe_cities import ZimbabweGraph
    
    graph = ZimbabweGraph("distance")
    comparator = TSPComparator(graph)
    
    # Test TSP algorithms
    print("Testing TSP algorithms:")
    results = comparator.compare_algorithms("Harare")
    
    for algo, result in results.items():
        print(f"\n{algo}:")
        print(f"  Tour: {' -> '.join(result.tour)}")
        print(f"  Total cost: {result.total_cost:.2f}")
        print(f"  Execution time: {result.execution_time:.6f}s")
        print(f"  Nodes visited: {result.nodes_visited}")
    
    # Benchmark
    print("\nBenchmarking TSP algorithms...")
    stats = comparator.benchmark_algorithms(20)
    print("\nPerformance Statistics:")
    for algo, stat in stats.items():
        print(f"\n{algo}:")
        print(f"  Average execution time: {stat['avg_time']:.6f}s")
        print(f"  Average cost: {stat['avg_cost']:.2f}")
        print(f"  Successful runs: {stat['successful_runs']}")
