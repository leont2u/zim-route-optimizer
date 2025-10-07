# tsp/constrained_tsp.py
import time
from typing import Optional, Dict, Set, List
from .types import TSPResult
from .base import BaseTSP
from ..shortest_path.dijkstra import DijkstraAlgorithm
from .held_karp import HeldKarpAlgorithm


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
            tour=full_tour,
            total_cost=total_cost,
            algorithm=algorithm_name,
            execution_time=execution_time,
            nodes_visited=nodes_visited
        )