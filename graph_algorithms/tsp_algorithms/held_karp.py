# tsp/held_karp.py
import itertools, time, numpy as np
from typing import Optional
from .types import TSPResult
from .base import BaseTSP

class HeldKarpAlgorithm(BaseTSP):
    """Exact Held-Karp algorithm for TSP."""
    
    def __init__(self, graph):
        super().__init__(graph)
        self.n = len(self.cities)
        self.city_to_index = {city: i for i, city in enumerate(self.cities)}
        self.index_to_city = {i: city for i, city in enumerate(self.cities)}
        self.dist_matrix = np.zeros((self.n, self.n))
        for i, c1 in enumerate(self.cities):
            for j, c2 in enumerate(self.cities):
                self.dist_matrix[i][j] = self.graph.get_weight(c1, c2)

    def solve_tsp(self, start_city: Optional[str] = None, constraints: dict = None) -> TSPResult:
        start_time = time.time()
        start_city = start_city or self.cities[0]
        constraints = constraints or {}
        max_budget = constraints.get('max_budget', float('inf'))
        mandatory = set(constraints.get('mandatory_cities', []))
        start_idx = self.city_to_index[start_city]

        dp, parent = {(1 << start_idx, start_idx): 0}, {}
        nodes_visited = 0

        for subset_size in range(2, self.n + 1):
            for subset in itertools.combinations(range(self.n), subset_size):
                if start_idx not in subset:
                    continue
                subset_mask = sum(1 << i for i in subset)
                for last in subset:
                    if last == start_idx: continue
                    nodes_visited += 1
                    prev_mask = subset_mask ^ (1 << last)
                    for prev in subset:
                        # allow the start city as a valid previous node; only skip if prev is the same as last
                        if prev == last:
                            continue
                        if (prev_mask, prev) in dp:
                            cost = dp[(prev_mask, prev)] + self.dist_matrix[prev][last]
                            if (subset_mask, last) not in dp or cost < dp[(subset_mask, last)]:
                                dp[(subset_mask, last)] = cost
                                parent[(subset_mask, last)] = prev

        full_mask = (1 << self.n) - 1
        min_cost, last_city = float('inf'), -1
        for i in range(self.n):
            if i == start_idx or (full_mask, i) not in dp: continue
            cost = dp[(full_mask, i)] + self.dist_matrix[i][start_idx]
            if cost < min_cost:
                min_cost, last_city = cost, i

        tour = []
        if last_city != -1:
            current_mask, current_city = full_mask, last_city
            while current_city != start_idx:
                tour.append(self.index_to_city[current_city])
                prev = parent[(current_mask, current_city)]
                current_mask ^= (1 << current_city)
                current_city = prev
            tour.append(start_city)
            tour.reverse()

        # ensure tour is closed (starts and ends at start_city)
        if tour:
            if tour[0] != start_city:
                # rotate if start_city is in tour
                if start_city in tour:
                    idx = tour.index(start_city)
                    tour = tour[idx:] + tour[:idx]
                else:
                    tour.insert(0, start_city)
            if tour[-1] != start_city:
                tour.append(start_city)
        total_cost = min_cost if min_cost != float('inf') else float('inf')

        # Validate mandatory nodes
        if mandatory and not mandatory.issubset(set(tour)):
            return TSPResult(tour=[], total_cost=-1, algorithm="Held-Karp", execution_time=time.time() - start_time, nodes_visited=nodes_visited)

        # Validate budget
        if total_cost > max_budget:
            return TSPResult(tour=[], total_cost=-1, algorithm="Held-Karp", execution_time=time.time() - start_time, nodes_visited=nodes_visited)

        return TSPResult(
            tour=tour,
            total_cost=total_cost if total_cost != float('inf') else -1,
            algorithm="Held-Karp",
            execution_time=time.time() - start_time,
            nodes_visited=nodes_visited
        )
