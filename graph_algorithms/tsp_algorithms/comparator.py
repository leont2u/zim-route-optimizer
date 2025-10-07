# tsp/comparator.py
import random
from typing import Optional, Dict
from .types import TSPResult
from .held_karp import HeldKarpAlgorithm
from .nearest_neighbor import NearestNeighborTSP
from .two_opt import TwoOptTSP

class TSPComparator:
    """Compare and benchmark different TSP algorithms."""
    
    def __init__(self, graph):
        self.graph = graph
        self.held_karp = HeldKarpAlgorithm(graph)
        self.nearest_neighbor = NearestNeighborTSP(graph)
        self.two_opt = TwoOptTSP(graph)

    def compare_algorithms(self, start_city: Optional[str] = None) -> Dict[str, TSPResult]:
        results = {}
        if len(self.graph.get_all_cities()) <= 12:
            results["Held-Karp"] = self.held_karp.solve_tsp(start_city)
        results["Nearest Neighbor"] = self.nearest_neighbor.solve_tsp(start_city)
        results["Two-Opt"] = self.two_opt.solve_tsp(start_city)
        return results
