# tsp/base.py
from typing import Optional
from .types import TSPResult

class BaseTSP:
    """Base class for all TSP algorithms."""
    
    def __init__(self, graph):
        self.graph = graph
        self.cities = graph.get_all_cities()

    def solve_tsp(self, start_city: Optional[str] = None, constraints: dict = None) -> TSPResult:
        """Override in subclasses."""
        raise NotImplementedError("Subclasses must implement solve_tsp()")
