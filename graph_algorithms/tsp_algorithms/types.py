# tsp/types.py
from dataclasses import dataclass
from typing import List

@dataclass
class TSPResult:
    """Result of a TSP algorithm."""
    tour: List[str]
    total_cost: float
    algorithm: str
    execution_time: float
    nodes_visited: int
