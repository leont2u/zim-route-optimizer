# algorithms/models.py
from dataclasses import dataclass
from typing import List

@dataclass
class PathResult:
    """Result of a shortest path algorithm."""
    distance: float
    path: List[str]
    algorithm: str
    execution_time: float
    nodes_visited: int
