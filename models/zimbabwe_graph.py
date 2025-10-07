import numpy as np
import json
from typing import Dict, List, Tuple
from data.cities import ZIMBABWE_CITIES
from data.distance_matrix import DISTANCE_MATRIX
from data.time_matrix import TIME_MATRIX
from data.cost_matrix import COST_MATRIX


class ZimbabweGraph:
    def __init__(self, weight_type: str = "distance"):
        self.cities = ZIMBABWE_CITIES
        self.weight_type = weight_type
        self.graph = self._build_graph()

    def _build_graph(self) -> Dict[str, Dict[str, float]]:
        if self.weight_type == "distance":
            return DISTANCE_MATRIX
        elif self.weight_type == "time":
            return TIME_MATRIX
        elif self.weight_type == "cost":
            return COST_MATRIX
        else:
            raise ValueError("Weight type must be 'distance', 'time', or 'cost'")

    def get_weight(self, from_city: str, to_city: str) -> float:
        return self.graph[from_city][to_city]

    def get_neighbors(self, city: str) -> List[str]:
        return [neighbor for neighbor, weight in self.graph[city].items() if weight > 0]

    def get_all_cities(self) -> List[str]:
        return list(self.cities.keys())

    def get_city_info(self, city: str) -> Dict:
        return self.cities.get(city, {})

    def get_coordinates(self, city: str) -> Tuple[float, float]:
        return self.get_city_info(city).get("coordinates", (0, 0))

    def switch_weight_type(self, weight_type: str):
        self.weight_type = weight_type
        self.graph = self._build_graph()

    def to_adjacency_matrix(self) -> np.ndarray:
        cities = self.get_all_cities()
        matrix = np.zeros((len(cities), len(cities)))
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                matrix[i][j] = self.get_weight(city1, city2)
        return matrix

    def save_to_file(self, filename: str):
        data = {"cities": self.cities, "graph": self.graph, "weight_type": self.weight_type}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        graph = cls(data["weight_type"])
        graph.cities = data["cities"]
        graph.graph = data["graph"]
        return graph
