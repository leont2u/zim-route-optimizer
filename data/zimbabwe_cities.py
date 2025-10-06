"""
Zimbabwe Cities Data Module
Contains city information, coordinates, and distance matrices for major Zimbabwe cities.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional

# Major cities in Zimbabwe with their coordinates and population
ZIMBABWE_CITIES = {
    "Harare": {
        "coordinates": (-17.8292, 31.0522),
        "population": 1485230,
        "province": "Harare",
        "description": "Capital city and largest urban center"
    },
    "Bulawayo": {
        "coordinates": (-20.1569, 28.5892),
        "population": 653337,
        "province": "Bulawayo",
        "description": "Second largest city, industrial hub"
    },
    "Chitungwiza": {
        "coordinates": (-18.0128, 31.0756),
        "population": 365026,
        "province": "Harare",
        "description": "Satellite city of Harare"
    },
    "Mutare": {
        "coordinates": (-18.9707, 32.6722),
        "population": 188243,
        "province": "Manicaland",
        "description": "Eastern border city, gateway to Mozambique"
    },
    "Gweru": {
        "coordinates": (-19.4500, 29.8167),
        "population": 141260,
        "province": "Midlands",
        "description": "Central Zimbabwe city"
        
    },
    "Kwekwe": {
        "coordinates": (-18.9167, 29.8167),
        "population": 103408,
        "province": "Midlands",
        "description": "Industrial city in Midlands"
    },
    "Kadoma": {
        "coordinates": (-18.3333, 29.9167),
        "population": 79049,
        "province": "Mashonaland West",
        "description": "Mining town"
    },
    "Masvingo": {
        "coordinates": (-20.0833, 30.8333),
        "population": 87277,
        "province": "Masvingo",
        "description": "Southern Zimbabwe city"
    },
    "Chinhoyi": {
        "coordinates": (-17.3667, 30.2000),
        "population": 77007,
        "province": "Mashonaland West",
        "description": "Agricultural center"
    },
    "Marondera": {
        "coordinates": (-18.1833, 31.5500),
        "population": 61000,
        "province": "Mashonaland East",
        "description": "Agricultural town"
    },
    "Bindura": {
        "coordinates": (-17.3000, 31.3333),
        "population": 46000,
        "province": "Mashonaland Central",
        "description": "Mining town"
    },
    "Victoria Falls": {
        "coordinates": (-17.9333, 25.8333),
        "population": 33000,
        "province": "Matabeleland North",
        "description": "Tourist destination"
    }
}

# Distance matrix between cities (in kilometers)
# Based on actual road distances in Zimbabwe - ADJACENCY ONLY
# Only includes direct road connections between adjacent cities
DISTANCE_MATRIX = {
    "Harare": {
        "Harare": 0, "Bulawayo": 0, "Chitungwiza": 30, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 150, "Masvingo": 0,
        "Chinhoyi": 120, "Marondera": 80, "Bindura": 80, "Victoria Falls": 0
    },
    "Bulawayo": {
        "Harare": 0, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 164, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 120,
        "Chinhoyi": 0, "Marondera": 0, "Bindura": 0, "Victoria Falls": 450
    },
    "Chitungwiza": {
        "Harare": 30, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 0, "Marondera": 110, "Bindura": 110, "Victoria Falls": 0
    },
    "Mutare": {
        "Harare": 0, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 0, "Marondera": 185, "Bindura": 185, "Victoria Falls": 0
    },
    "Gweru": {
        "Harare": 0, "Bulawayo": 164, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 76, "Kadoma": 126, "Masvingo": 0,
        "Chinhoyi": 0, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    },
    "Kwekwe": {
        "Harare": 0, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 76, "Kwekwe": 0, "Kadoma": 50, "Masvingo": 0,
        "Chinhoyi": 80, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    },
    "Kadoma": {
        "Harare": 150, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 126, "Kwekwe": 50, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 30, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    },
    "Masvingo": {
        "Harare": 0, "Bulawayo": 120, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 0, "Marondera": 0, "Bindura": 0, "Victoria Falls": 570
    },
    "Chinhoyi": {
        "Harare": 120, "Bulawayo": 0, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 80, "Kadoma": 30, "Masvingo": 0,
        "Chinhoyi": 0, "Marondera": 40, "Bindura": 40, "Victoria Falls": 0
    },
    "Marondera": {
        "Harare": 80, "Bulawayo": 0, "Chitungwiza": 110, "Mutare": 185,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 40, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    },
    "Bindura": {
        "Harare": 80, "Bulawayo": 0, "Chitungwiza": 110, "Mutare": 185,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 0,
        "Chinhoyi": 40, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    },
    "Victoria Falls": {
        "Harare": 0, "Bulawayo": 450, "Chitungwiza": 0, "Mutare": 0,
        "Gweru": 0, "Kwekwe": 0, "Kadoma": 0, "Masvingo": 570,
        "Chinhoyi": 0, "Marondera": 0, "Bindura": 0, "Victoria Falls": 0
    }
}

# Time matrix (estimated travel time in hours)
TIME_MATRIX = {
    city: {
        dest: round(distance / 60, 1)  # Assuming average speed of 60 km/h
        for dest, distance in distances.items()
    }
    for city, distances in DISTANCE_MATRIX.items()
}

# Cost matrix (estimated fuel cost in USD)
COST_MATRIX = {
    city: {
        dest: round(distance * 0.15, 2)  # Assuming $0.15 per km fuel cost
        for dest, distance in distances.items()
    }
    for city, distances in DISTANCE_MATRIX.items()
}

class ZimbabweGraph:
    """
    Graph representation of Zimbabwe cities with weighted edges.
    Supports distance, time, and cost weights.
    """
    
    def __init__(self, weight_type: str = "distance"):
        """
        Initialize the Zimbabwe cities graph.
        
        Args:
            weight_type: Type of edge weights ('distance', 'time', 'cost')
        """
        self.cities = ZIMBABWE_CITIES
        self.weight_type = weight_type
        self.graph = self._build_graph()
        
    def _build_graph(self) -> Dict[str, Dict[str, float]]:
        """Build the graph with specified weight type."""
        if self.weight_type == "distance":
            return DISTANCE_MATRIX
        elif self.weight_type == "time":
            return TIME_MATRIX
        elif self.weight_type == "cost":
            return COST_MATRIX
        else:
            raise ValueError("Weight type must be 'distance', 'time', or 'cost'")
    
    def get_weight(self, from_city: str, to_city: str) -> float:
        """Get weight between two cities."""
        return self.graph[from_city][to_city]
    
    def get_neighbors(self, city: str) -> List[str]:
        """Get all neighboring cities."""
        return [neighbor for neighbor, weight in self.graph[city].items() 
                if weight > 0]
    
    def get_all_cities(self) -> List[str]:
        """Get list of all cities."""
        return list(self.cities.keys())
    
    def get_city_info(self, city: str) -> Dict:
        """Get detailed information about a city."""
        return self.cities.get(city, {})
    
    def get_coordinates(self, city: str) -> Tuple[float, float]:
        """Get coordinates of a city."""
        info = self.get_city_info(city)
        return info.get("coordinates", (0, 0))
    
    def switch_weight_type(self, weight_type: str):
        """Switch the weight type of the graph."""
        self.weight_type = weight_type
        self.graph = self._build_graph()
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert graph to adjacency matrix."""
        cities = self.get_all_cities()
        n = len(cities)
        matrix = np.zeros((n, n))
        
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                matrix[i][j] = self.get_weight(city1, city2)
        
        return matrix
    
    def save_to_file(self, filename: str):
        """Save graph data to JSON file."""
        data = {
            "cities": self.cities,
            "graph": self.graph,
            "weight_type": self.weight_type
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str):
        """Load graph data from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        graph = cls(data["weight_type"])
        graph.cities = data["cities"]
        graph.graph = data["graph"]
        return graph

def create_sample_graphs():
    """Create sample graphs with different weight types."""
    distance_graph = ZimbabweGraph("distance")
    time_graph = ZimbabweGraph("time")
    cost_graph = ZimbabweGraph("cost")
    
    return distance_graph, time_graph, cost_graph

if __name__ == "__main__":
    # Test the graph creation
    graph = ZimbabweGraph("distance")
    print("Cities:", graph.get_all_cities())
    print("Distance from Harare to Bulawayo:", graph.get_weight("Harare", "Bulawayo"))
    print("Neighbors of Harare:", graph.get_neighbors("Harare"))
    
    # Test adjacency matrix
    adj_matrix = graph.to_adjacency_matrix()
    print("Adjacency matrix shape:", adj_matrix.shape)
