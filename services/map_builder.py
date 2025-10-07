# services/map_builder.py
import folium
from folium import PolyLine
from models.zimbabwe_graph import ZimbabweGraph

def build_map_for_path(graph: ZimbabweGraph, path: list[str]):
    m = folium.Map(location=[-19.0, 30.0], zoom_start=6, tiles="OpenStreetMap")
    coords = []
    for city in path:
        lat, lon = graph.get_coordinates(city)
        coords.append((lat, lon))
        folium.Marker([lat, lon], popup=city, tooltip=city).add_to(m)
    if len(coords) >= 2:
        PolyLine(coords, color="blue", weight=4, opacity=0.8).add_to(m)
    return m

def build_sparse_graph(base: ZimbabweGraph, max_edge_km: float) -> ZimbabweGraph:
    sparse = ZimbabweGraph(base.weight_type)
    sparse.cities = base.cities
    filtered = {}
    for city in base.get_all_cities():
        filtered[city] = {}
        for dest in base.get_all_cities():
            w = base.get_weight(city, dest)
            filtered[city][dest] = w if (w > 0 and w <= max_edge_km) else 0
        sparse.graph = filtered
    return sparse
