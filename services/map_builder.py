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
    # draw each segment so we can style the closing segment separately
    if len(coords) >= 2:
        for i in range(len(coords) - 1):
            segment = [coords[i], coords[i+1]]
            # if this is the closing segment (returns to start), color it red
            color = "red" if i == len(coords) - 2 and coords[i+1] == coords[0] else "blue"
            PolyLine(segment, color=color, weight=4, opacity=0.8).add_to(m)
    # add a small legend overlay
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 10px; width: 160px; height: 70px; 
                background-color: white; z-index:9999; padding: 8px; border:2px solid grey; border-radius:6px;">
      <b>Route legend</b><br>
      <i style="background: blue; width: 12px; height: 6px; display: inline-block; margin-right:6px;"></i> Route segment<br>
      <i style="background: red; width: 12px; height: 6px; display: inline-block; margin-right:6px;"></i> Return Route
    </div>
    '''
    from folium import Element
    legend = Element(legend_html)
    m.get_root().html.add_child(legend)

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
