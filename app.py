from flask import Flask, render_template, request
from graph_algorithms.shortest_path.comparator import ShortestPathComparator
from graph_algorithms.tsp_algorithms.comparator import TSPComparator
from graph_algorithms.tsp_algorithms.constrained_tsp import ConstrainedTSP
import folium
from folium import PolyLine
from models.zimbabwe_graph import ZimbabweGraph


app = Flask(__name__)


def build_map_for_path(graph: ZimbabweGraph, path: list[str]):
    # Default center on Zimbabwe roughly
    m = folium.Map(location=[-19.0, 30.0], zoom_start=6, tiles="OpenStreetMap")

    # Add markers for all cities in the path
    coords = []
    for city in path:
        lat, lon = graph.get_coordinates(city)
        coords.append((lat, lon))
        folium.Marker([lat, lon], popup=city, tooltip=city).add_to(m)

    # Draw polyline for the path if there are at least two points
    if len(coords) >= 2:
        PolyLine(coords, color="blue", weight=4, opacity=0.8).add_to(m)

    return m


def build_sparse_graph(base: ZimbabweGraph, max_edge_km: float) -> ZimbabweGraph:
    """Create a temporary sparse view of the graph with only edges <= max_edge_km."""
    sparse = ZimbabweGraph(base.weight_type)
    sparse.cities = base.cities
    filtered = {}
    for city in base.get_all_cities():
        filtered[city] = {}
        for dest in base.get_all_cities():
            w = base.get_weight(city, dest)
            if city == dest:
                filtered[city][dest] = 0
            else:
                filtered[city][dest] = w if (w > 0 and w <= max_edge_km) else 0
    sparse.graph = filtered
    return sparse

@app.route("/", methods=["GET", "POST"])
def index():
    graph = ZimbabweGraph("distance")
    cities = graph.get_all_cities()

    route_info = None
    map_html = None

    if request.method == "POST":
        mode = request.form.get("mode", "sp")  # 'sp' or 'tsp'
        start = request.form.get("start_city")
        end = request.form.get("end_city")
        weight_type = request.form.get("weight_type", "distance")
        # Default to road-like adjacency ON if not provided
        road_adj = request.form.get("road_adj", "on") == "on"

        # Switch weight type if requested
        if weight_type in ("distance", "time", "cost"):
            graph.switch_weight_type(weight_type)

        # Always use adjacency-only graph (no long-distance shortcuts)
        # The adjacency matrix already defines realistic connections
        working_graph = graph  # This now uses the adjacency-only DISTANCE_MATRIX
        
        if mode == "sp":
            comparator = ShortestPathComparator(working_graph)
            results = comparator.compare_algorithms(start, end)

            # Prefer Dijkstra result if valid
            best = results.get("Dijkstra")
            if best is None or best.distance < 0 or len(best.path) < 2:
                best = results.get("Bellman-Ford")

            if best and best.distance >= 0 and len(best.path) >= 1:
                route_info = {
                    "algorithm": best.algorithm,
                    "distance": round(best.distance, 2),
                    "path": best.path,
                    "execution_time": round(best.execution_time, 6),
                    "nodes_visited": best.nodes_visited,
                    "weight_type": weight_type,
                    "mode": mode,
                }
                m = build_map_for_path(working_graph, best.path)
                map_html = m._repr_html_()
            else:
                route_info = {"error": "No valid route found between the selected cities."}
                map_html = None
        else:
            # TSP mode
            tsp_algo = request.form.get("tsp_algo", "nearest")  # held_karp | nearest | two_opt | constrained
            mandatory_raw = request.form.get("mandatory_cities", "").strip()
            max_budget = request.form.get("max_budget")
            max_time = request.form.get("max_time")

            comparator = TSPComparator(working_graph)
            best = None

            try:
                # Fallback to a default start city if not provided
                effective_start = start or (working_graph.get_all_cities()[0] if working_graph.get_all_cities() else None)
                if not effective_start:
                    raise ValueError("No cities available for TSP start.")

                if tsp_algo == "held_karp":
                    best = comparator.held_karp.solve_tsp(effective_start)
                elif tsp_algo == "two_opt":
                    best = comparator.two_opt.solve_tsp(effective_start)
                elif tsp_algo == "constrained":
                    constraints = {}
                    if max_budget:
                        try:
                            constraints["max_budget"] = float(max_budget)
                        except ValueError:
                            pass
                    if max_time:
                        try:
                            constraints["max_time"] = float(max_time)
                        except ValueError:
                            pass
                    mandatory = set()
                    if mandatory_raw:
                        for c in [x.strip() for x in mandatory_raw.split(",") if x.strip()]:
                                # accept only known cities
                            if c in working_graph.get_all_cities():
                                mandatory.add(c)
                    if mandatory:
                        constraints["mandatory_cities"] = mandatory
                    solver = ConstrainedTSP(working_graph, constraints)
                    best = solver.solve_constrained_tsp(effective_start)
                else:
                    best = comparator.nearest_neighbor.solve_tsp(effective_start)
            except Exception as e:
                best = None

            if best and best.total_cost >= 0 and len(best.tour) >= 1:
                route_info = {
                    "algorithm": best.algorithm,
                    "distance": round(best.total_cost, 2),
                    "path": best.tour,
                    "execution_time": round(best.execution_time, 6),
                    "nodes_visited": best.nodes_visited,
                    "weight_type": weight_type,
                    "mode": mode,
                }
                m = build_map_for_path(working_graph, best.tour)
                map_html = m._repr_html_()
            else:
                route_info = {"error": "No valid TSP tour found for the current settings."}
                map_html = None

    return render_template("index.html", cities=cities, route_info=route_info, map_html=map_html)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


