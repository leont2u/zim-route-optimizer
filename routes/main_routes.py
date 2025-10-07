from flask import Blueprint, render_template, request
from models.zimbabwe_graph import ZimbabweGraph
from services.map_builder import build_map_for_path, build_sparse_graph
from graph_algorithms.shortest_path.comparator import ShortestPathComparator
from graph_algorithms.tsp_algorithms.comparator import TSPComparator
from graph_algorithms.tsp_algorithms.constrained_tsp import ConstrainedTSP

bp = Blueprint("main", __name__)

@bp.route("/", methods=["GET", "POST"])
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
        working_graph = graph  
        
        if mode == "sp":
            # If the user specified a budget or time constraint, use resource-aware SP
            max_budget = request.form.get("max_budget")
            max_time = request.form.get("max_time")
            try:
                if max_budget:
                    max_budget_val = float(max_budget)
                else:
                    max_budget_val = float('inf')
            except ValueError:
                max_budget_val = float('inf')

            try:
                if max_time:
                    max_time_val = float(max_time)
                else:
                    max_time_val = float('inf')
            except ValueError:
                max_time_val = float('inf')

            # If any resource constraint is present, use the resource-constrained solver
            if max_budget_val != float('inf') or max_time_val != float('inf'):
                from graph_algorithms.shortest_path.resource_constrained import resource_constrained_shortest_path

                # For budget, prefer using cost matrix if the graph is in 'cost' mode; otherwise we use distance as proxy
                cost_matrix = None
                if working_graph.weight_type == 'cost':
                    # build a simple dict from adjacency
                    cost_matrix = {}
                    for u in working_graph.get_all_cities():
                        for v in working_graph.get_all_cities():
                            cost_matrix[(u, v)] = working_graph.get_weight(u, v)

                # Use budget if provided; time constraints would require a time-weighted graph (not implemented here)
                res = resource_constrained_shortest_path(working_graph, start, end, max_cost=max_budget_val, cost_matrix=cost_matrix)
                if res.distance >= 0 and res.path:
                    route_info = {
                        "algorithm": res.algorithm,
                        "distance": round(res.distance, 2),
                        "path": res.path,
                        "execution_time": round(res.execution_time, 6),
                        "nodes_visited": res.nodes_visited,
                        "weight_type": weight_type,
                        "mode": mode,
                    }
                    m = build_map_for_path(working_graph, res.path)
                    map_html = m._repr_html_()
                else:
                    route_info = {"error": f"No route satisfies the constraints (budget: {max_budget or 'none'}, time: {max_time or 'none'})"}
                    map_html = None
            else:
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

                # Build constraints dict once and pass down
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
                        if c in working_graph.get_all_cities():
                            mandatory.add(c)
                if mandatory:
                    constraints["mandatory_cities"] = mandatory

                if tsp_algo == "held_karp":
                    best = comparator.held_karp.solve_tsp(effective_start, constraints=constraints)
                elif tsp_algo == "two_opt":
                    best = comparator.two_opt.solve_tsp(effective_start, constraints=constraints)
                elif tsp_algo == "constrained":
                    solver = ConstrainedTSP(working_graph, constraints)
                    best = solver.solve_constrained_tsp(effective_start)
                else:
                    best = comparator.nearest_neighbor.solve_tsp(effective_start, constraints=constraints)
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
                # Provide richer feedback when constraints were used
                constraint_parts = []
                if constraints.get('mandatory_cities'):
                    constraint_parts.append(f"mandatory stops: {', '.join(constraints.get('mandatory_cities'))}")
                if constraints.get('max_budget'):
                    constraint_parts.append(f"max budget: {constraints.get('max_budget')}")
                if constraints.get('max_time'):
                    constraint_parts.append(f"max time: {constraints.get('max_time')}")
                if constraint_parts:
                    route_info = {"error": f"No feasible TSP tour found for the provided constraints ({'; '.join(constraint_parts)})."}
                else:
                    route_info = {"error": "No valid TSP tour found for the current settings."}
                map_html = None
    return render_template(
    "index.html",
    cities=cities,
    route_info=route_info, 
    map_html=map_html      
)
