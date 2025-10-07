# Zimbabwe Route Optimization Application

A comprehensive Python application for finding **optimal routes between major cities in Zimbabwe** using graph algorithms, dynamic programming, and interactive map visualization.

## Features

- Graph representation of Zimbabwe cities with weighted edges: **distance, time, cost**
- Shortest path algorithms: **Dijkstra's** and **Bellman-Ford**
- Traveling Salesman Problem (TSP) solutions:

  - Exact solution using **Held-Karp (DP)**
  - Heuristic solutions: **Nearest Neighbor** and **Two-Opt**
  - **Constraint handling**: mandatory stops, max budget, max time

- Interactive **map visualization** with **Folium**
- **Web-based interface** using Flask
- Performance analysis and scalability evaluation

## Installation

Clone the repository and install dependencies:

```bash
git clone <https://github.com/leont2u/zim-route-optimizer.git>
cd zim-route-optimizer
pip install -r requirements.txt
```

## Usage

Run the interactive web application:

```bash
python app.py
```

Open your browser at [http://localhost:5000](http://localhost:5000) and:

1. Select **Start** and **End cities**
2. Choose **Optimization metric** (distance/time/cost)
3. Select **Mode**: Shortest Path (SP) or TSP
4. (Optional) For TSP: provide **mandatory stops**, **max budget**, and **max time**

The map displays **city markers** and **polylines for the computed route**.

### Modes

- **Shortest Path (SP)**
  Finds the shortest route between two cities using **Dijkstra** (primary) and **Bellman-Ford** (fallback).

- **Traveling Salesman Problem (TSP)**
  Options:

  - **Nearest Neighbor** (greedy heuristic)
  - **Two-Opt** (heuristic improvement)
  - **Held-Karp** (DP exact solution; recommended for small graphs)
  - **Constrained TSP** (obeys mandatory stops, max budget, max time)

### Constraints (TSP)

- **Mandatory Stops**: Comma-separated city names
- **Max Budget**: Upper bound on total cost
- **Max Time**: Upper bound on travel time

> Ensure city names match the dropdown options (e.g., `Harare`, `Bulawayo`, `Mutare`).

## Project Structure

```
zim-route-optimizer/
│
├─ app.py                  # Flask app entry point
├─ main.py                 # Alternative entry point / CLI
├─ requirements.txt        # Python dependencies
├─ templates/              # HTML templates (Jinja2)
│
├─ data/
│   └─ cities.py           # Zimbabwe cities and distance/time/cost matrices
│
├─ models/
│   └─ zimbabwe_graph.py   # Graph representation of cities
│
├─ graph_algorithms/
│   ├─ shortest_path/
│   │   └─ comparator.py   # Dijkstra & Bellman-Ford
│   └─ tsp_algorithms/
│       ├─ comparator.py   # TSP algorithm selectors
│       └─ constrained_tsp.py
│
├─ services/
│   └─ map_builder.py      # Folium map construction
│
├─ static/
│   └─ styles.css           # Custom CSS
│
├─ tests/                  # Unit tests and benchmarks
└─ docs/                   # Reports, diagrams, and documentation
```

## Algorithms Implemented

1. **Dijkstra's Algorithm** – Fast single-source shortest path
2. **Bellman-Ford Algorithm** – Handles negative edges and fallback scenarios
3. **Held-Karp Algorithm** – Exact TSP solution using dynamic programming
4. **Nearest Neighbor & Two-Opt** – TSP heuristic solutions
5. **Constrained TSP (DP-based)** – Supports budget, time, and mandatory stops

## Performance Analysis

- **Time Complexity**: Analyzed for each algorithm
- **Space Complexity**: Evaluated for adjacency matrices and DP states
- **Scalability**: Tested with various city graph sizes
- **Comparisons**: Heuristic vs. exact algorithms for TSP

## Visualization

- Interactive maps created with **Folium**
- Routes displayed with **markers** and **polylines**
- Route statistics shown in the web UI: distance, cost, nodes visited, execution time

## Contribution

1. Clone the repo
2. Create a feature branch
3. Submit a pull request

## License
