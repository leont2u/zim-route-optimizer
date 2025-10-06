# Zimbabwe Route Optimization Application

A comprehensive application for finding optimal routes between major cities in Zimbabwe using graph algorithms and dynamic programming.

## Features

- Graph representation of Zimbabwe cities with weighted edges (distance/time/cost)
- Shortest path algorithms (Dijkstra's and Bellman-Ford)
- Traveling Salesman Problem (TSP) solution using Held-Karp algorithm
- Constraint handling (time windows, budgets, multiple stops)
- Interactive visualization with NetworkX and Matplotlib
- Web-based interface with Flask
- Performance analysis and scalability testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the interactive web app (non-terminal UI):

```bash
python app.py
```

Then open `http://localhost:5000` in your browser. Select two cities and the optimization metric (distance/time/cost) to view the route on the map. The map shows markers for the chosen cities and draws the computed path polyline.

### Modes
- Shortest Path: Uses Dijkstra (and Bellman-Ford fallback) between Start and End.
- TSP: Choose algorithm: Nearest Neighbor, Two-Opt, Held-Karp (for small graphs), or Constrained.

### Constraints (TSP)
- Mandatory Stops: Comma-separated list of city names that must be included.
- Max Budget: Upper bound on route cost (uses selected weight units).
- Max Time: Upper bound on approximate time (uses distance/60 conversion when weight is distance).

Ensure city names match those in the dropdown (e.g., `Harare`, `Bulawayo`, `Mutare`).

## Project Structure

- `main.py` - Main application entry point
- `graph_algorithms/` - Core algorithm implementations
- `data/` - City data and distance matrices
- `visualization/` - Plotting and mapping components
- `app.py` and `templates/` - Flask web interface (Folium map)
- `tests/` - Unit tests and performance benchmarks
- `docs/` - Documentation and reports

## Algorithms Implemented

1. **Dijkstra's Algorithm** - Single-source shortest path
2. **Bellman-Ford Algorithm** - Handles negative weights
3. **Held-Karp Algorithm** - Exact TSP solution using DP
4. **Constraint-based DP** - Time windows and resource limits

## Performance Analysis

- Time complexity analysis for each algorithm
- Space complexity considerations
- Scalability testing with different graph sizes
- Comparison with heuristic approaches

## Detailed Report

See `REPORT.md` for an in-depth description of the project architecture, algorithms, UI, and scalability analysis.
