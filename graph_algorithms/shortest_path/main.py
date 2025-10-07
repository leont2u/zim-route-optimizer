# main.py
from models.zimbabwe_graph import ZimbabweGraph
from .comparator import ShortestPathComparator

if __name__ == "__main__":
    graph = ZimbabweGraph("distance")
    comparator = ShortestPathComparator(graph)

    print("Testing shortest path from Harare to Victoria Falls:")
    results = comparator.compare_algorithms("Harare", "Victoria Falls")

    for algo, result in results.items():
        print(f"\n{algo}:")
        print(f"  Distance: {result.distance}")
        print(f"  Path: {' -> '.join(result.path)}")
        print(f"  Execution time: {result.execution_time:.6f}s")
        print(f"  Nodes visited: {result.nodes_visited}")

    print("\nBenchmarking algorithms...")
    stats = comparator.benchmark_algorithms(50)
    print("\nPerformance Statistics:")
    for algo, stat in stats.items():
        print(f"\n{algo}:")
        print(f"  Average execution time: {stat['avg_time']:.6f}s")
        print(f"  Average nodes visited: {stat['avg_nodes']:.1f}")
        print(f"  Successful runs: {stat['successful_runs']}")
