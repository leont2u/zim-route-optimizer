from .distance_matrix import DISTANCE_MATRIX

COST_MATRIX = {
    city: {dest: round(distance * 0.15, 2) for dest, distance in distances.items()}
    for city, distances in DISTANCE_MATRIX.items()
}
