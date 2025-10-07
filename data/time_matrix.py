from .distance_matrix import DISTANCE_MATRIX
# Time matrix (estimated travel time in hours)
TIME_MATRIX = {
    city: {
        dest: round(distance / 60, 1)  # Assuming average speed of 60 km/h
        for dest, distance in distances.items()
    }
    for city, distances in DISTANCE_MATRIX.items()
}
