import numpy as np


def euclidean_distance_squared(x1, x2):
    '''returns square of euclidean distance
    between two np.arrays'''

    return np.sum((x1 - x2)**2)


def distance_between_sets(set1, set2, distance_matrix, criterion='single-linkage'):
    '''Returns distance between two sets of points
    with linkage criterion: single-linkage (min of pairwise distances)
    input:
        set1, set2: list of indices
        distance_matrix: 2D np.array of pairwise distances, with same indices as set1, set2'''

    distances = []
    for index1 in set1:
        for index2 in set2:
            ttuple = tuple(sorted([index1, index2]))
            distances.append(distance_matrix[ttuple])

    if criterion == 'single-linkage':
        return min(distances)
    else:
        print('Unknown criterion ', criterion)
        quit()
