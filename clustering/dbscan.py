import matplotlib.pyplot as plt
import numpy as np

# DBSCAN parameters
min_points = 4
epsilon = 3

# number of datapoints to generate
npoints = 500  

# labelling of data points (contents of array cluster_labels):
# -1: noise
# 0: not assigned
# >1: cluster index


def main():

    # generate some non-linearly seperable data
    X = get_data_pattern(npoints)

    # assign points to clusters
    cluster_labels = dbscan(X)

    # plot with different colours for each cluster
    clusters = np.unique(cluster_labels)
    for cluster in clusters:
        points = X[cluster_labels == cluster]
        plt.scatter(points[:,0], points[:,1], label=cluster)
    plt.legend()
    plt.show()


def dbscan(X):
    '''
    Args:
        X: np.array of shape n_samples * n_features
    Returns:
        np.array of length n_samples containing cluster indices

    DBSCAN clustering algorithm
    algo in pseudocode from https://en.wikipedia.org/wiki/DBSCAN
    '''

    npoints = X.shape[0]
    nclusters = 0

    # initialize all points to a cluster_label of 0 (unassigned)
    cluster_labels = np.zeros(npoints, dtype=np.int64)

    for i in range(npoints):

        # point has already been assigned to a cluster
        if cluster_labels[i] != 0:
            continue

        neighbours = get_neighbours(i, X)

        # if point is in low-density region
        if len(neighbours) < min_points:
            # label as noise
            cluster_labels[i] = -1
            continue

        # start new cluster
        nclusters += 1
        cluster_labels[i] = nclusters

        for j in neighbours:

            if i == j:
                continue

            if cluster_labels[j] == -1:
                # reassign noise point to current cluster
                cluster_labels[j] = nclusters

            # point has already been assigned to a cluster
            if cluster_labels[j] != 0:
                continue

            cluster_labels[j] = nclusters

            more_neighbours = get_neighbours(j, X)
            if len(more_neighbours) >= min_points:
                new_neighbours = set(more_neighbours) - set(neighbours)
                neighbours.extend(new_neighbours)  # modification of list being looped over

    return cluster_labels


def get_neighbours(i, X):
    '''
    return indices of all points in X that are within
    distance epsilon of the point i

    Non-optimized
    '''

    neighbours = []
    for j in range(X.shape[0]):
        if dist_fn(X[i], X[j]) < epsilon:
            neighbours.append(j)

    return neighbours


def get_data_pattern(npoints):
    '''
    generate 2D data with a dense cluster inside a circle
    '''

    radius = 15.0  # radius of outer circle
    frac_ic = 0.2  # fraction of points in inner circle
    frac_no = 0.05  # fraction of points that are noise

    npoints_ic = int(npoints * frac_ic)
    npoints_no = int(npoints * frac_no)
    npoints_oc = npoints - npoints_ic - npoints_no

    # inner circle
    X_ic = np.zeros((npoints_ic, 2), dtype=np.float64)

    # outer circle
    angles = np.linspace(0, 2 * np.pi, num=npoints_oc)
    X_oc = np.asarray([[radius * np.cos(theta), radius * np.sin(theta)] for theta in angles])

    # noise
    X_no = np.random.normal(loc=0.0, scale=radius, size=(npoints_no, 2))

    X = np.concatenate((X_ic, X_oc, X_no), axis=0)

    X += np.random.randn(X.shape[0], X.shape[1])

    return X


def dist_fn(x1, x2):
    '''
    Euclidean distance btwn two vectors
    '''
    return np.sqrt(np.sum(np.power((x1 - x2), 2)))


if __name__ == '__main__':
    main()
