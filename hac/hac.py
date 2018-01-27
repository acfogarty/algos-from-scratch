import sys  
sys.path.append('../common-functions')
import distances
import numpy as np
import copy

# hierarchical agglomerative clustering
# metric = Euclidean distance
# linkage criteria = single-linkage


def main():

    n_clusters_stop = 8  # stop clustering when number of clusters falls to this number
    n_samples = 100
    
    # generate 1D data
    target_data = np.random.uniform(low=0.0, high=100.0, size=(n_samples, 1))
    print('clustering {} samples'.format(n_samples))
    
    # pairwise distance matrix
    sample_distance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            sample_distance_matrix[i, j] = distances.euclidean_distance_squared(target_data[i], target_data[j])
    
    # initialise clusters, one per sample
    # clusters are stored as lists of indices
    clusters = [[i] for i in range(n_samples)]
    
    # keep aggregating clusters until we reach the stopping criterion
    while len(clusters) > n_clusters_stop:
        clusters = agglomerate_clusters(clusters, sample_distance_matrix)
    
    # print values in final clusters
    for cluster in clusters:
        print([target_data[i] for i in cluster])


def agglomerate_clusters(clusters, sample_distance_matrix):
    '''merge the two closest clusters
    input:
        clusters: list of lists of cluster indices
        sample_distance_matrix: 2D np.array of distances between samples,
                                with same indices as in list of clusters
    returns:
        new list of lists of cluster indices'''

    nclusters = len(clusters)
    print('Now have {} clusters {}'.format(nclusters, clusters))

    # create matrix filled with nan (because we will only fill one half)
    cluster_distance_matrix = np.full((nclusters, nclusters), np.nan)
    for i in range(nclusters):
        for j in range(i+1, nclusters):
            cluster_distance_matrix[i, j] = distances.distance_between_sets(clusters[i], clusters[j], sample_distance_matrix)
    max_value = np.nanmax(cluster_distance_matrix)
    cluster_distance_matrix[np.isnan(cluster_distance_matrix)] = max_value + 1

    # get the indices of the two closest clusters
    indices_closest_clusters = np.unravel_index(cluster_distance_matrix.argmin(), cluster_distance_matrix.shape)

    # merge the clusters
    print('Merging', indices_closest_clusters)
    clusters[indices_closest_clusters[0]].extend(copy.deepcopy(clusters[indices_closest_clusters[1]]))
    del clusters[indices_closest_clusters[1]]

    return clusters


if __name__ == "__main__":
    main()
