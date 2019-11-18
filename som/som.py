import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import RegularPolyCollection
import matplotlib.cm as cm


def main():
    # dimensions of output lattice
    lattice_dim = (3, 3)
    
    # number of iterations
    # (sigma and learning_rate are updated after each iteration)
    n_timesteps = 50
    
    # number of input datapoints to sample in each iteration
    n_samples_per_step = 100
   
    hyperparameters = {}
    # hyperparameters controlling size of BMU's neighbourhood
    # radius at time 0
    hyperparameters['sigma_0'] = max(*lattice_dim) * 0.5
    # exponential half-life for radius decline
    hyperparameters['lambda_sigma'] = 2.5 
    
    # hyperparameters controlling learning rate
    # learning rate at time 0
    hyperparameters['learning_rate_0'] = 0.5
    # exponential half-life for learning reate decline
    hyperparameters['lambda_lr'] = 2.5
    
    # dimensions of visualisation matrix
    umatrix_dim = np.asarray(lattice_dim)*2-1
    
    # dimensions n_samples * n_features
    X = create_example_data()
    
    # number of inputs
    samples_dim = X.shape[0]
    
    # dimension of each neuron weights vector and each input vector
    features_dim = X.shape[1]

    weights = initialize_weights(lattice_dim, features_dim)

    distance_to_bmu = []
    plot_step = 10
    for t in range(n_timesteps):
        for i in range(n_samples_per_step):
            weights, dist = update_weights(X, weights, hyperparameters, t)
            distance_to_bmu.append(dist)
            
            if (len(distance_to_bmu) % plot_step) == 0:
                plt.plot(range(len(distance_to_bmu)), distance_to_bmu)
                plt.show()


def create_example_data():
    """
    create fake data containing feature vectors that can be grouped into
    four approximately similar types
    """

    type1 = np.asarray([106,105,103.5,1])
    type2 = np.asarray([0.5,30,30,70])
    type3 = np.asarray([3,3,3,3])
    type4 = np.asarray([0.5,90,90,0.5])
    
    X = np.vstack((np.tile(type1,(4,1)),
                   np.tile(type2,(3,1)),
                   np.tile(type3,(4,1)),
                   np.tile(type4,(9,1))))

    # add some random noise to each vector
    X += np.random.normal(size=X.shape)
    
    X = (X - X.mean()) / X.std()

    return X


def initialize_weights(lattice_dim, features_dim):
    """
    Args:
        lattice_dim (vector of length 2): dimensions of output lattice (n_height * n_width)
        features_dim (float): length of input vector and of weights vectors
    Returns: 
        3D np.array
    """
    weights = np.random.normal(size=(*lattice_dim, features_dim))

    print('Initialized weights with shape ', weights.shape)
    print('Output lattice contains {} nodes'.format(weights.shape[0] * weights.shape[1]))
    
    return weights


def update_weights(X, weights, hyperparameters, t):
    samples_dim = X.shape[0]
    lattice_dim = weights.shape[:2]

    # randomly choose an input vector
    index_random = np.random.randint(low=0, high=samples_dim)

    # compare input vector to all neurons
    dist = np.zeros(lattice_dim, dtype=np.float64)
    for i in range(lattice_dim[0]):
        for j in range(lattice_dim[1]):
            dist[i,j] = np.linalg.norm(weights[i,j]-X[index_random])

    # Best Matching Unit
    index_bmu = np.unravel_index(dist.argmin(), dist.shape)
    #print('distance to Best Matching Unit', dist[index_bmu])
    
    # get radius of BMU neighbourhood at time t
    sigma_t = hyperparameters['sigma_0'] * np.exp(-t / hyperparameters['lambda_sigma'])
    sigma_t_sq_2 = 2 * sigma_t * sigma_t
    
    # get learning rate at time t
    learning_rate_t = hyperparameters['learning_rate_0'] * np.exp(-t / hyperparameters['lambda_lr'])

    # update weights based on distance from BMU
    for i in range(lattice_dim[0]):
        for j in range(lattice_dim[1]):
            
            # factor depending on grid dist of node from BMU
            dist_sq = (i - index_bmu[0])*(i - index_bmu[0]) + (j - index_bmu[1])*(j - index_bmu[1])
            theta_ij = np.exp(-dist_sq / sigma_t_sq_2)
            
            update_factor = learning_rate_t * theta_ij * (X[index_random] - weights[i,j])
            weights[i,j] += update_factor
            
    return weights, dist[index_bmu]


def assign_samples_to_nodes():
    # number of samples assigned to a node
    n_samples_per_node = np.zeros(lattice_dim, dtype=np.float64)
    # average feature vector of all samples assigned to a node
    average_vector_per_node = np.zeros((*lattice_dim, features_dim), dtype=np.float64)
    # list of indices of all samples assigned to a node
    samples_per_node = []
    for i in range(lattice_dim[0]):
        samples_per_node.append([])
        for j in range(lattice_dim[1]):
            samples_per_node[i].append([])
    
    # for each sample
    for ix in range(samples_dim):
        # find the nearest node
        dist = np.zeros(lattice_dim, dtype=np.float64)
        for i in range(lattice_dim[0]):
            for j in range(lattice_dim[1]):
                dist[i,j] = np.linalg.norm(weights[i,j]-X[ix])

        # update contents of nearest node
        index_closest = np.unravel_index(dist.argmin(), dist.shape)
        n_samples_per_node[index_closest] += 1.0
        samples_per_node[index_closest[0]][index_closest[1]].append(ix)
        average_vector_per_node[index_closest] += X[ix]

    average_vector_per_node /= np.atleast_3d(n_samples_per_node)
    
    return average_vector_per_node, samples_per_node, n_samples_per_node


if __name__=='__main__':
    main()
