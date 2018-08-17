import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import math
import numpy as np

n_gauss = 3  # number of Gaussians
npoints = 500
n_dim = 2  # dimensionality of data (n_features)


# EM algorithm

# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

# model parameters theta are the means (mu_k) and stdev (sigma_k)
# of each Gaussian k

# z_ik = 1 if datapoint i comes from Gaussian component k, and 0 otherwise

# alpha_k = p(z_k) = probability that a randomly selected datapoint
# comes from Gaussian k

# E-step: for current theta values, compute w_ik for each datapoint
# i and each Gaussian k
# w_ik = p(z_ik = 1 | x_i, theta)
#      = p_k(x_i | z_k, theta_k) * alpha_k / (sum_m p_m(x_i | z_m, theta_m) * alpha_m)
# W is a matrix n_points * n_gauss, each row sums to 1

# M-step: update theta values
# n_k = sum_n_points w_ik (effective number of datapoints assigned to Gaussian k)
# alpha_k = n_k / n_points
# mu_k = (1/n_k) * sum (w_ik * x_i)
# sigma_k = weighted stdev


def main():

    # data from three gaussians
    X = generate_gaussian_data(npoints, n_gauss)
    cluster_labels = gmm(X, n_gauss)
    plot_clustering_results(X, cluster_labels)

    # non-linearly seperable data (circle in circle)
    X = get_data_pattern(npoints)
    cluster_labels = gmm(X, n_gauss)
    plot_clustering_results(X, cluster_labels)


def gmm(X, n_gauss):
    '''
    Gaussian mixture model

    Args:
        n_gauss (int): number of Gaussians
        X (np.array): data with dimensions n_samples * n_dim
    Returns:
        np.array of int of length n_samples containing cluster indices
    '''

    npoints = X.shape[0]
    n_dim = X.shape[1]

    # initialise parameters
    mu = np.random.randn(n_gauss, n_dim)
    sigma = np.abs(np.random.randn(n_gauss, n_dim))
    alpha = np.asarray([1.0/n_gauss for i in range(n_gauss)])
    # initialize all points to a cluster_label of 0 (unassigned)
    cluster_labels = np.zeros(npoints, dtype=np.int64)
    w = np.zeros((npoints, n_gauss), dtype=np.float64)

    converged = False

    loglikelihood_prev = 1000.0
    loglikelihood = 0.0

    while abs(loglikelihood_prev - loglikelihood) > 1.0:
        #plt.clf()
        #clusters = np.unique(cluster_labels)
        #for cluster in clusters:
        #    points = X[cluster_labels == cluster]
        #    plt.scatter(points[:,0], points[:,1], label=cluster)
        #plt.scatter(mu[:,0], mu[:,1], label='means')
        #plt.legend()
        #plt.show()

        print('loglikelihood', loglikelihood)

        loglikelihood = 0.0

        # E-step
        for i in range(npoints):

            # get probability p(x_i|theta_k) for each Gaussian
            p = np.zeros(n_gauss, dtype=np.float64)
            for k in range(n_gauss):
                mahalanobis_dist = (X[i] - mu[k])

                cov = np.diag(sigma[k])
                det_cov = np.linalg.det(cov)
                inv_cov = np.linalg.inv(cov)

                prefactor = np.sqrt(math.pow((2 * np.pi), n_dim) * det_cov)
                p[k] = np.exp(np.dot(np.dot(np.transpose(mahalanobis_dist), inv_cov), mahalanobis_dist) / -2.0) / prefactor

            loglikelihood += np.log(np.sum(p * alpha))

            # get membership weights
            w[i] = p * alpha / np.sum(p * alpha)

            # get Gaussian with max probability
            cluster_labels[i] = np.argmax(p)

        # M-step
        # update Gaussian parameters
        # number of effective points assigned to this cluster = sum of membership weights
        n_k = np.sum(w, axis=0)
        alpha = n_k / float(npoints)

        mu = np.dot(np.transpose(w), X) / n_k.reshape(3, 1)
        #sigma = np.std(X_j, axis=0)

        loglikelihood_prev = loglikelihood

    return cluster_labels


def plot_clustering_results(X, cluster_labels):
    '''
    plot with different colours for each cluster
    '''
    clusters = np.unique(cluster_labels)
    for cluster in clusters:
        points = X[cluster_labels == cluster]
        plt.scatter(points[:,0], points[:,1], label=cluster)
    plt.legend()
    plt.show()


def generate_gaussian_data(npoints, n_gauss):
    means = np.random.randn(n_gauss, n_dim) * 3
    stdevs = np.abs(np.random.randn(n_gauss, n_dim))
    print('means of generated data', means)
    print('sigma of generated data', stdevs)
    npoints_k = int(npoints / n_gauss)

    X = []
    for mu, sigma in zip(means, stdevs):
        X_k = np.random.multivariate_normal(mean=mu, cov=np.diag(sigma), size=(npoints_k))
        X.append(X_k)

    X = np.concatenate(X, axis=0)
    plt.scatter(X[:,0], X[:,1])
    #plt.show()

    return X


if __name__ == '__main__':
    main()
