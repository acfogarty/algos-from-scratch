import math
import numpy as np
import matplotlib.pyplot as plt
from dbscan import get_data_pattern

n_gauss = 3  # number of Gaussians
n_points = 500
n_dim = 2  # dimensionality of data (n_features)

# Gaussian Mixture Model
# Expectation-Maximisation algorithm

# https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

# model parameters theta are the means (mu_k), stdev (sigma_k)
# and weight (alpha_k) of each Gaussian k

# z_ik = 1 if datapoint i comes from Gaussian component k, and 0 otherwise

# alpha_k = p(z_k) = probability that a randomly selected datapoint
# comes from Gaussian k

# E-step: for current theta values, compute w_ik for each datapoint
# i and each Gaussian k
# w_ik = p(z_ik = 1 | x_i, theta)
#      = p_k(x_i | z_k, theta_k) * alpha_k / (sum_m p_m(x_i | z_m, theta_m) * alpha_m)
# W is a matrix n_points * n_gauss, each row sums to 1

# M-step: update model parameters
# n_k = sum_n_points w_ik (effective number of datapoints assigned to Gaussian k)
# alpha_k = n_k / n_points
# mu_k = (1/n_k) * sum (w_ik * x_i)
# sigma_k = weighted stdev


def main():

    # data from three gaussians
    X = generate_gaussian_data(n_points, n_gauss)
    cluster_labels = gmm(X, n_gauss)
    plot_clustering_results(X, cluster_labels)

    # # non-linearly seperable data (circle in circle)
    # X = get_data_pattern(n_points)
    # cluster_labels = gmm(X, n_gauss)
    # plot_clustering_results(X, cluster_labels)


def gmm(X, n_gauss):
    '''
    Gaussian mixture model

    Args:
        n_gauss (int): number of Gaussians
        X (np.array): data with dimensions n_samples * n_dim
    Returns:
        np.array of int of length n_samples containing cluster indices
    '''

    n_points = X.shape[0]
    n_dim = X.shape[1]

    # initialise parameters
    w = np.zeros((n_points, n_gauss), dtype=np.float64)
    alpha = np.asarray([1.0/n_gauss for i in range(n_gauss)])
    mu = np.random.randn(n_gauss, n_dim)
    # generate a random square matrix which is symmetric and
    # whose determinant is positive (ad - bc > 0)
    sigma = np.abs(np.random.randn(n_gauss, n_dim, n_dim))
    ad = sigma[:, 0, 0] * sigma[:, 1, 1] - np.abs(np.random.randn(n_gauss))
    sigma[:, 0, 1] = sigma[:, 1, 0] = np.sqrt(np.where(ad > 0, ad, 0))

    # initialize all points to a cluster_label of 0 (unassigned)
    cluster_labels = np.zeros(n_points, dtype=np.int64)

    loglikelihood_prev = 1000.0
    loglikelihood = 0.0

    h = 0
    while abs(loglikelihood_prev - loglikelihood) > 0.1:

        # plot progress of algorithm
        if (h % 5 == 0):
            plt.clf()
            clusters = np.unique(cluster_labels)
            for cluster in clusters:
                points = X[cluster_labels == cluster]
                plt.scatter(points[:,0], points[:,1], label=cluster)
            plt.scatter(mu[:,0], mu[:,1], label='means')
            plt.legend()
            plt.show()
        h += 1

        loglikelihood_prev = loglikelihood
        loglikelihood = 0.0

        # E-step
        for i in range(n_points):

            # get probability p(x_i|theta_k) for each Gaussian
            p = np.zeros(n_gauss, dtype=np.float64)
            for k in range(n_gauss):
                mahalanobis_dist = (X[i] - mu[k])

                det_cov = np.linalg.det(sigma[k])
                inv_cov = np.linalg.inv(sigma[k])

                prefactor = np.sqrt(math.pow((2 * np.pi), n_dim) * det_cov)
                p[k] = np.exp(np.dot(np.dot(np.transpose(mahalanobis_dist), inv_cov), mahalanobis_dist) / -2.0) / prefactor
                #y = multivariate_normal.pdf(X[i], mean=mu[k], cov=cov)

            loglikelihood += np.log(np.sum(p * alpha))

            # get membership weights
            w[i] = p * alpha / np.sum(p * alpha)

            # get Gaussian with max probability
            cluster_labels[i] = np.argmax(p)

        # M-step
        # update Gaussian parameters
        # number of effective points assigned to this cluster = sum of membership weights
        n_k = np.sum(w, axis=0)
        alpha = n_k / float(n_points)

        mu = np.dot(np.transpose(w), X) / n_k.reshape(3, 1)

        # TODO vectorise
        for k in range(n_gauss):
            s = 0.0
            for i in range(n_points):
                s += w[i,k]*np.matmul((X[i] - mu[k]).reshape(2,1), (X[i] - mu[k]).reshape(1,2))
            s /= n_k[k]
            sigma[k] = s

        print('loglikelihood', loglikelihood)

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


def generate_gaussian_data(n_points, n_gauss):
    means = np.random.randn(n_gauss, n_dim) * 3
    stdevs = np.abs(np.random.randn(n_gauss, n_dim))
    print('means of generated data', means)
    print('sigma of generated data', stdevs)
    n_points_k = int(n_points / n_gauss)

    X = []
    for mu, sigma in zip(means, stdevs):
        X_k = np.random.multivariate_normal(mean=mu, cov=np.diag(sigma), size=(n_points_k))
        X.append(X_k)

    X = np.concatenate(X, axis=0)
    plt.scatter(X[:,0], X[:,1])
    plt.show()

    return X


if __name__ == '__main__':
    main()
