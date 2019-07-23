import numpy as np
import matplotlib.pyplot as plt

#############################
# Kernel Density Estimation #
#############################


def main():

    ################################
    # define underlying distribution
    # (sum of two Gaussians)
    ################################
    
    mean1 = 4.5
    mean2 = 1.0
    var1 = 0.3
    var2 = 1.2
    w1 = 3
    w2 = 1
    
    x = np.arange(-2, 6, 0.1)
    y = w1*gauss(x, mean1, var1) + w2*gauss(x, mean2, var2)
    y = y / np.sum(y)
    
    plt.plot(x, y, label='true distribution', linestyle=':')
    
    ###################################################
    # generate random data from underlying distribution
    ###################################################
    
    npoints = 500
    npoints1 = int(npoints * w1 / float(w1 + w2))
    npoints2 = int(npoints * w2 / float(w1 + w2))
    data = np.concatenate((np.random.normal(loc=mean1,
                                            scale=np.sqrt(var1),
                                            size=npoints1),
                           np.random.normal(loc=mean2,
                                            scale=np.sqrt(var2),
                                            size=npoints2)))
    
    plt.scatter(data,
                np.zeros(npoints1+npoints2),
                label='data drawn from distribution',
                marker='x')
    
    ###########################
    # kernel density estimation
    ###########################
    
    def plot_kde(x, data, wrapper, bandwidth, title):
        kde = np.asarray([calculate_kde(xi, data,
                                        bandwidth,
                                        wrapper) for xi in x])
        plt.plot(x, kde/np.sum(kde),
                 label='KDE with {} kernel, bw={}'.format(title, bandwidth))
    
    
    plot_kde(x, data, gauss_wrapper, bandwidth=2.0, title='Gaussian')
    plot_kde(x, data, gauss_wrapper, bandwidth=0.5, title='Gaussian')
    plot_kde(x, data, gauss_wrapper, bandwidth=0.1, title='Gaussian')
    plot_kde(x, data, uniform_wrapper, bandwidth=0.5, title='uniform')
    plot_kde(x, data, uniform_wrapper, bandwidth=0.1, title='uniform')
    
    #############################
    # histogram of generated data
    #############################
    
    #plt.hist(data, normed=True, alpha=0.2)
    
    ##########
    # plot all
    ##########
    
    plt.legend(loc='upper left')
    plt.show()


def gauss(x, mean, var):
    '''
    normal distribution
    '''
    prefactor = np.sqrt(2 * np.pi * var * var)
    return np.exp(-1 * (x - mean)*(x - mean) / (2*var)) / prefactor


def gauss_wrapper(x):
    mean = 0
    var = 1
    return gauss(x, mean, var)


def uniform_wrapper(x):
    mean = 0
    halfwidth = 0.2
    height = 0.333
    return np.where(abs(x-mean) <= halfwidth, height, 0)


def calculate_kde(x, data, bandwidth, kernel_function):
    '''
    calculate the kernel density estimation at point x, using
    the supplied data
    Args:
      x (float)
      data (np.array)
      bandwidth (float)
      kernel_function (python function)
    '''
    return np.mean(kernel_function((x - data) / bandwidth)) / bandwidth


if __name__ == '__main__':
    main()
