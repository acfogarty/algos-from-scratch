import numpy as np
import pandas as pd

# useful functions replicating behaviour of sklearn


def split_train_test(X, Y, test_fraction=0.2):
    '''randomly split X and Y into two parts of size splitFraction*total and (1-splitFraction)*total'''

    n_samples_test = int(len(Y) * test_fraction)

    # shuffle both arrays using same permutation
    p = np.random.permutation(len(Y))
    X_shuffled, Y_shuffled = X[p], Y[p]

    # split into test set (test_fraction of data) and test set (1 - test_fraction of data)
    X_test = X[:n_samples_test]
    Y_test = Y[:n_samples_test]
    X_train = X[n_samples_test:]
    Y_train = Y[n_samples_test:]

    print('Split sample data into training set of size ', len(Y_train), ' and test set of size ', len(Y_test))

    return X_train, Y_train, X_test, Y_test


def get_data(filename, target):
    '''target: header of column containing target variable

    Loads data from csv file and returns matrix X (features),
    vector Y (target),
    and the column headers (feature names) for matrix X
    '''
    df = pd.read_csv(filename)
    X = df.drop(target, axis=1).values
    Y = df[target].values
    cols = df.columns.values
    X_feature_names = cols[cols != target]
    print("Predicting target '{}' from features {}".format(target, X_feature_names))
    print('Features array X has shape {}, target vector Y has length {}'.format(X.shape, Y.size))

    return X, Y, X_feature_names


def one_hot_encoder(Y):
    '''
    only works if Y contains integers
    assumes all integers between 0 and Y.max() are present in Y

    input: 1D np.array of length n_samples
    returns:
        np.array with dimensions n_samples * n_classes
    '''

    # n_classes = len(classes)
    n_classes = Y.max() + 1
    n_samples = Y.size

    Y_encoded = np.zeros((n_samples, n_classes))
    Y_encoded[np.arange(n_samples), Y] = 1

    return Y_encoded


def tanh(x):
    return np.tanh(x)


def tanh_gradient(x):
    '''TODO used cached value of tanh(x)'''
    t = np.tanh(x)
    return 1 - t * t


def softmax(x):
    '''
    Returns softmax for each row in input array
    input: np.array with dimensions n_nodes * n_samples
    '''
    
    e_x = np.exp(x - np.max(x))  # subtract this constant for numerical stability
    e_sum = e_x.sum(axis=0)
    return e_x / e_sum.reshape(1, len(e_sum))


def cross_entropy_loss(Y_true, Y_predict):
    '''
    Returns mean cross entropy error over all samples
    input: np.arrays with dimensions n_samples * n_classes
    Y_true contains 1 and 0
    Y_predict contains probabilities
    Doesn't include error handling for zeroes in Y_predict

    L(y, yhat) = -(1/N) * sum_N( sum_C( y*log(yhat) ) )
    N: n_samples
    C: n_classes
    '''
    n_samples = Y_true.shape[1]

    # Y_true contains 1 and 0, so we use it as index
    mask = (Y_true == 1)  # TODOO check Y_true is in correct format with one 1 per line
    loss_per_sample = np.log(Y_predict[mask])

    loss = np.sum(loss_per_sample) / float(n_samples) * -1

    return loss


def cross_entropy_loss_gradient_wrt_yhat(Y_true, Y_predict):
    '''dL/dyhat = y/yhat'''

    return Y_true / Y_predict * -1.0


def cross_entropy_loss_gradient_wrt_z(Y_true, Y_predict):
    '''dL/dyhat = y/yhat'''

    return Y_predict - Y_true


def cross_entropy_loss_gradient_wrt_W(Y_true, Y_predict, X):
    '''dL/dW_ij = x_j(yhat_i - y_i)
    i is over n_classes
    j is over n_nodes
    L(Y_true, Y_predict) = cross entropy
    Y_predict = softmax(Z)
    Z = WX
    input dimensions:
        Y_true, Y_predict: n_classes * n_examples
        X: n_nodes * n_examples
    local variable dimensions:
        Z: n_classes * n_examples
        W: n_classes * n_nodes 
    output dimensions:
        dW: n_classes * n_nodes
    '''

    return np.dot((Y_predict - Y_true), X.transpose())


def cross_entropy_loss_per_sample(Y_true, Y_predict):
    '''
    Returns cross entropy error for each sample
    input: np.arrays with dimensions n_samples * n_classes
    Y_true contains 1 and 0
    Y_predict contains probabilities
    Doesn't include error handling for zeroes in Y_predict

    L(y, yhat) = -( sum_C( y*log(yhat) ) )
    N: n_samples
    C: n_classes
    '''
    n_samples = Y_true.shape[1]
    
    # transpose so that mask indexing works
    Y_true = Y_true.transpose()
    Y_predict = Y_predict.transpose()

    # Y_true contains 1 and 0, so we use it as index
    mask = (Y_true == 1)  # TODOO assert Y_true is in correct format with one 1 per line
    loss_per_sample = np.log(Y_predict[mask])
    loss_per_sample *= -1

    return loss_per_sample


def normalize_input_data(X):
    '''
    input: np.array with dimensions n_features * n_samples
    returns:
        X: normalised np.array with dimensions n_features * n_samples
        X_mean, X_std: np.array with dimensions n_features
    '''
    X = X.astype(np.float64)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X -= X_mean
    X /= X_std

    return X, X_mean, X_std
