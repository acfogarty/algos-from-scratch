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


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


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
    print(loss)

    return loss
