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