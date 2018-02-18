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
    X_test = X_shuffled[:n_samples_test]
    Y_test = Y_shuffled[:n_samples_test]
    X_train = X_shuffled[n_samples_test:]
    Y_train = Y_shuffled[n_samples_test:]

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

    # check matrices are in correct format with one 1 per line
    for Y in [Y_true, Y_predict]:
        if not np.all(np.sum(Y_true, axis=1) == 1):
            print('One of the matrices is not in correct format in cross_entropy_loss')
            print('Should be n_samples * n_classes')
            quit()

    n_samples = Y_true.shape[0]

    mask = (Y_true == 1)  
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


def hardmax(Y):
    '''
    input:
        array of floats of dimension n_samples*n_classes containing class probabilities
    return:
        array of integers of dimension n_samples*n_classes with 1 for the class
        with max probability and 0 for all others
    '''

    axis = 1
    return np.apply_along_axis(lambda x: np.where(x == x.max(), 1, 0), axis, Y)


def accuracy_onehot(Y_true, Y_predict):
    '''
    calculate accuracy = (true positives + true negatives) / (total population)
    for two one-hot-encoded arrays
    if Y_predict contains probabilities instead of 1s and 0s, hardmax is applied first
    input:
        Y_true, Y_predict: array of dimension n_samples*n_classes containing integers 1, 0 (one-hot-encoding) or probabilities
    output: float
    '''

    if Y_predict.shape != Y_true.shape:
        raise Exception('in accuracy Y_predict shape = {} and Y_true shape = {}'.format(Y_predict.shape, Y_true.shape))

    # detect whether Y_predict contains probabilities instead of one-hot encoding, and convert if necessary
    # (detect if at least one element is not equal to 0 or 1)
    if not np.all((Y_predict == 1) | (Y_predict == 0)):
        Y_predict = hardmax(Y_predict)

    Y_true = Y_true.astype(np.int64)
    Y_predict = Y_predict.astype(np.int64)

    n_samples = Y_predict.shape[0]
    axis = 1
    matches = np.apply_along_axis(np.all, axis, (Y_predict == Y_true))
    n_matches = matches.sum()

    return float(n_matches) / float(n_samples)


def confusion_matrix(Y_true, Y_predict, print_matrix=False):
    '''calculate confusion matrix for two label-encoded arrays
    assumes classes are integers starting from 0
    actual class: rows
    predicted class: columns
    input:
        Y_true, Y_predict: array of length n_samples with classes encoded as integers
    output:
        array of dimension n_classes * n_classes
    '''
    Y_predict = Y_predict.astype(np.int64)
    Y_true = Y_true.astype(np.int64)

    n_classes = max(np.max(Y_predict), np.max(Y_true)) + 1

    matrix = np.zeros((n_classes, n_classes))

    for y_i, y_true_i in zip(Y_predict, Y_true):
        matrix[y_true_i][y_i] += 1

    if print_matrix:
        string_format = '{:5d}'
        classes = range(n_classes)
        print('       predicted classes')
        print('      ' + ''.join([string_format.format(c) for c in classes]))
        print('      ' + ''.join(['-----' for c in classes]))
        for c in classes:
            print(string_format.format(c) + '|' + ''.join([string_format.format(int(i)) for i in matrix[c]]))

    return matrix


def calculate_scores_from_confusion_matrix(matrix):
    '''
    calculate accuracy from multiclass confusion matrix 
    input:
        array of dimension n_classes * n_classes with actual class: rows, predicted class: columns
    output:
        accuracy (float), precision and recall (arrays of length n_classes)
    '''

    scores = {}
    
    scores['accuracy'] = np.sum(np.diagonal(matrix)) / np.sum(matrix)

    # tp / (tp + fp) sum over columns
    scores['precision'] = np.diagonal(matrix) / np.sum(matrix, axis=0)

    # recall tp / (tp + fn), sum over rows
    scores['recall'] = np.diagonal(matrix) / np.sum(matrix, axis=1)

    return scores


def cross_validation_scores(model, X, Y, n_fold, score_fn, shuffle=True, train_subset_size=None):
    '''
    inputs:
        model: class with functions .fit() and .predict()
        X: np.array with dimensions n_samples * n_features
        Y: np.array with dimensions n_samples * n_classes
        n_fold: number of CV folds, int
        score_fn: function that takes arguments (Y_true, Y_predict)
    if train_subset_size != None, only train on a subset of that size of the 
    training set (for constructing learning curves). The size of the test set
    is not affected
    '''

    n_samples = Y.shape[0]
    n_samples_per_fold = n_samples / n_fold

    # shuffle both arrays using same permutation
    if shuffle:
        p = np.random.permutation(len(Y))
        X, Y = X[p], Y[p]

    training_scores = np.zeros((n_fold))
    test_scores = np.zeros((n_fold))

    for i in range(n_fold):

        # split into n_fold folds
        indices = np.arange(n_samples)
        test_mask = ((indices >= (i * n_samples_per_fold)) & (indices < ((i + 1) * n_samples_per_fold)))
        X_train, Y_train = X[~test_mask], Y[~test_mask]
        X_test, Y_test = X[test_mask], Y[test_mask]

        if train_subset_size:
            # only take subset of training set (will be the same points for each call unless shuffle==True)
            X_train, Y_train = X[:train_subset_size], Y[:train_subset_size]

        # fit network on train set
        model.fit(X_train, Y_train)

        # predict on train
        Y_predict = model.predict(X_train, transpose_Y=True)
        training_scores[i] = score_fn(Y_true=Y_train, Y_predict=Y_predict)

        # predict on test
        Y_predict = model.predict(X_test, transpose_Y=True)
        test_scores[i] = score_fn(Y_true=Y_test, Y_predict=Y_predict)

    return training_scores, test_scores


def learning_curve(model, X, Y, score_fn, cv_n_fold, train_sizes):
    '''
    if some integers in train_sizes are greater than the max possible number with the given sample size and cv_n_fold, they will simply be ignored
    train_sizes: list of int
    cv_n_fold: int
    score_fn: function that takes arguments (Y_true, Y_predict)
    '''

    n_samples = Y.shape[0]
    # max number of training set samples we can have with cv_n_fold-CV
    n_max_samples_train = int(float(n_samples) / cv_n_fold * (cv_n_fold - 1))

    training_scores = []
    test_scores = []
    trainsetsize_xtics = []

    # cross validation for different training set sizes (test set size is always the same)
    for train_subset_size in train_sizes:

        if train_subset_size > n_max_samples_train:
            continue
    
        training_scores_c, test_scores_c = cross_validation_scores(model, X, Y, n_fold=cv_n_fold, score_fn=score_fn, shuffle=True, train_subset_size=train_subset_size)

        training_scores.append(training_scores_c)
        test_scores.append(test_scores_c)
        trainsetsize_xtics.append(train_subset_size)

    np.asarray(training_scores)
    np.asarray(test_scores)
    np.asarray(trainsetsize_xtics)

    return trainsetsize_xtics, training_scores, test_scores


def print_model_test(model, X, Y, score_fn, oh_enc, l_enc):
    Y_predict = model.predict(X, transpose_Y=True)
    # print('Y_predict')
    # print(Y_predict.shape)
    # print('Y')
    # print(Y.shape)
    loss = score_fn(Y_true=Y, Y_predict=Y_predict)
    print('score', loss)
    # convert probabilities to 0 and 1 (max prob = 1)
    Y_predict_hard = hardmax(Y_predict)
    # print('Y_predict_hard')
    # print(Y_predict_hard)
    Y_decoded = oh_enc.decode(Y)
    Y_predict_decoded = oh_enc.decode(Y_predict_hard)
    print('Y_decoded')
    print(Y_decoded)
    print('Y_predict_decoded')
    print(Y_predict_decoded)
    cm = confusion_matrix(Y=Y_predict_decoded, Y_true=Y_decoded, print_matrix=True)
    l_enc.print_decoder()
    scores = calculate_scores_from_confusion_matrix(cm)
    for k in scores.keys():
        print(k, scores[k])
