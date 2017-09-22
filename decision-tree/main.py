import numpy as np
import pandas as pd
#from common_fns import splitTrainTest


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


def calc_gini_impurity(nodes=None):
    '''calculate gini impurity (measure of how often a randomly
    chosen element would be incorrectly identified, i.e.
    meausure of misclassification)
    nodes should be dict of datasets identified by node_id, e.g.
    {0: {'X': X_0, 'Y': Y_0}, 1: {'X': X_1, 'Y': Y_1}}'''
    classes = [0, 1]  # TODOO don't hardcode, detect at root of tree
    n_classes = len(classes)

    for node_id in nodes.keys():
        Y = nodes[node_id]['Y']
        n_samples = len(Y)
        counts = dict(zip(np.unique(Y, return_counts=True)))
        for cclass in classes:
            try:
                prob = counts[cclass]/n_samples
            except KeyError:
                prob = 0.0
    return gini_impurity


def information_gain():
    return


def make_binary_split(i_feature=None, split_value=None, X=None, Y=None):
    '''split X and Y into two nodes at the feature i_feature using
    the split_value
    Y has dimension n_samples
    X has dimensions n_samples * n_features TODO
    return the split datasets and the split value'''
    mask = X[:, i_feature] > split_value
    X_left = X[mask]  # > split_value
    Y_left = Y[mask]
    X_right = X[~mask]  # <= split_value
    Y_right = Y[~mask]

    return {'left': {'X': X_left, 'Y': Y_left}, 'right': {'X': X_right, 'Y': Y_right}}


def get_best_split(X=None, Y=None):
    '''for a given dataset X and Y, find the best feature and feature-value to split on'''
    n_features = X.shape[1]
    best_gini_impurity = 1000
    for i_feature in range(n_features):
        unique_values = set(X[:, i_feature])
        for split_value in unique_values:
            nodes = make_binary_split(i_feature, split_value, X, Y)
            gini_impurity = calc_gini_impurity(nodes)
            if gini_impurity < best_gini_impurity:
                best_feature = i_feature
                best_split_value = split_value
                best_gini_impurity = gini_impurity
                best_nodes = nodes

    return best_feature, best_split_value, best_nodes, best_gini_impurity


def fit_decision_tree(X=None, Y=None, max_depth=None):
    '''input'''
    if Y.size != X.shape[0]:
        print('Error! Dimension mismatch between X and Y in fit_decision tree')
        quit()

    tree = {}
    get_best_split(X=X, Y=Y)

    return tree


def predict(tree=None, X=None):
    return Y


def calculate_accuracy_score(Y_predict=None, Y_ref=None):
    return score


filename = 'test-data.csv'
target = 'hospitalised'
test_fraction = 0.25
max_depth = 10

X, Y, X_feature_names = get_data(filename=filename, target=target)

# X_train, Y_train, X_test, Y_test = split_train_test(X, Y, test_fraction=test_fraction)
X_train = X
Y_train = Y

tree = fit_decision_tree(X=X_train, Y=Y_train, max_depth=max_depth)

Y_predict = predict(tree=tree, X=X_test)

score = calculate_accuracy_score(Y_predict=Y_predict, Y_ref=Y_test)
