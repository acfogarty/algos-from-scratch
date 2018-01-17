import numpy as np
import pandas as pd

max_node_id = 0  # global variable for tracking id of all nodes created


def calc_gini_impurity(Y=None):
    '''calculate gini impurity of a list of elements (measure
    of how often a randomly
    chosen element would be incorrectly identified, i.e.
    meausure of misclassification)
    gini_impurity for the list = sum over classes (p*(1-p)) =
    1 - (sum over classes(p*p))
    where p is the relative frequency of each class in the list
    '''
    gini_imp_node = 0.0
    n_samples = float(len(Y))

    # get dict of number of times each class appears in this node
    classes, counts = np.unique(Y, return_counts=True)
    counts = dict(zip(list(classes), list(counts)))

    for cclass in classes:
        prob = float(counts[cclass])/n_samples
        gini_imp_node += prob * prob
        # print('In node {}, class {} occurs with frequency {}'.format(node_id, cclass, prob))
    gini_imp_node = 1.0 - gini_imp_node
    # print('Node {}, gini impurity {}'.format(Y, gini_imp_node))

    return gini_imp_node, n_samples


def calc_criterion_nodes(nodes=None, criterion='gini'):
    '''calc weighted criterion (such as gini impurity
    or SSE) of set of nodes.
    Nodes should be list of dict of datasets e.g.
    [{'node_id': 0, 'X': X_0, 'Y': Y_0}, {'node_id': 1, 'X': X_1, 'Y': Y_1}]'''
    criterion_val = 0.0
    n_total_samples = 0.0

    for node in nodes:
        Y = node['Y']
        if criterion == 'gini':
            criterion_val_node, n_samples = calc_gini_impurity(Y)
        elif criterion == 'sse':
            criterion_val_node, n_samples = Y.var(), float(len(Y))
        else:
            print('Unknown value {} for criterion'.format(criterion))
            quit()

        # weight by number of samples in this node
        criterion_val += criterion_val_node * n_samples
        n_total_samples += n_samples

    # normalise weights
    criterion_val /= n_total_samples

    return criterion_val


def information_gain():
    return 0


def make_binary_split(i_feature=None, split_value=None, X=None, Y=None):
    '''split X and Y into two nodes at the feature i_feature using
    the split_value
    Y has dimension n_samples
    X has dimensions n_samples * n_features TODO
    return the split datasets and the split value'''

    global max_node_id

    mask = X[:, i_feature] < split_value
    X_left = X[mask]  # < split_value
    Y_left = Y[mask]
    X_right = X[~mask]  # >= split_value
    Y_right = Y[~mask]

    return [{'node_id': 0, 'X': X_left, 'Y': Y_left}, {'node_id': 1, 'X': X_right, 'Y': Y_right}]


def get_best_split(X=None, Y=None, criterion='gini'):
    '''for a given dataset X and Y, find the best feature and feature-value to split on'''
    n_features = X.shape[1]
    best_criterion_val = 1000
    for i_feature in range(n_features):
        # get all unique values this feature takes in the data
        unique_values = set(X[:, i_feature])
        # test a split on every possible split value
        for split_value in unique_values:
            # print('Testing split on value {} of feature {}'.format(split_value, i_feature))
            nodes = make_binary_split(i_feature, split_value, X, Y)
            criterion_val = calc_criterion_nodes(nodes, criterion)
            if criterion_val < best_criterion_val:
                best_feature = i_feature
                best_split_value = split_value
                best_criterion_val = criterion_val
                best_nodes = nodes
    # print('Choosing to split on feature {} and value {} into nodes {}. Gini impurity is {}'.format(best_feature, best_split_value, best_nodes, best_gini_impurity))

    return best_feature, best_split_value, best_nodes, best_criterion_val


def fit_decision_tree(X=None, Y=None, max_depth=None, min_sample_per_node=None, criterion=None):
    '''for a given dataset and hyperparameters, build a decision tree'''

    global max_node_id

    if Y.size != X.shape[0]:
        print('Error! Dimension mismatch between X and Y in fit_decision tree')
        quit()

    max_node_id = 0
    root_node = {'node_id': 0, 'X': X, 'Y': Y, 'terminal': False, 'depth': 0}
    attempt_split(root_node, max_depth=max_depth, min_sample_per_node=min_sample_per_node, criterion=criterion)

    return [root_node]


def attempt_split(node=None, max_depth=None, min_sample_per_node=None, criterion=None):
    '''check if node should be split, based on hyperparameters'''

    global max_node_id

    # print('At start of attempt_split with node {}'.format(node))

    if len(node['Y']) < min_sample_per_node:
        node['terminal'] = True
    if node['depth'] >= max_depth:
        node['terminal'] = True
    if (len(np.unique(node['Y'])) == 1):  # pure node
        node['terminal'] = True

    if not node['terminal']:
        best_feature, best_split_value, best_nodes, best_criterion_val = get_best_split(X=node['X'], Y=node['Y'], criterion=criterion)
        node['split_feature'] = best_feature
        node['split_value'] = best_split_value
        node['criterion_val'] = best_criterion_val
        # print('Parent node is now {}'.format(node))

        for child_node in best_nodes:

            child_node['depth'] = node['depth'] + 1
            child_node['terminal'] = False
            max_node_id += 1
            child_node['node_id'] = max_node_id

            # print('attempting split in child node {}'.format(child_node))

            attempt_split(child_node, max_depth=max_depth, min_sample_per_node=min_sample_per_node, criterion=criterion)

        node['children'] = best_nodes

    # print('At end of attempt node is now {}'.format(node))


def print_node(node=None, X_feature_names=None, parent='is_root'):

    has_children = ('children' in node.keys())

    if has_children:
        children_node_ids = [child['node_id'] for child in node['children']]
        feature = X_feature_names[node['split_feature']]
        string = 'node_id={}, depth={}, terminal={}, parent={}, children={}, split on {}={}, values={}'.format(node['node_id'], node['depth'], node['terminal'], parent, children_node_ids, feature, node['split_value'], node['Y'])
    else:
        string = 'node_id={}, depth={}, terminal={}, parent={}, values={}'.format(node['node_id'], node['depth'], node['terminal'], parent, node['Y'])

    print(string)

    if has_children:
        for child in node['children']:
            print_node(node=child, X_feature_names=X_feature_names, parent=node['node_id'])


def print_tree(tree=None, X_feature_names=None):

    print()
    print('Tree:')
    for node in tree:
        print_node(node, X_feature_names)


def predict_all(tree=None, X=None, prediction_type='classification'):
    '''use tree to get predicted Y_i for each sample X_i
    possible values of prediction_type are: regression, classification'''

    print()
    print('Predictions:')

    # TODOO vectorize
    Y = []
    root_node = tree[0]
    for sample in X:
        Y.append(predict(root_node, sample, prediction_type))
    Y = np.asarray(Y)

    return Y


def get_next_node(node=None, Xi=None):
    '''decide which child node to go to next based on contents of Xi
    left node: < split value
    right node: >= split value'''

    index = node['split_feature']

    if Xi[index] < node['split_value']:  # check if <= TODOO
        child_index = 0
    else:
        child_index = 1

    return node['children'][child_index]


def predict(node=None, Xi=None, prediction_type=None):
    '''travel through tree'''

    if node['terminal']:
        if prediction_type == 'classification':
            return terminal_predict_classification(node)
        elif prediction_type == 'regression':
            return terminal_predict_regression(node)
        else:
            print('Unknown prediction type ', prediction_type)
            quit()
    else:
        next_node = get_next_node(node, Xi)
        return predict(next_node, Xi, prediction_type)


def terminal_predict_classification(node=None):
    '''return most frequent class in a node
    if there's a tie, take first value'''

    classes, counts = np.unique(node['Y'], return_counts=True)
    return classes[np.argmax(counts)]


def terminal_predict_regression(node=None):
    '''return mean of predictions in a node'''

    return node['Y'].mean()


def calculate_scores(Y_predict=None, Y_ref=None):
    '''accuracy = correct / total'''

    if len(Y_predict) != len(Y_ref):
        print('Error! Arrays of different length in calculate_scores')
        return 0.0

    true_neg = len(Y_predict[(Y_predict == 0) & (Y_ref == 0)])
    true_pos = len(Y_predict[(Y_predict == 1) & (Y_ref == 1)])
    false_neg = len(Y_predict[(Y_predict == 0) & (Y_ref == 1)])
    false_pos = len(Y_predict[(Y_predict == 1) & (Y_ref == 0)])

    print('            prediction')
    print('            pos     neg')
    print('actual pos {:4d}    {:4d}'.format(true_pos, false_neg))
    print('actual neg {:4d}    {:4d}'.format(false_pos, true_neg))

    accuracy = float(true_neg + true_pos) / float(len(Y_ref))
    print()
    print('Accuracy:', accuracy)

    precision = float(true_pos) / float(true_pos + false_pos)
    print()
    print('Precision:', precision)

    recall = float(true_pos) / float(true_pos + false_neg)
    print()
    print('Recall:', recall)

    f1 = 2 * precision * recall / (precision + recall)
    print()
    print('F1 score:', f1)
