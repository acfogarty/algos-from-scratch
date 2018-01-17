import numpy as np
import pandas as pd
import decision_tree as dtree
import sys  
sys.path.append('../common-functions')
import common_fns

# classification tree from scratch

filename = '../data/test-data-classification-smoking.csv'
target = 'hospitalised'
test_fraction = 0.25
max_depth = 2
min_sample_per_node = 2
criterion = 'gini'
prediction_type = 'classification'

X, Y, X_feature_names = common_fns.get_data(filename=filename, target=target)

X_train, Y_train, X_test, Y_test = common_fns.split_train_test(X, Y, test_fraction=test_fraction)
X_train = X
X_test = X
Y_train = Y
Y_test = Y

tree = dtree.fit_decision_tree(X=X_train, Y=Y_train, max_depth=max_depth, min_sample_per_node=min_sample_per_node, criterion=criterion)

dtree.print_tree(tree=tree, X_feature_names=X_feature_names)

Y_predict = dtree.predict_all(tree=tree, X=X_test, prediction_type=prediction_type)

dtree.calculate_scores(Y_predict=Y_predict, Y_ref=Y_test)
