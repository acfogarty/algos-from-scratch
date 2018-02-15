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

# load dataset
X, Y, X_feature_names = common_fns.get_data(filename=filename, target=target)

# random split into test and train
X_train, Y_train, X_test, Y_test = common_fns.split_train_test(X, Y, test_fraction=test_fraction)

# fit tree on train set
tree = dtree.fit_decision_tree(X=X_train, Y=Y_train, max_depth=max_depth, min_sample_per_node=min_sample_per_node, criterion=criterion)

# print tree for debugging/reference
dtree.print_tree(tree=tree, X_feature_names=X_feature_names)

# make predictions on test set
Y_predict = dtree.predict_all(tree=tree, X=X_test, prediction_type=prediction_type)

# calculate various classification scores
dtree.calculate_scores(Y_predict=Y_predict, Y_ref=Y_test)
