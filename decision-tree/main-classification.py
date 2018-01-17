import numpy as np
import pandas as pd
import decision_tree as dtree
#from common_fns import splitTrainTest

filename = 'test-data-classification.csv'
target = 'hospitalised'
test_fraction = 0.25
max_depth = 2
min_sample_per_node = 2
criterion = 'gini'

X, Y, X_feature_names = dtree.get_data(filename=filename, target=target)

# X_train, Y_train, X_test, Y_test = split_train_test(X, Y, test_fraction=test_fraction)
X_train = X
Y_train = Y
X_test = X
Y_test = Y

tree = dtree.fit_decision_tree(X=X_train, Y=Y_train, max_depth=max_depth, min_sample_per_node=min_sample_per_node, criterion=criterion)

dtree.print_tree(tree=tree, X_feature_names=X_feature_names)

Y_predict = dtree.predict_all(tree=tree, X=X_test)

dtree.calculate_scores(Y_predict=Y_predict, Y_ref=Y_test)
