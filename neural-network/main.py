import numpy as np
import neural_network
import sys
sys.path.append('../common-functions')
import common_fns

# neural network from scratch
# one hidden layer
# activation function: tanh
# loss function: cross-entropy loss

filename = '../data/test-data-classification-smoking.csv'
target = 'hospitalised'
test_fraction = 0.25
n_nodes_per_hidden_layer = [10, 7]
n_hidden_layers = len(n_nodes_per_hidden_layer)
lambda_l = 0.1  # regularisation hyperparameter
alpha = 0.1  # learning rate

# load dataset
X, Y, X_feature_names = common_fns.get_data(filename=filename, target=target)
#X = X[:,[0,1]] # XXXX 2 features

# encode Y
Y = common_fns.one_hot_encoder(Y)

# random split into test and train
X_train, Y_train, X_test, Y_test = common_fns.split_train_test(X, Y, test_fraction=test_fraction)

n_samples = len(Y_train)
n_features = X.shape[1]
n_classes = len(np.unique(Y))
print('n_samples', n_samples)
print('n_features', n_features)
print('input matrix shape', X.shape)
print('n_classes', n_classes)

# initialise model
nn = neural_network.neural_network(n_features, n_classes, n_nodes_per_hidden_layer)
nn.initialize_weights()

# fit network on train set

# make predictions on test set
Y_predict = nn.predict(X)
print(X, Y_predict)
