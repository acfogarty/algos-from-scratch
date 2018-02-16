import numpy as np
import neural_network
import sys
sys.path.append('../common_functions')
import common_fns
import matplotlib.pyplot as plt

# neural network from scratch
# multiple hidden layers
# activation function: tanh
# loss function: cross-entropy loss

filename = '../data/test-data-classification-smoking.csv'
target = 'hospitalised'
test_fraction = 0.25
n_nodes_per_hidden_layer = [6, 5]
n_hidden_layers = len(n_nodes_per_hidden_layer)
lambda_regul = 0.5  # regularisation hyperparameter
alpha = 0.1  # learning rate

# load dataset
X, Y, X_feature_names = common_fns.get_data(filename=filename, target=target)
X = X[:,[0, 3]] # XXXX 2 features, numerical only

# encode Y
Y = common_fns.one_hot_encoder(Y)

# normalise X
X, _, _ = common_fns.normalize_input_data(X)

# random split into test and train
X_train, Y_train, X_test, Y_test = common_fns.split_train_test(X, Y, test_fraction=test_fraction)

n_samples = len(Y_train)
n_features = X.shape[1]
n_classes = len(np.unique(Y_train))
print('n_samples', n_samples)
print('n_features', n_features)
print('input matrix shape', X.shape)
print('n_classes', n_classes)

# initialise model
nn = neural_network.NeuralNetwork(n_features, n_classes, n_nodes_per_hidden_layer)

# fit network on train set
nn.fit(X_train, Y_train, alpha=alpha, lambda_regul=lambda_regul)

# # plot data (2 features)
# for c in range(n_classes):
#     X_plot = X_train[Y_train[:,c] == 1][:,0]
#     Y_plot = X_train[Y_train[:,c] == 1][:,1]
#     plt.scatter(X_plot, Y_plot, label='class {}'.format(c))
# plt.legend()
# plt.savefig('check.png')

Y_predict = nn.predict(X_train, transpose_Y=True)
print(Y_predict)
print(Y_train)
loss = common_fns.cross_entropy_loss(Y_true=Y_train, Y_predict=Y_predict)
print('loss on training set', loss)

# make predictions on test set
Y_predict = nn.predict(X_test, transpose_Y=True)
print(Y_predict)
print(Y_test)
loss = common_fns.cross_entropy_loss(Y_true=Y_test, Y_predict=Y_predict)
print('loss on test set', loss)
