import numpy as np
import sys
sys.path.append('../common-functions')
from common_fns import tanh, softmax


class neural_network:

    def __init__(self, n_features, n_classes, n_nodes_per_hidden_layer, activation_fn=None, output_activation_fn=None):

        self.n_features = n_features
        self.n_classes = n_classes

        self.n_nodes_per_layer = [n_features]
        for n in n_nodes_per_hidden_layer:
            self.n_nodes_per_layer.append(n)
        self.n_nodes_per_layer.append(n_classes)
        self.n_layers = len(self.n_nodes_per_layer)

        if activation_fn is None:
            self.activation_fn = tanh
        else:
            self.activation_fn = activation_fn

        if output_activation_fn is None:
            self.output_activation_fn = softmax
        else:
            self.output_activation_fn = output_activation_fn

        self.weights = []

        print('initialised a neural network with {} classes, {} features'.format(self.n_classes, self.n_features))
        print('and {} layers with nodes per layer: {}'.format(self.n_layers, self.n_nodes_per_layer))

    def forward_propogate_layer(self, a, W, activation_fn):
        '''a: np.array with dimension n_nodes_prev_layer * n_samples
        W: weights, np.array with dimensions n_nodes * n_nodes_prev_layer
        '''
        print('a',a)
        print('W',W)
        z = np.dot(W, a)
        print('z',z)
        return activation_fn(z)

    def predict(self, X):
        '''
        X: np.array with dimensions n_samples * n_features
        returns: np.array with dimensions n_samples * n_classes'''
        A = X.transpose()

        for W in self.weights[:-1]:
            # add bias node to A (add row of 1s at start of matrix)
            bias = np.ones((1, A.shape[1]))
            A = np.vstack((bias, A))
            # get node outputs
            A = self.forward_propogate_layer(A, W, self.activation_fn)

        bias = np.ones((1, A.shape[1]))
        A = np.vstack((bias, A))
        Y = self.forward_propogate_layer(A, self.weights[-1], self.output_activation_fn)

        Y = Y.transpose()

        return Y

    def initialize_weights(self):
        # TODOO add different initialisation options
        print('Initialising weights with dimensions:')
        for i in range(self.n_layers-1):
            # self.n_nodes_per_layer[i] + 1 because of bias term
            W = np.random.random((self.n_nodes_per_layer[i+1], self.n_nodes_per_layer[i] + 1))
            W *= 2.0
            W -= 1.0
            self.weights.append(W)
            print('layer {}: {}'.format(i, W.shape))
            print(W)


