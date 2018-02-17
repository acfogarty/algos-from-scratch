import numpy as np
import copy
import sys
sys.path.append('../common_functions')
import common_fns
from common_fns import tanh, softmax, tanh_gradient


# indexing

# layers are numbered from 0 (input layer), through the hidden layers, to n_layer (output layer)
# A^{(0)} = X.transpose()
# Z^{(1)} = W^{(1)} x A^{(0)}
# A^{(1)} = S(Z^{(1)})
# Z^{(2)} = W^{(2)} x A^{(1)}
# A^{(2)} = S(Z^{(2)})
# ...
# Z^{(L)} = W^{(L)} x A^{(L-1)}
# A^{(L)} = softmax(Z^{(L)})
# Y_hat = A^{(L)}.transpose()


class NeuralNetwork:

    def __init__(self, n_features, n_classes, n_nodes_per_hidden_layer, activation_fn=None, activation_fn_gradient=None, output_activation_fn=None):

        self.n_features = n_features
        self.n_classes = n_classes

        self.n_nodes_per_layer = [n_features]
        for n in n_nodes_per_hidden_layer:
            self.n_nodes_per_layer.append(n)
        self.n_nodes_per_layer.append(n_classes)
        self.n_layers = len(self.n_nodes_per_layer)

        if activation_fn is None:
            self.activation_fn = tanh
            self.activation_fn_gradient = tanh_gradient
        else:
            self.activation_fn = activation_fn
            self.activation_fn_gradient = activation_fn_gradient

        if output_activation_fn is None:
            self.output_activation_fn = softmax
        else:
            print('For now, only the gradient of the softmax output activation fn is implemented')
            quit()
            # self.output_activation_fn = output_activation_fn

        self.weights = []
        # node inputs
        self.a_by_layer = []
        # node outputs
        self.z_by_layer = []

        print('initialised a neural network with {} classes, {} features'.format(self.n_classes, self.n_features))
        print('and {} layers with nodes per layer: {}'.format(self.n_layers, self.n_nodes_per_layer))

    def forward_propogate_layer(self, a, W, activation_fn):
        '''a: np.array with dimension n_nodes_prev_layer * n_samples
        W: weights, np.array with dimensions n_nodes * n_nodes_prev_layer
        '''
        #print('a',a)
        #print('W',W)
        z = np.dot(W, a)
        #print('z',z)
        return z, activation_fn(z)

    def predict(self, X, transpose_Y=False):
        '''
        X: np.array with dimensions n_samples * n_features
        returns: np.array with dimensions n_classes * n_samples
        if transpose_Y==True, returns n_samples * n_classes
        '''

        A = X.transpose()

        for l in range(self.n_layers - 1):

            # add bias node to A (add row of 1s at start of matrix)
            # same for Z, so that it will have correct dimensions in backprop
            bias = np.ones((1, A.shape[1]))
            A = np.vstack((bias, A))
            if l > 0:
                self.a_by_layer[l-1] = copy.deepcopy(A)

            # get activation function
            if l == (self.n_layers - 2):
                fn = self.output_activation_fn
            else:
                fn = self.activation_fn

            # move forward one layer
            W = self.weights[l]
            Z, A = self.forward_propogate_layer(A, W, fn)
            self.z_by_layer[l] = copy.deepcopy(Z)

        Y = A
        if transpose_Y:
            Y = Y.transpose()

        # for l in range(self.n_layers - 1):
        #     print(l+1, 'a', self.a_by_layer[l].shape)
        #     print(l+1, 'z', self.z_by_layer[l].shape)
        #     print(l+1, 'w', self.weights[l].shape)

        return Y

    def initialize_weights(self):
        # TODOO add different initialisation options

        print('Initialising weights with dimensions:')

        for i in range(1, self.n_layers):

            # weights
            # self.n_nodes_per_layer[i-1] + 1 because of bias term
            W = np.random.random((self.n_nodes_per_layer[i], self.n_nodes_per_layer[i-1] + 1))
            W *= 2.0
            W -= 1.0
            self.weights.append(W)
            print('layer {}: {}'.format(i, W.shape))

            # inputs and outputs of each layer (not including layer 0, the input layer)
            place_holder = np.array(0)
            self.a_by_layer.append(place_holder)
            self.z_by_layer.append(place_holder)

    def fit(self, X, Y, loss_tolerance=0.0005, alpha=0.05, lambda_regul=0.0):
        '''backprop and gradient descent to fit self.weights on X and Y
        loss = -1 / n_samples * sum (y*log(y_hat)) + lambda / (2 * n_samples) * sum (w^2)
        inputs:
            X: np.array with dimensions n_samples * n_features
            Y: np.array with dimensions n_samples * n_classes
            loss_tolerance: iterate until old_loss - new_loss < loss_tolerance
            alpha: learning rate
            lambda_regul: regularisation parameter lambda
        '''

        Y = Y.transpose()
        bias = np.ones((X.shape[0], 1))
        X_b = np.hstack((bias, X))

        self.initialize_weights()

        loss_diff = 1000
        loss_prev = 1000

        # get factor to rescale weights by for regularisation
        n_samples = X.shape[0]
        regul_rescale_factor = (1.0 - alpha * lambda_regul / n_samples)

        while loss_diff > loss_tolerance:

            # prediction with current value of W
            Y_hat = self.predict(X)

            # gradient descent
            # output layer
            dJ_dZ = Y_hat - Y
            dJ_dW = np.dot(dJ_dZ, self.a_by_layer[-2].transpose())
            self.weights[-1][:,1:] *= regul_rescale_factor  # don't regularize first column (bias weights)
            self.weights[-1] -= alpha * dJ_dW

            # loop backwords through other layers
            for l in range(-2, -1 * (self.n_layers - 1), -1):
                # TODO cache tanh activation fn to reuse when calculating gradient
                temp = np.dot(self.weights[l+1][:,1:].transpose(), dJ_dZ)  # exclude first column of weights matrix, which was for bias
                dJ_dZ = temp * self.activation_fn_gradient(self.z_by_layer[l])  # element-wise multiplication
                if l == (-1 * (self.n_layers - 1)):  # input layer
                    A_T = X_b  # X is already transposed
                else:  # intermediate layers
                    A_T = self.a_by_layer[l-1].transpose()
                dJ_dW = np.dot(dJ_dZ, A_T)
                self.weights[l][:,1:] *= regul_rescale_factor  # don't regularize first column (bias weights)
                self.weights[l] -= alpha * dJ_dW

            # loss for current value of W
            loss = common_fns.cross_entropy_loss(Y_true=Y.transpose(), Y_predict=Y_hat.transpose())
            loss_diff = loss_prev - loss
            loss_prev = loss
            print('loss in fit (without regularisatio)', loss, loss_diff)

