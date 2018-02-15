import numpy as np


class OneHotEncoder():

    def __init__(self):
        self.nclasses = 0

    def transform(self, Y):
        '''
        only works if Y contains integers

        input: 1D np.array of integers of length n_samples
        returns:
            np.array with dimensions n_samples * n_classes
        '''

        n_samples = Y.size

        Y_encoded = np.zeros((n_samples, self.n_classes))
        Y_encoded[np.arange(n_samples), Y] = 1

        return Y_encoded

    def fit_transform(self, Y):
        '''
        fit encoder on Y, and also output encoded Y

        only works if Y contains integers
        assumes all integers between 0 and Y.max() are present in Y

        input: 1D np.array of integers of length n_samples
        returns:
            np.array with dimensions n_samples * n_classes
        '''

        self.n_classes = Y.max() + 1

        Y_encoded = self.transform(Y)

        return Y_encoded


class LabelEncoder():
    '''maps e.g. ['France', 'Germany', 'Belgium', 'France'] to [0, 1, 2, 0]'''

    def __init__(self):
        self.nclasses = 0
        self.label_to_int = {}

    def transform(self, Y):
        '''
        input:
            list of strings of length n_samples
        returns:
            np.array of integers of length n_samples
        '''

        Y_encoded = np.asarray([self.label_to_int[y_i] for y_i in Y])

        return Y_encoded

    def fit_transform(self, Y):
        '''
        fit encoder on Y, and also output encoded Y

        input:
            list of strings of length n_samples
        returns:
            np.array of integers of length n_samples
        '''

        Y = np.asarray(Y)
        classes = np.unique(Y)
        self.nclasses = len(classes)
        self.label_to_int = dict(zip(classes, range(self.nclasses)))

        Y_encoded = self.transform(Y)

        return Y_encoded

