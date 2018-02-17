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


    def decode(self, Y):
        '''
        assumes classes were labelled with integers from 0 to nclasses-1
        input: 
            np.array with dimensions n_samples * n_classes
        returns:
            1D np.array of integers of length n_samples
        '''
        axis = 1
        return np.apply_along_axis(np.argmax, axis, Y)


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
        self.int_to_label = dict(zip(range(self.nclasses), classes))

        Y_encoded = self.transform(Y)

        return Y_encoded


    def decode(self, Y):
        '''
        input:
            np.array of integers of length n_samples
        returns:
            list of strings of length n_samples
        '''

        Y_decoded = [self.int_to_label[i] for i in Y]

        return Y_decoded


    def print_decoder(self):

        print(self.int_to_label)
