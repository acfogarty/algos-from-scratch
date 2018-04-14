# true positive rate = sensitivity = hit rate = recall = TP / (TP + FN)
# we maximise this to miss as few positive points as possible
# (first column of confusion matrix)

# false positive rate = fall-out = FP / (FP + TN)
# how many of the total negative data points are mistakenly considered positive
# we minimise this to avoid misclassifying data points
# (second column of confusion matrix)

#                actual class
#                   1       0
# predicted  1     TP      FP
# class      0     FN      TN

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def calc_roc_point(y_true, y_predict, print_matrix=False):
    '''input: np.arrays containing 1's and 0's only'''

    # contingency table
    TP = np.sum((y_true == 1) & (y_predict == 1))
    TN = np.sum((y_true == 0) & (y_predict == 0))
    FN = np.sum((y_true == 1) & (y_predict == 0))
    FP = np.sum((y_true == 0) & (y_predict == 1))

    if print_matrix:
        print('                actual class')
        print('                  1      0  ')
        print('predicted  1    {}     {}'.format(TP, FP))
        print('class      0    {}     {}'.format(FN, TN))

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    return (FPR, TPR)


def probability_to_prediction(probability, threshold):
    '''input:
         probability: array of probabilities
         threshold: value above which y=1
    '''

    y = np.where(probability > threshold, 1, 0)

    return y


def generate_roc(y_true, y_predict_probability):
    '''
    generate all FPR and TPR points for plotting ROC
    input:
         y_predict_probability: array of probabilities
         y_true: array containing 1's and 0's only
    '''
    roc_points = []

    # get point on curve for all thresholds from 0 to 1
    for threshold in np.arange(0, 1, 0.05):
        y_random = probability_to_prediction(y_predict_probability, threshold)
        roc_points.append(calc_roc_point(y_true, y_random))

    roc_points = np.asarray(roc_points)
    fpr = roc_points[:, 0]
    tpr = roc_points[:, 1]
    return fpr, tpr


# generate 'true' data (containing 0's and 1's)
n_examples = 1000
p_positive = 0.8  # proportion of examples that are positive
y_true_probabilities = np.random.uniform(low=0.0, high=1.0, size=(n_examples))
# y_true_probabilities = np.random.beta(a=5, b=1, size=(n_examples))
y_true = probability_to_prediction(y_true_probabilities, 1.0 - p_positive)

# values, counts = np.unique(y_true, return_counts=True)
# print(values, counts)

# generate random data from uniform probability distribution
random_uniform_probs = np.random.uniform(low=0.0, high=1.0, size=(n_examples))
fpr, tpr = generate_roc(y_true, random_uniform_probs)
plt.plot(fpr, tpr, label='random uniform distribution')

# generate fake data from good classifier
temp = np.where(y_true == 1, 1.0, -1.0)
temp += np.random.normal(size=(n_examples))  # add random noise
good_clf_probs = 1.0 / (1.0 + np.exp(-temp))  # convert to probabilities
fpr, tpr = generate_roc(y_true, good_clf_probs)
plt.plot(fpr, tpr, label='good classifier')

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()
