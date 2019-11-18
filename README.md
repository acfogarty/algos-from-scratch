Various ML algorithms from scratch (using numpy only, no scipy, no scikit-learn)

------------------------------------------------

# deep feed-forward neural network

sub-directory: `neural_network`

softmax output layer for classification, L2 regularization

------------------------------------------------

# decision tree

sub-directory: `decision_tree`

regression tree (main_regression.py) or classification tree (main_classification.py)

------------------------------------------------

# clustering algorithms

sub-directory: `clustering`

* hierarchical agglomerative clustering (with metric = Euclidean distance and linkage criteria = single-linkage)

* DBSCAN (with metric = Euclidean distance)

* Gaussian Mixture Model (EM algorithm)

------------------------------------------------

# Hidden Markov Model

sub-directory: `hmm`

Baum-Welch (EM) algorithm

TODO: switch to logspace for underflow; add Viterbi algorithm

------------------------------------------------

# Self-Organising Map

sub-directory: `som`

With visualisation via U-matrix

------------------------------------------------

# scores

sub-directory: `scores`

* ROC-AUC

------------------------------------------------

# Naive Bayes classifier for unlabelled ebook files

sub-directory: `naive_bayes_classifier`

* flag to choose between probabilities and log probabilities
* flag to switch on/off Laplacian correction (dataset must be big enough!)
* flag to choose between calculating prior from training set or using uniform prior

```
classify_scratch.py  # Naive Bayes from scratch
classify_skl.py  # uses scikit-learn for comparison
```

------------------------------------------------

# miscellaneous

sub-directory: `misc`

* properties of hash functions
* Kernel Density Estimation
* data structures (queue, linked list etc.)
