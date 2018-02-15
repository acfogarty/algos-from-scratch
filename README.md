Various ML algorithms from scratch (using numpy only, no scipy, no scikit-learn)

------------------------------------------------

# deep feed-forward neural network

sub-directory: `neural-network`

softmax output layer for classification, L2 regularization

------------------------------------------------

# decision tree

sub-directory: `decision-tree`

regression tree (main-regression.py) or classification tree (main-classification.py)

------------------------------------------------

# hierarchical agglomerative clustering

sub-directory: `hac`

with metric = Euclidean distance and linkage criteria = single-linkage

------------------------------------------------

# Naive Bayes classifier for unlabelled ebook files

sub-directory: `naive-bayes-classifier`

* reads labelled datafiles
* partitions into training and test sets
* trains Naive Bayes classifier
* flag to choose between probabilities and log probabilities
* flag to switch on/off Laplacian correction (dataset must be big enough!)
* flag to choose between calculating prior from training set or using uniform prior

Usage:

python classify-skl.py  # uses scikit-learn for comparison
python classify-scratch.py  # Naive Bayes from scratch

control variables (vocabulary size, train/test split fraction, flags for Laplacian correction, uniform prior, etc.) all hard-coded

