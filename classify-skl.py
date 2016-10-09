import io
import sys
import numpy as np
import pandas as pd
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import nltk
import string
import commonfns
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Naive Bayes classifier

# l is member of L (labels)
# f is member of F (features)
# P(l|F) = P(l,F)/P(F)
# P(l,F) = P(l)*Product_F[P(f|l)]
# best label = argmax ( P(l)*Product_F[P(f|l)] )
# in log form: argmax ( log(P(l)) + Sum_F[log(P(f|l))] )

#reads labelled datafiles
#partitions into training and test sets
#trains Naive Bayes classifier
#flag to choose between probabilities and log probabilities
#flag to switch on/off Laplacian correction (dataset must be big enough!)
#flag to choose between calculating prior from training set or using uniform prior

def main():

  vocabSize = 5 #no. of words used as features (most common words in training set)
  splitFraction = 0.6 #fraction used for training
  laplaceCorr = True #True - use Laplacian correction
  calcPrior = False #True - calc P(l) from training set, False - assume uniform prior
  logProb = True #True - sum of log probabilities, False - product of probabilities

  if logProb == True and laplaceCorr == False:
    print 'Shouldn''t use logProb == True and laplaceCorr == False because log(0) may occur'
    quit()

  dataDir = '/Users/fogarty/ereader-backup/train'
  labelledDataFile = dataDir + '/list-ebook-files.txt'

  #get names and labels of all pre-labeled ebook files
  filenames, sampleLabels = commonfns.getDataList(dataDir, labelledDataFile)

  #get vocabulary 
  vocabulary = commonfns.getVocabulary(filenames,vocabSize)
  print '# Created vocabulary of length ',len(vocabulary)

  #get vocabulary matrix (dim: nsamples*vocabSize, entries: 1 or 0)
  countVectorizer = CountVectorizer(input='filename', vocabulary=vocabulary, decode_error='ignore', binary=True, stop_words=commonfns.stopwords)
  sampleVocabMatrix = countVectorizer.fit_transform(filenames)
  print '# Created vocabulary matrix of dimensions ',sampleVocabMatrix.shape

  #clf = MultinomialNB(alpha=1.0,fit_prior=calcPrior)
  #clf.fit(xTrain, yTrain)
  #yPred = clf.predict_proba(xVal)[:, 1]
 
if __name__ == '__main__':
  main() 
