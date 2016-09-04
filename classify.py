import io
import sys
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import numpy as np
import nltk
from nltk.corpus import stopwords
import string

# global variables
stopwords = stopwords.words('english')
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
vocabSize = 20 #no. of words used as features

def main():

  dataDir = '/Users/fogarty/ereader-backup/train'
  labelledDataFile = dataDir + '/list-ebook-files.txt'

  #get names and labels of all pre-labeled ebook files
  filenames, sampleLabels = getDataList(dataDir, labelledDataFile)

  #get vocabulary matrix (dim: nsamples*vocabSize)
  sampleVocabMatrix = getVocabMatrix(filenames)
  print '# Created vocabulary matrix of dimensions ',sampleVocabMatrix.shape
  print sampleVocabMatrix

  #get probabilities P(l) for all l in labels
  labelProbabilities = getLabelProbabilities(sampleLabels)
  print '# label probabilities ',labelProbabilities


def getFeatures(words,vocabulary):
  '''check which elements of the vocabulary are in the list words'''
  wordSet = set(words) #for speed
  features = [] #ordered list of T or F
  for word in vocabulary:
    features.append(word in wordSet)
  return features

def getDataList(rootdir, labelledDataFile):
  '''get names and labels of all pre-labelified ebook files'''
  filenames = []
  labels = []
  ff = open(labelledDataFile,'r')
  for line in ff:
    line = line.split()
    filenames.append(rootdir+'/'+line[0])
    labels.append(line[1])
  ff.close()
  return filenames, labels

def getVocabMatrix(filenames):
  '''get vocabulary matrix
     dimensions: len(nsamples)*len(vocabulary)
     contents: 1 or 0'''

  # extract contents of each document
  tokensList = []
  for filename in filenames:
    f = io.open(filename,'r',encoding="utf-8")
    raw = f.read()
    f.close()
    processed = raw.lower()
    #processed = [s.translate(remove_punctuation_map) for s in processed]
    tokens = nltk.word_tokenize(processed)
    tokensNoStopwords = [w for w in tokens if w not in stopwords]
  
    tokensList.append(tokensNoStopwords)
    #text = nltk.Text(tokensNoStopwords)
    #fdist = nltk.FreqDist(text)
  
  #get most common words in the entire set of documents
  allFd = nltk.FreqDist([y for x in tokensList for y in x]) #flatten
  vocabulary = [w for w,n in allFd.most_common(vocabSize)]
  
  vocabMatrix = []
  for tokens in tokensList:
    features = getFeatures(tokens,vocabulary)
    vocabMatrix.append(features)
  vocabMatrix = np.asarray(vocabMatrix)

  return vocabMatrix

def getLabelProbabilities(sampleLabels):
  '''calculates percentage for each label in list of all labels'''
  setOfLabels = set(sampleLabels)
  nSamples = float(len(sampleLabels))
  labelProbabilities = {}
  for l in setOfLabels:
    prob = float(sampleLabels.count(l))/nSamples
    labelProbabilities[l] = prob
  return labelProbabilities

    
 
if __name__ == '__main__':
  main() 
