import io
import sys
import numpy as np
import pandas as pd
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import nltk
from nltk.corpus import stopwords
import string

#Naive Bayes
# l is member of labels
# f is member of features
# P(l|features) = P(l,features)/P(features)
# P(l,features) = P(l)*Product[P(f|l)]

# global variables
stopwords = stopwords.words('english')
stopwords += ['.',',',';','?','!','-',':','',"n't","'d","'re","'s","'m"]
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
vocabSize = 20 #no. of words used as features

def main():

  dataDir = '/Users/fogarty/ereader-backup/train'
  labelledDataFile = dataDir + '/list-ebook-files.txt'

  #get names and labels of all pre-labeled ebook files
  filenames, sampleLabels = getDataList(dataDir, labelledDataFile)

  #get vocabulary matrix (dim: nsamples*vocabSize)
  vocabulary, sampleVocabMatrix = getVocabMatrix(filenames)
  print '# Created vocabulary matrix of dimensions ',sampleVocabMatrix.shape
  print sampleVocabMatrix

  sampleDataFrame = pd.DataFrame(data = sampleVocabMatrix,
                                 index = filenames,
                                 columns = vocabulary)
  sampleDataFrame['label'] = sampleLabels
  print sampleDataFrame.info()

  #get probabilities P(l) for all l in labels
  labelProbabilities = getLabelProbabilities(sampleLabels)
  print '# label probabilities ',labelProbabilities

  labels = labelProbabilities.keys() #ordered set

  #get conditional probabilities P(f|l) for all l in labels and f in features
  featureLabelCondProbabilites = getFeatureLabelCondProbabilites(sampleDataFrame,labels)

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
    raw = ''.join(i for i in raw if ord(i)<128) #clean non-ascii characters
    processed = raw.lower()
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

  return vocabulary, vocabMatrix

def getLabelProbabilities(sampleLabels):
  '''calculates percentage for each label in list of all labels'''
  setOfLabels = set(sampleLabels)
  nSamples = float(len(sampleLabels))
  labelProbabilities = {}
  for l in setOfLabels:
    prob = float(sampleLabels.count(l))/nSamples
    labelProbabilities[l] = prob
  return labelProbabilities

#def getFeatureLabelCondProbabilites(sampleVocabMatrix,sampleLabels,labels):
def getFeatureLabelCondProbabilites(sampleDataFrame,labels):
  '''calculate conditional probabilities of features on labels:
     P(f|l) for all f and l'''
  vocabulary = sampleDataFrame.columns.values.tolist()[:-1] #TODO do the columns stay in order, so that the last one is 'label'?
  featureLabelCondProbabilites = pd.DataFrame(index=labels,columns=vocabulary)
  for label in labels:
    #get samples labelled "label"
    samplesInClass = sampleDataFrame[sampleDataFrame['label'] == label]
    countLabel = float(samplesInClass.shape[0])
    #print 'count(l) ',label,countLabel
    for word in vocabulary:
      #count how many of the samples labelled "label" contain the feature "word"
      countFeature = float(samplesInClass[samplesInClass[word]].shape[0])
      #print 'count(f|l) ',word,label,countFeature
      print 'P(f|l) ',word,label,countFeature/countLabel #TODO add smoothing
      featureLabelCondProbabilites[word][label] = countFeature/countLabel #P(f|l)
  return featureLabelCondProbabilites
 
if __name__ == '__main__':
  main() 
