import io
import sys
import numpy as np
import pandas as pd
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import nltk
from nltk.corpus import stopwords
import string

#Naive Bayes from scratch (no scikit-learn)
# l is member of labels
# f is member of features
# P(l|features) = P(l,features)/P(features)
# P(l,features) = P(l)*Product[P(f|l)]

#reads labelled datafiles
#partitions into training and test sets
#trains Naive Bayes classifier
#uses Laplacian correction (dataset must be big enough!)

# global variables
stopwords = stopwords.words('english')
stopwords += ['.',',',';','?','!','-',':','',"n't","'d","'re","'s","'m"]
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
vocabSize = 20 #no. of words used as features
splitFraction = 0.6 #fraction used for training
laplaceCorr = False #switch on or off Laplacian correction
calcPrior = False #True - calc P(l) from training set, False - assume uniform prior

def main():

  dataDir = '/Users/fogarty/ereader-backup/train'
  labelledDataFile = dataDir + '/list-ebook-files.txt'

  #get names and labels of all pre-labeled ebook files
  filenames, sampleLabels = getDataList(dataDir, labelledDataFile)

  split = int(splitFraction*len(filenames))

  #get vocabulary matrix (dim: nsamples*vocabSize, entries: T or F)
  vocabulary, sampleVocabMatrix = getVocabMatrix(filenames)
  print '# Created vocabulary matrix of dimensions ',sampleVocabMatrix.shape

  sampleDataFrame = pd.DataFrame(data = sampleVocabMatrix,
                                 index = filenames,
                                 columns = vocabulary)
  sampleDataFrame['classLabel'] = sampleLabels

  #get probabilities P(l) for all l in labels
  labelProbabilities = getLabelProbabilities(sampleLabels[:split])
  print '# label probabilities ',labelProbabilities

  #get set of unique labels
  labels = labelProbabilities.keys() 
  print '# using labels :',labels

  #get conditional probabilities P(f|l) for all l in labels and f in features
  featureLabelCondProbabilites = getFeatureLabelCondProbabilites(sampleDataFrame[:split],labels)

  testDataFrame = sampleDataFrame.ix[split:]

  testProbabilities = calcPrediction(testDataFrame,labelProbabilities,featureLabelCondProbabilites)

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
    line = line.rsplit(' ',1)
    filenames.append(rootdir+'/'+line[0])
    labels.append(line[1].strip())
  ff.close()
  return filenames, labels

def getVocabMatrix(filenames):
  '''get vocabulary matrix
     dimensions: len(nsamples)*len(vocabulary)
     contents: 1 or 0'''

  # extract contents of each document
  tokensList = []
  for filename in filenames:
    f = io.open(filename,'r',encoding="latin-1")
    #f = io.open(filename,'r',encoding="utf-8")
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
  if 'classLabel' in vocabulary: #very unlikely
    print 'Error! Feature vocabulary cannot contain the word classLabel'
    quit()
  
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

def getFeatureLabelCondProbabilites(sampleDataFrame,labels):
  '''calculate conditional probabilities of features on labels:
     P(f|l) for all f and l'''
  vocabulary = sampleDataFrame.columns.values.tolist()
  vocabulary.remove('classLabel') #the 'classLabel' column is the class label, not a word in the vocabulary 
  nWords = len(vocabulary)
  print '#Working with vocabulary containing ',nWords,' words'
  if laplaceCorr:
    print '#Using Laplacian correction to avoid 0 probabilities'
  featureLabelCondProbabilites = pd.DataFrame(index=labels,columns=vocabulary)
  for label in labels:
    #get samples labelled "label"
    samplesInClass = sampleDataFrame[sampleDataFrame['classLabel'] == label]
    #count how many samples are labelled "label"
    countLabel = float(samplesInClass.shape[0]) 
    if laplaceCorr: countLabel += nWords #Laplacian correction
    for word in vocabulary:
      #count how many of the samples labelled "label" contain the feature "word"
      countFeature = float(samplesInClass[samplesInClass[word]].shape[0])
      if laplaceCorr: countFeature += 1.0
      print 'P(f|l) ',word,label,countFeature/countLabel
      featureLabelCondProbabilites[word][label] = countFeature/countLabel #P(f|l)
  return featureLabelCondProbabilites

def calcPrediction(testDataFrame,labelProbabilities,featureLabelCondProbabilites):
  for index,sample in testDataFrame.iterrows():
    labelFeatureCondProbabilities = {} #P(l|features)
    for label in labelProbabilities.keys():
      if calcPrior: #use calculated P(l)
        labelFeatureCondProbabilities[label] = labelProbabilities[label]
      else: #otherwise assume uniform prior
        labelFeatureCondProbabilities[label] = 1.0
      for word in featureLabelCondProbabilites.columns.values:
        if sample[word]:
          labelFeatureCondProbabilities[label] *= featureLabelCondProbabilites[word][label]
    print labelFeatureCondProbabilities
    probValues=list(labelFeatureCondProbabilities.values())
    prediction=list(labelFeatureCondProbabilities.keys())[probValues.index(max(probValues))]
    print 'known=',sample['classLabel'],'prediction=',prediction
  #TODO
  return 0
 
if __name__ == '__main__':
  main() 
