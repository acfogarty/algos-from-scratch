import io
import sys
import numpy as np
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import nltk
from nltk.corpus import stopwords
import string

# global variables
stopwords = stopwords.words('english')
stopwords += ['.',',',';','?','!','-',':','',"n't","'d","'re","'s","'m","''",'``','...','--',"'","'ll","'ve"]
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

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

def getFeatures(words,vocabulary):
  '''check which elements of the vocabulary are in the list words'''
  wordSet = set(words) #for speed
  features = [] #ordered list of T or F
  for word in vocabulary:
    features.append(word in wordSet)
  return features

def getVocabMatrix(filenames,vocabSize):
  '''get vocabulary list and vocabulary matrix
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

def getVocabulary(filenames,vocabSize):
  '''get vocabulary list consisting of
  vocabSize most common words across all
  the files listed in list filenames'''

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
  
  #get most common words in the entire set of documents
  allFd = nltk.FreqDist([y for x in tokensList for y in x]) #flatten
  vocabulary = [w for w,n in allFd.most_common(vocabSize)]

  return vocabulary

def getLabelProbabilities(sampleLabels):
  '''calculates percentage for each label in list of all labels'''
  setOfLabels = set(sampleLabels)
  nSamples = float(len(sampleLabels))
  labelProbabilities = {}
  for l in setOfLabels:
    prob = float(sampleLabels.count(l))/nSamples
    labelProbabilities[l] = prob
  return labelProbabilities

def splitTrainTest(sampleDataFrame, splitFraction):
  '''split DataFrame into two parts of size splitFraction*total and (1-splitFraction)*total'''

  #assign random no. between 0 and 1 to each sample
  randnum = np.random.random_sample(len(sampleDataFrame))
  sampleDataFrame['randnum'] = randnum

  #split into training set (splitFraction of data) and test set (1 - splitFraction of data)
  trainDataFrame = sampleDataFrame[sampleDataFrame['randnum'] < splitFraction]
  testDataFrame = sampleDataFrame[sampleDataFrame['randnum'] >= splitFraction]

  del trainDataFrame['randnum']
  del testDataFrame['randnum']
 
  print 'Split sample data into training set of size ', len(trainDataFrame), ' and test set of size ', len(testDataFrame)

  #TODO confirm above approach is faster than iterrows and append
  #for index,row in sampleDataFrame.iterrows(): #TODO slow!
  #  rnum = np.random.rand()
  #  if rnum < splitFraction:
  #    trainDataFrame.append(...)
  #  else:
  #    testDataFrame.append(...)

  return trainDataFrame, testDataFrame

def fixedSplitTrainTest(sampleDataFrame, splitFraction):
  '''split DataFrame into two parts of size splitFraction*total and (1-splitFraction)*total, no random shuffle (for debugging)'''

  #split into training set (splitFraction of data) and test set (1 - splitFraction of data)
  nTrain = int(splitFraction*len(sampleDataFrame))
  trainDataFrame = sampleDataFrame.ix[:nTrain]
  testDataFrame = sampleDataFrame.ix[nTrain:]

  print 'Split sample data into training set of size ', len(trainDataFrame), ' and test set of size ', len(testDataFrame)

  return trainDataFrame, testDataFrame
