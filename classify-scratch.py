import io
import sys
import numpy as np
import pandas as pd
sys.path.insert(1,'/Library/Python/2.7/site-packages')
import nltk
import string
import commonfns

#Naive Bayes from scratch (no scikit-learn)
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

  #get vocabulary matrix (dim: nsamples*vocabSize, entries: T or F)
  vocabulary, sampleVocabMatrix = commonfns.getVocabMatrix(filenames,vocabSize)
  print '# Created vocabulary matrix of dimensions ',sampleVocabMatrix.shape

  sampleDataFrame = pd.DataFrame(data = sampleVocabMatrix,
                                 index = filenames,
                                 columns = vocabulary)
  sampleDataFrame['classLabel'] = sampleLabels

  #split into training and test sets 
  trainDataFrame, testDataFrame = splitTrainTest(sampleDataFrame, splitFraction)

  #get probabilities P(l) for all l in labels
  labelProbabilities = commonfns.getLabelProbabilities(trainDataFrame['classLabel'].tolist())
  print '# label probabilities ',labelProbabilities

  #get set of unique labels
  labels = labelProbabilities.keys() 
  print '# using labels :',labels

  #get conditional probabilities P(f|l) for all l in labels and f in features
  featureLabelCondProbabilites = getFeatureLabelCondProbabilites(trainDataFrame,labels,laplaceCorr)

  testProbabilities = calcPrediction(testDataFrame,labelProbabilities,featureLabelCondProbabilites,calcPrior,logProb)

def getFeatureLabelCondProbabilites(sampleDataFrame,labels,laplaceCorr):
  '''calculate conditional probabilities of features on labels:
     P(f|l) for all f and l'''

  vocabulary = sampleDataFrame.columns.values.tolist()
  vocabulary.remove('classLabel') #the 'classLabel' column is the class label, not a word in the vocabulary 
  nWords = len(vocabulary)
  print '#Working with vocabulary containing ',nWords,' words'
  print vocabulary
  if laplaceCorr:
    print '#Using Laplacian correction to avoid zero probabilities'
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

def calcPrediction(testDataFrame,labelProbabilities,featureLabelCondProbabilites,calcPrior,logProb):
  '''for each sample in testDataFrame, predict label as argmax( P(l)*Product_F[P(f|l)] ) if global control variable logProb is True, or argmax( log(P(l)) + Sum_F[log(P(f|l))] ) if logProb is False; calculate accuracy'''

  accuracy = 0.0

  #loop over test samples
  for index,sample in testDataFrame.iterrows():

    labelFeatureCondProbabilities = {} #P(l|features)

    #loop over labels
    for label in labelProbabilities.keys():

      #set P(l)
      if calcPrior: #use calculated P(l)
        labelFeatureCondProbabilities[label] = labelProbabilities[label]
      else: #otherwise assume uniform prior
        labelFeatureCondProbabilities[label] = 1.0
      if logProb: #convert to log(P(l))
        labelFeatureCondProbabilities[label] = np.log(labelFeatureCondProbabilities[label])

      #loop over words in vocabulary
      for word in featureLabelCondProbabilites.columns.values:
        if sample[word]:
          if logProb: #calc log(P(l)) + Sum_F[log(P(f|l))]
            labelFeatureCondProbabilities[label] += np.log(featureLabelCondProbabilites[word][label])
          else: #calc P(l)*Product_F[P(f|l)]
            labelFeatureCondProbabilities[label] *= featureLabelCondProbabilites[word][label]

    print labelFeatureCondProbabilities
    probValues=list(labelFeatureCondProbabilities.values())
    prediction=list(labelFeatureCondProbabilities.keys())[probValues.index(max(probValues))]
    print 'filename: ',index
    print 'known=',sample['classLabel'],'prediction=',prediction
    if sample['classLabel'] == prediction: accuracy += 1.0

  accuracy /= float(len(testDataFrame))
  print 'accuracy: ',accuracy*100,'%'

  return 0

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
 
if __name__ == '__main__':
  main() 
