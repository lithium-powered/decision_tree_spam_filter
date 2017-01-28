import random
import numpy as np
import math
import csv
from scipy import io
import NodeClass
import TreeClass
import ForestTreeClass
from sklearn.metrics import zero_one_loss
from sklearn.preprocessing import normalize

spam = io.loadmat('spam-dataset/spam_data.mat')
spamTrainingData = spam['training_data']
spamTrainingLabel = np.ravel(spam['training_labels'])


spamTrainingSamples = []
spamTrainingLabels = []
spamValidationSamples = []
spamValidationLabels = []
randomIndex = random.sample(range(0,len(spamTrainingData)), len(spamTrainingData))
for i in randomIndex[:4000]:
	spamTrainingSamples.append(spamTrainingData[i])
	spamTrainingLabels.append(spamTrainingLabel[i])
for i in randomIndex[4000:]:
	spamValidationSamples.append(spamTrainingData[i])
	spamValidationLabels.append(spamTrainingLabel[i])
spamTrainingSamples = np.array(spamTrainingSamples)
spamTrainingLabels = np.array(spamTrainingLabels)
spamValidationSamples = np.array(spamValidationSamples)
spamValidationLabels = np.array(spamValidationLabels)

attributes = range(0,spamTrainingData.shape[1])
trainingData = {'training_data':spamTrainingSamples, 'training_labels':spamTrainingLabels}
validationData = {'training_data':spamValidationSamples, 'training_labels':spamValidationLabels}


maxDepth = 30
tree = hw5TreeClass.Tree(trainingData, maxDepth)

'''
pred = []
for i in range(0,spamValidationLabels.size):
	pred.append(tree.classify(spamValidationSamples[i]))

ValidationErrorWithC = zero_one_loss(spamValidationLabels, pred)
print ValidationErrorWithC
'''











'''
def growTree(trainingData, attributes, currentLevel):
	if ((np.unique(trainingData['training_labels']).size == 1) or (currentLevel == maxLevel)):
		return hw5NodeClass.Node(trainingData, None, currentLevel)
	bestAtt = getBestAtt(trainingData, attributes)
	if ((bestAtt == None) or (len(attributes) == 0)):
		return hw5NodeClass.Node(trainingData, None, currentLevel)
	set1, set2 = splitData(trainingData, bestAtt)
	attributes.remove(bestAtt)
	return hw5NodeClass.Node(trainingData, bestAtt, currentLevel, \
		growTree(set1, attributes, currentLevel+1), \
		growTree(set2, attributes, currentLevel+1), False)

def getBestAtt(trainingData, attributes):
	print "start"
	trainingSamples = trainingData['training_data']
	trainingLabels = trainingData['training_labels']
	parentEntropy = getEntropy(trainingData)
	maxInfoGained = 0
	curBestAtt = None
	totalNum = float(trainingLabels.size)

	for i in attributes:
		set1, set2 = splitData(trainingData, i)
		if ((len(getLabels(set1)) == 0) or (len(getLabels(set2)) == 0)):
			None
		else:
			infoGain = parentEntropy - set1['training_labels'].size/totalNum*getEntropy(set1) \
				- set2['training_labels'].size/totalNum*getEntropy(set2)
			if (infoGain > maxInfoGained):
				curBestAtt = i
				maxInfoGained = infoGain
	print "finish"
	return curBestAtt


def dataProb(trainingData):
	trainingLabels = getLabels(trainingData)
	totalNum = float(trainingLabels.size)
	uniqueLabels, labelOccur = np.unique(trainingLabels, return_counts=True)
	dataProb = np.zeros(uniqueLabels.size)
	for i in range(uniqueLabels.size):
		dataProb[i] = labelOccur[i]/totalNum
	return dataProb


def getLabels(trainingData):
		return trainingData['training_labels']

def getEntropy(trainingData):
	trainingLabels = getLabels(trainingData)
	totalNum = float(trainingLabels.size)
	uniqueLabels, labelOccur = np.unique(trainingLabels, return_counts=True)
	if len(uniqueLabels) == 1:
		return 0
	try:
		prob = {uniqueLabels[0]: labelOccur[0]/totalNum, uniqueLabels[1]: labelOccur[1]/totalNum}
	except:
		print trainingData
	return prob[0]*(-math.log(prob[0],2)) + prob[1]*(-math.log(prob[1],2))

def splitData(trainingData, att):
	trainingSamples = trainingData['training_data']
	trainingLabels = trainingData['training_labels']
	set1Samples = []
	set1Labels = []
	set2Samples = []
	set2Labels = []
	for k in range(0,trainingLabels.size):
		if (trainingSamples[k][att] > 0):
			set1Samples.append(trainingSamples[k])
			set1Labels.append(trainingLabels[k])
		else:
			set2Samples.append(trainingSamples[k])
			set2Labels.append(trainingLabels[k])
	set1 = {'training_data':np.array(set1Samples), 'training_labels':np.array(set1Labels)}
	set2 = {'training_data':np.array(set2Samples), 'training_labels':np.array(set2Labels)}
	return (set1, set2)

def classify(rootNode, dataPoint):
	node = rootNode
	while (node.leaf == False):
		node = node.nextNode(dataPoint)
	return node.classify()
'''


