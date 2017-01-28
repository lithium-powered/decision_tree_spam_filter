import random
import numpy as np
import math
import NodeClass

class Tree:
	def __init__(self, trainingData, maxDepth):
		self.maxDepth = maxDepth
		self.rootNode = self.growTree(trainingData, range(0,self.numOfFeatures(trainingData)), 1)

	def growTree(self, trainingData, attributes, currentLevel):
		if ((self.numOfClasses(trainingData) == 1) or (currentLevel == self.maxDepth) or (len(attributes) == 0)):
			return hw5NodeClass.Node(trainingData, None, currentLevel)
		bestAtt = self.getBestAtt(trainingData, attributes)
		if (bestAtt == None):
			return hw5NodeClass.Node(trainingData, None, currentLevel)
		set1, set2 = self.splitData(trainingData, bestAtt)
		attributes.remove(bestAtt)
		return hw5NodeClass.Node(trainingData, bestAtt, currentLevel, \
			self.growTree(set1, attributes, currentLevel+1), \
			self.growTree(set2, attributes, currentLevel+1), False)

	def getBestAtt(self, trainingData, attributes):
		parentEntropy = self.getEntropy(trainingData)
		maxInfoGained = 0
		curBestAtt = None
		totalNum = float(self.numOfPoints(trainingData))
		for i in attributes:
			set1, set2 = self.splitData(trainingData, i)
			set1Size = self.numOfPoints(set1)
			set2Size = self.numOfPoints(set2)
			if ((set1Size == 0) or (set2Size == 0)):
				continue
			infoGain = parentEntropy - set1Size/totalNum*self.getEntropy(set1) \
				- set2Size/totalNum*self.getEntropy(set2)
			if (infoGain > maxInfoGained):
				curBestAtt = i
				maxInfoGained = infoGain
		return curBestAtt

	def getEntropy(self, trainingData):
		trainingLabels = self.getLabels(trainingData)
		totalNum = float(self.numOfPoints(trainingData))
		uniqueLabels, labelOccur = np.unique(trainingLabels, return_counts=True)
		if len(uniqueLabels) == 1:
			return 0
		prob = {uniqueLabels[0]: labelOccur[0]/totalNum, uniqueLabels[1]: labelOccur[1]/totalNum}
		return prob[0]*(-math.log(prob[0],2)) + prob[1]*(-math.log(prob[1],2))

	def splitData(self, trainingData, att):
		trainingSamples = self.getSamples(trainingData)
		trainingLabels = self.getLabels(trainingData)
		set1Samples = []
		set1Labels = []
		set2Samples = []
		set2Labels = []
		for k in range(0,self.numOfPoints(trainingData)):
			if (trainingSamples[k][att] > 0):
				set1Samples.append(trainingSamples[k])
				set1Labels.append(trainingLabels[k])
			else:
				set2Samples.append(trainingSamples[k])
				set2Labels.append(trainingLabels[k])
		set1 = {'training_data':np.array(set1Samples), 'training_labels':np.array(set1Labels)}
		set2 = {'training_data':np.array(set2Samples), 'training_labels':np.array(set2Labels)}
		return (set1, set2)

	def classify(self, dataPoint, trainingData):
		node = self.rootNode
		while (node.leaf == False):
			node = node.nextNode(dataPoint)
		return node.classify()

	def getLabels(self, trainingData):
		return trainingData['training_labels']

	def getSamples(self, trainingData):
		return trainingData['training_data']

	def numOfClasses(self, trainingData):
		trainingLabels = self.getLabels(trainingData)
		return np.unique(trainingData['training_labels']).size

	def numOfPoints(self, trainingData):
		return self.getLabels(trainingData).size

	def numOfFeatures(self, trainingData):
		trainingSamples = self.getSamples(trainingData)
		return trainingSamples[0].size

