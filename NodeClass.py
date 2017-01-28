import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import random
from scipy import io

class Node:
	def __init__(self, trainingData, split, level, leftChild=None, rightChild=None, leaf=True):
		self.leftChild = leftChild
		self.rightChild = rightChild
		self.trainingLabels = self.getLabels(trainingData)
		self.level = level
		self.leaf = leaf
		self.split = split

	def classify(self):
		if not (self.leaf):
			raise "Trying to classify data from a none leaf node."
		total = 0
		for i in self.trainingLabels:
			total += i
		if total >= self.trainingLabels.size/2.0:
			return 1
		else:
			return 0

	def getLabels(self, trainingData):
		return trainingData['training_labels']

	def notLeaf(self):
		self.leaf = False

	def nextNode(self, dataPoint):
		if (dataPoint[self.split] > 0):
			return self.leftChild
		else:
			return self.rightChild
'''
	def getEntropy(trainingData):
		prob = dataProb(trainingData)
		entropy = 0
		for i in range(prob.size):
			entropy += prob[i]*(-math.log(prob[i],2))
		return entropy

	def dataProb(trainingData):
		trainingLabels = self.getLabels(trainingData)
		totalNum = float(trainingLabels.size)
		uniqueLabels, _, _, labelOccur = np.unique(trainingLabels, return_counts=True)
		dataProb = np.zeros(uniqueLabels.size)
		for i in range(uniqueLabels.size):
			dataProb[i] = labelOccur[i]/totalNum
		return dataProb
'''
