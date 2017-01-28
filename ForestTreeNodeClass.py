import random
import numpy as np
import math
import csv
from scipy import io
import NodeClass
import TreeClass

class forestTreeNode(TreeClass.Tree):

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
