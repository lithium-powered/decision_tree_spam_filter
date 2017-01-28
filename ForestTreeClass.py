import random
import numpy as np
import math
import NodeClass
import TreeClass

class ForestTree(TreeClass.Tree):

	def getBestAtt(self, trainingData, attributes):
		parentEntropy = self.getEntropy(trainingData)
		maxInfoGained = 0
		curBestAtt = None
		totalNum = float(self.numOfPoints(trainingData))

		randomIndex = random.sample(range(0,len(attributes)), int(math.sqrt(len(attributes))))

		for k in randomIndex:
			i = attributes[k]
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