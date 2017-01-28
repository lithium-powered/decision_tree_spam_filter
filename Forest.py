import random
import numpy as np
import math
import csv
from scipy import io
import NodeClass
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

maxDepth = 40
treeArray = []
for i in range(0,20):
	treeArray.append(hw5ForestTreeClass.ForestTree(trainingData,maxDepth))

pred = np.zeros(spamValidationLabels.size)
for i in range(0,spamValidationLabels.size):
	for k in range(0,20):
		pred[i] += treeArray[k].classify(spamValidationSamples[i])
	if (pred[i] >= 10):
		pred[i] = 1
	else:
		pred[i] = 0

ValidationErrorWithC = zero_one_loss(spamValidationLabels, pred)
print ValidationErrorWithC

spamTestData = spam['test_data']
testSize = spamTestData.shape[0]
pred = np.zeros(testSize)
for i in range(0,testSize):
	for k in range(0,20):
		pred[i] += treeArray[k].classify(spamTestData[i])
	if (pred[i] >= 10):
		pred[i] = 1
	else:
		pred[i] = 0

spamCSV = csv.writer(open('spamKagglePredictions.csv', 'wt'))
spamCSV.writerow(['Id', 'Category'])
for i in range(0,testSize):
	spamCSV.writerow([i+1,int(pred[i])])


