import numpy as np 
import math

def getAcurracy(test_labels, predictions):
    error = []
    (length, _) = predictions.shape
    for index in range(length):

        pred = predictions[index]
        x, y =parseLocation(pred)
        predPts = zip(x, y)

        label = test_labels[index]
        x, y =parseLocation(label)
        labelPts = zip(x, y)   

        interocular_distance = math.sqrt((labelPts[37][0] - labelPts[46][0])**2 + (labelPts[37][1] - labelPts[46][1])**2)


        cum = 0
        for i in range(68):
            cum = cum + math.sqrt((predPts[i][0] - labelPts[i][0])**2 + (predPts[i][1] - labelPts[i][1])**2);

        error.append(cum/(68*interocular_distance))

    error = numpy.asarray(error)
    correct = numpy.mean(error)

    # error = numpy.mean(numpy.absolute(test_labels - predictions))
    # correct = 1.0 - error
    
    # correct = test_labels[:, 0] == predictions[:, 0]

    return correct 


def parseLocation(array):
    x = []
    y = []
    for i in range(0, len(array), 2):
        x.append(array[i])
        y.append(array[i + 1])
    return x, y



pred = np.random.int()
label = [3.0, 4.0] * 68

newPred = []
newLabel = []
newPred.append(pred)
newLabel.append(label)

newPred = np.asarray(newPred)
newLabel = np.asarray(newLabel)

print getAcurracy(newLabel, newPred)