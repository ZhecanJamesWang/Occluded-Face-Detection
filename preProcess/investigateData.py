

import pickle
import cv2
import numpy as np

def parseLocation(array):
    x = []
    y = []
    for i in range(0, len(array), 2):
        x.append(array[i])
        y.append(array[i + 1])
    return x, y

def plotLandmarks(img, X, Y):
    # plot landmarks on original image
    assert len(X) == len(Y)      
    for index in range(len(X)):
        cv2.circle(img,(int(X[index]*50), int(Y[index]*50)), 1, (0,0,255), -1)

    cv2.imshow("image3",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

trainTestDir = "./data/trainTestData/"


# xTrain = pickle.load( open( trainTestDir + "xTrainFlatten.p", "rb" ) )[:2000]
# yTrain = pickle.load( open( trainTestDir + "yTrain.p", "rb" ) )[:2000]



# xTest = pickle.load( open( trainTestDir + "xTestFlatten.p", "rb" ) )[:2000]
# yTest = pickle.load( open( trainTestDir + "yTest.p", "rb" ) )[:2000]

# pred = pickle.load( open( trainTestDir + "supervisedTrainPred.p", "rb" ) )[:2000]


# print pred.shape
# print yTest.shape
# print xTest.shape

output = pickle.load( open( trainTestDir + "firstAEDoutput.p", "rb" ) )[:2000]
xTest = output

print len(xTest)

for i in range(0, len(xTest)):
    img = xTest[i]
    img = img.reshape((50, 50))
    cv2.imshow("image1",img)
    cv2.waitKey(0)

    # newImg1 = np.zeros((2500))
    # for j in range(len(newImg1)):
    #     newImg1[j] = 255
    # newImg1 = newImg1.reshape((50, 50))

    # newImg2 = np.zeros((2500))
    # for j in range(len(newImg2)):
    #     newImg2[j] = 255
    # newImg2 = newImg2.reshape((50, 50))

    # print i
    # predArray = pred[i]
    # labelArray = yTest[i]
    # predX, predY = parseLocation(predArray)
    # labelX, labelY = parseLocation(labelArray)

    # for index in range(len(labelX)):
    #     cv2.circle(newImg1, (int(labelX[index]*50), int(labelY[index]*50)), 1, (0,0,255), -1)
    # cv2.imshow("image2",newImg1)

    # plotLandmarks(newImg2, predX, predY)



