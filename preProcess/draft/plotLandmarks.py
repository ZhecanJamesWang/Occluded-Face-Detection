
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
def plotLandmarks(img, X, Y, ymax, xmax, ifRescale = False, name = "image", onhold = False):
    # plot landmarks on original image
    assert len(X) == len(Y)      
    for index in range(len(X)):
        if ifRescale:
            cv2.circle(img,(int(X[index]*xmax), int(Y[index]*ymax)), 1, (0,0,255), -1)
        else:
            cv2.circle(img,(int(X[index]), int(Y[index])), 1, (0,0,255), -1)

    if onhold:
        cv2.imshow(name,img)

    else:
        cv2.imshow(name,img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

trainTestDir = "./data/trainTestData/"

pred = pickle.load( open( trainTestDir + "supervisedTrainPred.p", "rb" ) )

xTrain = pickle.load( open( trainTestDir + "xTrainFlatten.p", "rb" ) )[:2000]
yTrain = pickle.load( open( trainTestDir + "yTrain.p", "rb" ) )[:2000]
xTest = pickle.load( open( trainTestDir + "xTestFlatten.p", "rb" ) )[:2000]
yTest = pickle.load( open( trainTestDir + "yTest.p", "rb" ) )[:2000]


index = 100

img = xTest[index]
predArray = pred[index]
labelArray = yTest[index]


newImg = np.zeros(img.shape)
for i in range(len(newImg)):
    newImg[i] = 255

img = img.reshape((50, 50))
newImg = newImg.reshape((50, 50))

cv2.imshow('image1',img)



predX, predY = parseLocation(predArray)
labelX, labelY = parseLocation(labelArray)

plotLandmarks(newImg, labelX, labelY, 50, 50, ifRescale = True, name = "image3", onhold = True)
plotLandmarks(newImg, predX, predY, 50, 50, ifRescale = True, name = "image2")
