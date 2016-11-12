import numpy as np
import cv2

dataDir = "./data/300W/01_Indoor/"
# Load an color image in grayscale
picName = "indoor_049.png"
ptsName = "indoor_049.pts"


img = cv2.imread(dataDir + picName,1)

(ymax, xmax, _) = img.shape

file = open(dataDir + ptsName, 'r')
initDataCounter = 0
X = []
Y = []

for point in file:
    if initDataCounter > 2:
        if "{" not in point and "}" not in point:
            strPoints = point.split(" ")
            x = int(float(strPoints[0]))
            y = int(float(strPoints[1]))
            X.append(x)
            Y.append(y)

            cv2.circle(img,(x, y), 4, (0,0,255), -1)
    else:
        initDataCounter += 1

padding = 50
Xmin = min(X) - padding
Ymin = min(Y) - padding
Xmax = max(X) + padding
Ymax = max(Y) + padding

if Xmin < 0:
    Xmin = 0
if Ymin < 0:
    Ymin = 0
if Xmax > xmax:
    Xmax = xmax
if Ymax > ymax:
    Ymax = ymax


cv2.rectangle(img,(Xmin, Ymin),(Xmax,Ymax),(0,255,0),3)
cv2.imshow('image',img)

cropImg = img[Ymin:Ymax, Xmin:Xmax] # Crop from (Xmin, Ymin),(Xmax,Ymax)
resizedImage = cv2.resize(cropImg, (64, 64))

cv2.imshow("resized", resizedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()