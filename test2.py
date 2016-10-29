import numpy as np
import cv2

dataDir = "./data/300W-2/01_Indoor/"
# Load an color image in grayscale
picName = "indoor_001.png"
ptsName = "indoor_001.pts"


img = cv2.imread(dataDir + picName,0)
# print img.shape
file = open(dataDir + ptsName, 'r')
initDataCounter = 0
points = []
for point in file:
	if initDataCounter > 2:
		if "{" not in point and "}" not in point:
			strPoints = point.split(" ")
			points.append((float(strPoints[0]), float(strPoints[1])))

	else:
		initDataCounter += 1
	# print type(point)
	# break

 # cv2.circle(img,(447,63), 10, (0,0,255), -1)

# print file.read()


# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()