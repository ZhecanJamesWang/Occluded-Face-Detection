import os
import cv2

name = "lfpw"
tag = "Preprocessed"
resizedDir = "./data/" + name + tag + "/"    
files = os.listdir(resizedDir)
img = cv2.imread(resizedDir + files[4])

print img.shape
# print img

# img = img * 255
# cv2.imshow("img", img)
# cv2.waitKey(0)