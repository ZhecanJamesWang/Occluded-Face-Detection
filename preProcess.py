import numpy as np
import cv2
import os


class preProcess(object):
    def __init__(self):
        self.rawDir = "./data/300W/"
        self.croppedDir = "./data/300W_cropped/"
        # Load an color image in grayscale
        # picName = "indoor_002.png"
        # ptsName = "indoor_002.pts" 
        self.padding = 20

    def readImg(self):
        counter = 0
        folders = os.listdir(self.rawDir)
        for fold in folders:
            if fold != ".DS_Store":
                path = os.path.abspath(self.rawDir + fold)
                files = os.listdir(path)
                for file in files:
                    if file != ".DS_Store" and ".png" in file:
                        try:
                            img = self.crop(path + "/", file)
                            counter += 1
                        except Exception as e:
                            print e
                            print file
                            raise "debug"
                        # img = crop(self.rawDir + fold + "/", file)
                        self.saveImag(self.croppedDir, file[:-4] + "_cropped.png", img)
                        if counter % 100 == 0:
                            print counter
                            print path

    def saveImag(self, dataDir, fileName, file):
         cv2.imwrite(dataDir + fileName,file)


    def crop(self, dataDir, picName):
        ptsName = picName[:-4] + ".pts"
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
                    # cv2.circle(img,(x, y), 4, (0,0,255), -1)
            else:
                initDataCounter += 1
        Xmin = min(X) - self.padding
        Ymin = min(Y) - self.padding
        Xmax = max(X) + self.padding
        Ymax = max(Y) + self.padding

        if Xmin < 0:
            Xmin = 0
        if Ymin < 0:
            Ymin = 0
        if Xmax > xmax:
            Xmax = xmax
        if Ymax > ymax:
            Ymax = ymax

        cropImg = img[Ymin:Ymax, Xmin:Xmax] # Crop from (Xmin, Ymin),(Xmax,Ymax)
        resizedImage = cv2.resize(cropImg, (50, 50))
        # cv2.rectangle(img,(Xmin, Ymin),(Xmax,Ymax),(0,255,0),3)

        # cv2.imshow("cropped", cropImg)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resizedImage

    def run(self):
        self.readImg()


if __name__ == '__main__':
    preProcess().run()