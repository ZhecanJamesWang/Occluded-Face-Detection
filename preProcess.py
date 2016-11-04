import numpy as np
import cv2
import os
from PIL import Image, ImageChops
import pickle


class preProcess(object):
    def __init__(self):
        # self.rawDir = "./data/300W/"
        self.name = "afw"
        self.croppedDir = "./data/" + self.name + "Cropped/"
        self.ptsDir = "./data/" + self.name + "PTS/"

        # self.rawDir = "./data/afw/"
        # self.croppedDir = "./data/afwCropped/"
        # self.rawDir = "./data/afw/"
        # self.croppedDir = "./data/afwCropped/"

        self.padding = 50
        self.format = ".jpg"
        self.size = (50, 50)
    def readImg(self):
        counter = 0
        # folders = os.listdir(self.rawDir)
        files = os.listdir(self.rawDir)

        for fold in folders:
            if fold != ".DS_Store":
                path = os.path.abspath(self.rawDir + fold)
                files = os.listdir(path)
                for file in files:
                    if file != ".DS_Store" and self.format in file:
                        img = self.crop(path + "/", file)
                        # img = self.crop(self.rawDir + "/", file)
                        counter += 1
                        self.saveImag(self.croppedDir, file[:-4] + "_cropped.png", img)
                        if counter % 100 == 0:
                            print counter
                            print path

    def saveImag(self, dataDir, fileName, file):
         cv2.imwrite(dataDir + fileName,file)

    def resize(self, image):

        image = Image.fromarray(np.uint8(image))
        image.thumbnail(self.size, Image.ANTIALIAS)
        image_size = image.size

        thumb = image.crop( (0, 0, self.size[0], self.size[1]) )

        offset_x = max( (self.size[0] - image_size[0]) / 2, 0 )
        offset_y = max( (self.size[1] - image_size[1]) / 2, 0 )

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
        image = np.asarray(thumb)
        return image

    def extractPTS(self, dataDir, ptsName, separate = True):
        file = open(dataDir + ptsName, 'r')
        initDataCounter = 0
        if separate:
            X = []
            Y = []
        else:
            pts = []

        for point in file:
            if initDataCounter > 2:
                if "{" not in point and "}" not in point:
                    strPoints = point.split(" ")
                    x = int(float(strPoints[0]))
                    y = int(float(strPoints[1]))
                    if separate:
                        X.append(x)
                        Y.append(y)
                    else:
                        pts.append((x, y))
                    # cv2.circle(img,(x, y), 4, (0,0,255), -1)
            else:
                initDataCounter += 1
        if separate:
            return X, Y
        else:
            return pts

    def crop(self, dataDir, picName):
        ptsName = picName[:-4] + ".pts"
        img = cv2.imread(dataDir + picName,1)
        (ymax, xmax, _) = img.shape

        X, Y = self.extractPTS(dataDir, ptsName)
                
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
        resizedImage = self.resize(cropImg)
        # resizedImage = cv2.resize(cropImg, (50, 50))
        # cv2.rectangle(img,(Xmin, Ymin),(Xmax,Ymax),(0,255,0),3)

        # cv2.imshow("cropped", cropImg)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resizedImage
    
    def generateData(self):
        x = []
        y = []
        cropFiles = os.listdir(self.croppedDir)
        ptsFiles = os.listdir(self.ptsDir)

        assert len(cropFiles) == len(ptsFiles)

        for i in range(len(cropFiles)):
            image = cropFiles[i]
            while image == ".DS_Store":
                i += 1
                image = cropFiles[i]
            pts = image[:-12] + ".pts"

            img = cv2.imread(self.croppedDir + image,0)
            pts = self.extractPTS(self.ptsDir, pts, separate = False)


            x.append(img)
            y.append(pts)
            
            i += 1

        x = np.asarray(x)
        y = np.asarray(y)

        print x.shape
        print y.shape

        print type(x)
        print type(y)


        pickle.dump( x, open( self.name + "_x.p", "wb" ) )
        pickle.dump( y, open( self.name + "_y.p", "wb" ) )


    def run(self):
        # self.readImg()
        self.generateData()


if __name__ == '__main__':
    preProcess().run()