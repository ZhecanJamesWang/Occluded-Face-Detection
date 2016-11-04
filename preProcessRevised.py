import numpy as np
import cv2
import os
from PIL import Image, ImageChops
import pickle
from sklearn.cross_validation import train_test_split

class preProcess(object):
    def __init__(self):
        self.name = "300W"
        self.rawDir = "./data/" +self.name + "/"
        self.resizedDir = "./data/" + self.name + "Resized/"        
        self.pFileDir = "./data/pFile/"      
        self.trainTestDir = "./data/trainTestData/"

        self.padding = 50
        self.format = ".jpg"
        self.size = (50, 50)

    def readData(self):
        counter = 0
        folders = os.listdir(self.rawDir)
        # files = os.listdir(self.rawDir)

        for fold in folders:
            if fold != ".DS_Store":
                path = os.path.abspath(self.rawDir + fold)
                files = os.listdir(path)

                for file in files:
                    if file != ".DS_Store" and self.format in file:
                        img, pts= self.process(path + "/", file)
                        # img, pts= self.process(self.rawDir + "/", file)
                        counter += 1
                        self.saveImag(self.resizedDir, file[:-4] + "_resized.png", img)
                        pickle.dump( pts, open( self.resizedDir + file[:-4] + ".p", "wb" ) )

                        if counter % 100 == 0:
                            print counter
                            # print path

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
            else:
                initDataCounter += 1
        if separate:
            return X, Y
        else:
            return pts
    def plotPTS(self, X, Y, ymax, xmax, img):
        assert len(X) == len(Y)      
        for index in range(len(X)):
            print (int(X[index]*xmax), int(Y[index]*ymax))
            cv2.circle(img,(int(X[index]*xmax), int(Y[index]*ymax)), 1, (0,0,255), -1)

        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process(self, dataDir, picName):
        ptsName = picName[:-4] + ".pts"
        img = cv2.imread(dataDir + picName,1)
        (ymax, xmax, _) = img.shape

        X, Y = self.extractPTS(dataDir, ptsName)
  
        X = [x / float(xmax) for x in X]
        Y = [y / float(ymax) for y in Y]

        resizedImage = self.resize(img)
        (ymax, xmax, _) = resizedImage.shape

        # self.plotPTS(X, Y, ymax, xmax, resizedImage)
        pts = zip(X, Y)
        # resizedImage = cv2.resize(cropImg, (50, 50))
        # cv2.rectangle(img,(Xmin, Ymin),(Xmax,Ymax),(0,255,0),3)

        # cv2.imshow("cropped", cropImg)
        # cv2.imshow('image',img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return resizedImage, pts
    
    def generateData(self):
        x = []
        y = []
        Files = os.listdir(self.resizedDir)

        for i in range(len(Files)):
            image = Files[i]
            while image == ".DS_Store" or image[-2:] == ".p":
                i += 1
                image = Files[i]

            pts = image[:-12] + ".p"
            img = cv2.imread(self.resizedDir + image,0)
            pts = pickle.load( open( self.resizedDir + pts, "rb" ) )

            x.append(img)
            y.append(pts)
            
            i += 1

        x = np.asarray(x)
        y = np.asarray(y)

        print x.shape
        print y.shape

        print type(x)
        print type(y)


        pickle.dump( x, open( self.pFileDir + self.name + "_x.p", "wb" ) )
        pickle.dump( y, open( self.pFileDir + self.name + "_y.p", "wb" ) )

    def splitData(self):
        files = os.listdir(self.pFileDir)
        X = []
        Y = []

        for file in files:
            if file != ".DS_Store":
                if file[-3:-2] == "x":
                    x = pickle.load( open( self.pFileDir + file, "rb" ) )
                    print x.shape
                    X.extend(x)
                elif file[-3:-2] == "y":
                    y = pickle.load( open( self.pFileDir + file, "rb" ) )
                    print y.shape
                    Y.extend(y)

        X = np.asarray(X)
        Y = np.asarray(Y)

        print X.shape
        print Y.shape

        print type(X)
        print type(Y)
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)
        pickle.dump( xTrain, open( self.trainTestDir + "xTrain.p", "wb" ) )
        pickle.dump( xTest, open( self.trainTestDir + "xTest.p", "wb" ) )
        pickle.dump( yTrain, open( self.trainTestDir + "yTrain.p", "wb" ) )
        pickle.dump( yTrain, open( self.trainTestDir + "yTest.p", "wb" ) )

    def run(self):
        # self.readData()
        # self.generateData()
        # self.splitData()


if __name__ == '__main__':
    preProcess().run()