import numpy as np
import cv2
import os
from PIL import Image, ImageChops
import pickle
from sklearn.cross_validation import train_test_split
import utility as ut

class preProcess(object):
    def __init__(self):
        self.name = "lfpw"
        self.tag = "Preprocessed"
        self.rawDir = "./data/" +self.name + "/"
        self.preProcessedDir = "./data/" + self.name + self.tag + "/"        
        self.pFileDir = "./data/pFile/"      
        self.trainTestDir = "./data/trainTestData/"

        self.padding = 50
        self.format = ".png"
        self.size = (50, 50)
        self.debug = True

    def getData(self):
        # This function initialize reading image and landmark location data

        counter = 0
        folders = os.listdir(self.rawDir)
        # files = os.listdir(self.rawDir)

        for fold in folders:
            if fold != ".DS_Store":
                path = os.path.abspath(self.rawDir + fold)
                files = os.listdir(path)

                for file in files:
                    if file != ".DS_Store" and self.format in file:
                        imgs, landmarks = self.extract(path + "/", file)
                        # resizedImage, normalizedImage, pts = self.extract(self.rawDir + "/", file)
                        self.saveImag(self, imgs, landmarks, file)
                        counter += 1

                        if counter % 100 == 0:
                            print counter
                            # print path



    def saveImag(self, imgs, landmarks, fileName):
        # save image to directory
         # cv2.imwrite(dataDir + fileName,file)

        for index in range(len(imgs)):
            img = imgs[index]
            landmark = landmarks[index]
            pickle.dump( img, open( self.preProcessedDir + fileName[:-4] + "Normalized" + str(index) + ".p", "wb" ) )
            pickle.dump( pts, open( self.preProcessedDir + fileName[:-4] + "Landmarks" + str(index) + ".p", "wb" ) )            
            # pickle.dump( normalizedImage, open( self.preProcessedDir + file[:-4] + "Normalized" + ".p", "wb" ) )
            # pickle.dump( resizedImage, open( self.preProcessedDir + file[:-4] + "Resized" + ".p", "wb" ) )

    def crop(self, img, X, Y):
        (yMaxBound, xMaxBound, _) = img.shape
                
        xMin = min(X) - self.padding
        yMin = min(Y) - self.padding
        xMax = max(X) + self.padding
        yMax = max(Y) + self.padding

        if xMin < 0:
            xMin = 0
        if yMin < 0:
            yMin = 0
        if xMax > xMaxBound:
            xMax = xMaxBound
        if yMax > yMaxBound:
            yMax = yMaxBound


        newX = [x - xMin for x in X]
        newY = [y - yMin for y in Y]


        croppedImage = img[int(yMin):int(yMax), int(xMin):int(xMax)] # Crop from (Xmin, Ymin),(Xmax,Ymax)        

        if self.debug:
            cv2.rectangle(img,(int(xMin), int(yMin)),(int(xMax),int(yMax)),(0,255,0),3)

        return croppedImage, newX, newY

    def normalize(self, img):
        img = img/float(255)
        return img
    
    def getDerivatives(self, originalImg, X, Y):
        # img = np.asarray(originalImg)
        imgs = []
        Xs, Ys = [], []

        img = originalImg.copy()

        mirImage, newX, newY = ut.mirrorImage(img, X, Y)
        mirImage = mirImage.copy()
        imgs.append(mirImage)
        Xs.append(newX)
        Ys.append(newY)

        if self.debug:
            self.plotLandmarks(mirImage, newX, newY, "mirror")

        img = originalImg.copy()
        scaleImage, newX, newY = ut.resize(img, X, Y, random = True)
        imgs.append(mirImage)
        Xs.append(newX)
        Ys.append(newY)

        if self.debug:
            self.plotLandmarks(scaleImage, newX, newY, "scale")

        img = originalImg.copy()
        rotateImage, newX, newY = ut.rotate(img, X, Y)
        imgs.append(rotateImage)
        Xs.append(newX)
        Ys.append(newY)

        if self.debug:
            self.plotLandmarks(rotateImage, newX, newY, "rotate")

        img = originalImg.copy()
        cbImage, newX, newY = ut.contrastBrightess(img, X, Y)
        imgs.append(cbImage)
        Xs.append(newX)
        Ys.append(newY)
        if self.debug: 
            self.plotLandmarks(cbImage, newX, newY, "contrastBrightness")

        img = originalImg.copy()
        transImage, newX, newY = ut.translateImage(img, X, Y)
        imgs.append(transImage)
        Xs.append(newX)
        Ys.append(newY)
        if self.debug:  
            self.plotLandmarks(transImage, newX, newY, "trans")

        return imgs, Xs, Ys

    def getOrigin(self, img, X, Y):
        croppedImage, X, Y = self.crop(img, X, Y) 
        
        if self.debug:
            self.plotLandmarks(croppedImage, X, Y, "cropped")

        resizedImage, X, Y = ut.resize(croppedImage, X, Y)

        if self.debug:
            self.plotLandmarks(resizedImage, X, Y, "resized")
        return resizedImage, X, Y

    def extract(self, dataDir, picName):
        # extract both image and landmark location data
        # return image and its landmarks location data
        ptsName = picName[:-4] + ".pts"
        img = cv2.imread(dataDir + picName,1)
        (ymax, xmax, _) = img.shape
        X, Y = self.parseLandmark(dataDir, ptsName)


        resizeImage, X, Y = self.getOrigin(img, X, Y)
        images, Xs, Ys = self.getDerivatives(resizeImage, X, Y)
        images.append(resizeImage)
        Xs.append(X)
        Ys.append(Y)

        if self.debug:
            cv2.waitKey(1000)
            # cv2.destroyAllWindows()

        filterImages = []
        landmarks = []
        for index in range(len(Xs)):
            image = images[index]
            x = Xs[index]
            y = Ys[index]
            grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)     
            normalizedImage = self.normalize(grayImage)            
            filterImages.append(normalizedImage)
            landmarks.append(self.packLandmarks(x, y))
        filterImages, landmarks = np.asarray(filterImages), np.asarray(landmarks)
        print filterImages.shape
        print landmarks.shape
        return filterImages, landmarks

    def packLandmarks(self, X, Y):
        X = [x / float(self.size[0]) for x in X]
        Y = [y / float(self.size[0]) for y in Y]

        pts = zip(X, Y)
        landmarks = []
        for p in pts:
            landmarks.extend(list(p))
        return landmarks        

    def parseLandmark(self, dataDir, ptsName, separate = True):
        # parse and extract landmark location data from point files

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

    def plotLandmarks(self, img, X, Y, name, ymax = None, xmax = None, ifRescale =False):
        # plot landmarks on original image
        assert len(X) == len(Y)      
        for index in range(len(X)):
            if ifRescale:
                cv2.circle(img,(int(X[index]*xmax), int(Y[index]*ymax)), 1, (0,0,255), -1)
            else:
                cv2.circle(img,(int(X[index]), int(Y[index])), 1, (0,0,255), -1)
        cv2.imshow(name,img)

    def collectData(self):
        # collect and transfer data to store in pickle file
        x = []
        y = []

        Files = os.listdir(self.preProcessedDir)
        limit = len(Files)
        index = 0
        while index < len(Files):
            image = Files[index]

            while image == ".DS_Store" or image[-12:] != "Normalized.p":
                index += 1
                if index >= limit:
                    break
                image = Files[index]
                
            if index >= limit:
                break


            if image[-12:] == "Normalized.p":
                pts = image[:-12] + ".p"
                img = pickle.load( open( self.preProcessedDir + image, "rb" ) )
                img = img.reshape((self.size[0], self.size[1], 1))
                pts = pickle.load( open( self.preProcessedDir + pts, "rb" ) )

                x.append(img)
                y.append(pts)
            
            index += 1

        x = np.asarray(x)
        y = np.asarray(y)

        print x.shape
        print y.shape

        print type(x)
        print type(y)


        pickle.dump( x, open( self.pFileDir + self.name + "_x.p", "wb" ) )
        pickle.dump( y, open( self.pFileDir + self.name + "_y.p", "wb" ) )


    # def resize(self, image, X, Y):
    #     # resize imgage to determined size maintaing the original ratio

    #     (yMaxBound, xMaxBound, _) = image.shape

    #     newX = [x/float(xMaxBound) for x in X]
    #     newY = [y/float(yMaxBound) for y in Y]


    #     image = Image.fromarray(np.uint8(image))
    #     image.thumbnail(self.size, Image.ANTIALIAS)
    #     image_size = image.size


    #     (yMaxBound, xMaxBound) = image.size

    #     newX = [x*float(xMaxBound) for x in newX]
    #     newY = [y*float(yMaxBound) for y in newY]


    #     thumb = image.crop( (0, 0, self.size[0], self.size[1]) )

    #     image = np.asarray(thumb)


    #     offset_x = max( (self.size[0] - image_size[0]) / 2, 0 )
    #     offset_y = max( (self.size[1] - image_size[1]) / 2, 0 )

    #     newX = [x + offset_x for x in newX]
    #     newY = [y + offset_y for y in newY]

    #     thumb = ImageChops.offset(thumb, offset_x, offset_y)

    #     return image, newX, newY

    # def splitData(self):
    #     # split data into train and test sets
    #     files = os.listdir(self.pFileDir)
    #     X = []
    #     Y = []

    #     for file in files:
    #         if file != ".DS_Store":
    #             if file[-3:-2] == "x":
    #                 print file
    #                 x = pickle.load( open( self.pFileDir + file, "rb" ) )
    #                 (num, d1, d2, _) = x.shape
    #                 x = x.reshape((num, d1 * d2))  
    #                 # x = x.reshape((num, d1, d2, 1)) 
    #                 X.extend(x)
    #             elif file[-3:-2] == "y":
    #                 print file
    #                 y = pickle.load( open( self.pFileDir + file, "rb" ) )
    #                 Y.extend(y)

    #     X = np.asarray(X)
    #     Y = np.asarray(Y)

    #     print X.shape
    #     print Y.shape

    #     print type(X)
    #     print type(Y)
    #     xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

    #     # np.save(self.trainTestDir + 'xTrain.npy', xTrain) 
    #     # np.save(self.trainTestDir + 'yTrain.npy', yTrain) 
    #     # np.save(self.trainTestDir + 'xTest.npy', xTest) 
    #     # np.save(self.trainTestDir + 'yTest.npy', yTest) 

    #     pickle.dump( xTrain, open( self.trainTestDir + "xTrainFlatten.p", "wb" ) )
    #     pickle.dump( xTest, open( self.trainTestDir + "xTestFlatten.p", "wb" ) )
    #     pickle.dump( yTrain, open( self.trainTestDir + "yTrain.p", "wb" ) )
    #     pickle.dump( yTest, open( self.trainTestDir + "yTest.p", "wb" ) )

    # def combineData(self, dir, tag):
    #     files = os.listdir(dir)
    #     X = []
    #     Y = []

    #     for file in files:
    #         if file != ".DS_Store":
    #             if file[-3:-2] == "x":
    #                 print file
    #                 x = pickle.load( open( dir + file, "rb" ) )
    #                 (num, d1, d2, _) = x.shape
    #                 x = x.reshape((num, d1 * d2))  
    #                 # x = x.reshape((num, d1, d2, 1)) 
    #                 X.extend(x)
    #             elif file[-3:-2] == "y":
    #                 print file
    #                 y = pickle.load( open( dir + file, "rb" ) )
    #                 Y.extend(y)

    #     X = np.asarray(X)
    #     Y = np.asarray(Y)

    #     print X.shape
    #     print Y.shape

    #     print type(X)
    #     print type(Y)


    #     pickle.dump( X, open( self.trainTestDir + "x" + tag + "FlattenSpec" + ".p", "wb" ) )
    #     pickle.dump( Y, open( self.trainTestDir + "y" + tag + "Spec" + ".p", "wb" ) )

    def run(self):
        self.getData()
        # self.collectData()
        # self.splitData()
        # self.combineData(self.pFileDir + "test/", "Test")



if __name__ == '__main__':
    preProcess().run()