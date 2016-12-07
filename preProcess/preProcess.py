import numpy as np
import cv2
import os
from PIL import Image, ImageChops
import pickle
from sklearn.cross_validation import train_test_split
import utility as ut

class preProcess(object):
    def __init__(self):       
        self.tag = "Preprocessed"
        self.pFileDir = "./data/pFile/"      
        self.trainTestDir = "./data/trainTestData/"

        self.padding = 50
        self.size = (50, 50)
        self.debug = False
        self.derivativeNum = 3
        self.firstSave = True
        self.dataDict = {
        "afw": {"type": ".jpg", "method": "files"}, 
        "helen": {"type": ".jpg", "method": "folders"}, 
        "ibug": {"type": ".jpg", "method": "files"}, 
        "lfpw": {"type": ".png", "method": "folders"}}

    def getData(self):

        for key in self.dataDict:
            data = self.dataDict[key]
            print "getting data from: ", key
            self.format = data["type"]
            self.name = key
            self.rawDir = "./data/" +self.name + "/"
            self.preProcessedDir = "./data/" + self.name + self.tag + "/" 
            if data["method"] == "folders":
                self.getDataByFolders()
            else:
                self.getDataByFiles()


    def getDataByFolders(self):
        # This function initialize reading image and landmark location data

        preCounter, counter = 0, 0

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
                        self.saveImg(imgs, landmarks, file)
                        counter += self.derivativeNum * 4 + 2

                        if counter - preCounter > 100:
                            preCounter = counter
                            print counter
                            # print path

    def getDataByFiles(self):
        counter = 0
        # folders = os.listdir(self.rawDir)
        files = os.listdir(self.rawDir)

        # for fold in folders:
        #     if fold != ".DS_Store":
        #         path = os.path.abspath(self.rawDir + fold)
        #         files = os.listdir(path)

        for file in files:
            if file != ".DS_Store" and self.format in file:
                # imgs, landmarks = self.extract(path + "/", file)
                imgs, landmarks = self.extract(self.rawDir + "/", file)                
                self.saveImg(imgs, landmarks, file)
                counter += 1

                if counter % 100 == 0:
                    print counter
                    # print path



    def saveImg(self, imgs, landmarks, fileName):
        # save image to directory
         # cv2.imwrite(dataDir + fileName,file)

        for index in range(len(imgs)):
            img = imgs[index]
            landmark = landmarks[index]
            if self.debug:
                X, Y = ut.unpackLandmarks(landmark)
                self.plotLandmarks(img, X, Y, "spec", ifRescale = True)
                cv2.waitKey(1000)
            else:
                pickle.dump( img, open( self.preProcessedDir + fileName[:-4] + "Normalized" + str(index) + ".p", "wb" ) )
                pickle.dump( landmark, open( self.preProcessedDir + fileName[:-4] + "Landmarks" + str(index) + ".p", "wb" ) )            
            

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

        for i in range(self.derivativeNum):
            img = originalImg.copy()
            scaleImage, newX, newY = ut.resize(img, X, Y, random = True)
            imgs.append(scaleImage)
            Xs.append(newX)
            Ys.append(newY)

            if self.debug:
                self.plotLandmarks(scaleImage, newX, newY, "scale")

        for i in range(self.derivativeNum):
            img = originalImg.copy()
            rotateImage, newX, newY = ut.rotate(img, X, Y)
            if rotateImage != None:
                imgs.append(rotateImage)
                Xs.append(newX)
                Ys.append(newY)
                if self.debug:
                    self.plotLandmarks(rotateImage, newX, newY, "rotate")

        for i in range(self.derivativeNum):
            img = originalImg.copy()
            cbImage, newX, newY = ut.contrastBrightess(img, X, Y)
            imgs.append(cbImage)
            Xs.append(newX)
            Ys.append(newY)
            if self.debug: 
                self.plotLandmarks(cbImage, newX, newY, "contrastBrightness")

        for i in range(self.derivativeNum):
            img = originalImg.copy()
            transImage, newX, newY = ut.translateImage(img, X, Y)
            if transImage != None:
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
        return filterImages, landmarks

    def unpackLandmarks(self, array):
        x = []
        y = []
        for i in range(0, len(array), 2):
            x.append(array[i])
            y.append(array[i + 1])
        return x, y

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

    def plotLandmarks(self, img, X, Y, name, ymax = 50, xmax = 50, ifRescale = False):
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
        for key in self.dataDict:
            print "getting data from: ", key
            data = self.dataDict[key]            
            self.format = data["type"]
            self.name = key
            self.preProcessedDir = "./data/" + self.name + self.tag + "/"

            x = []
            y = []

            Files = os.listdir(self.preProcessedDir)
            limit = len(Files)
            index = 0
            while index < len(Files):
                file = Files[index]

                while file == ".DS_Store" and "Normalized" not in file:
                    index += 1
                    if index >= limit:
                        break
                    file = Files[index]
                
                if index >= limit:
                    break

                if "Normalized" in file:
                    image = file
                    position = image.index("Normalized")
                    header = image[: position]
                    ender = image[position + len("Normalized"):]
                    landmark = header + "Landmarks" + ender

                    img = pickle.load( open( self.preProcessedDir + image, "rb" ) )
# !!!!!
                    # img = img.reshape((self.size[0], self.size[1], 1))
                    if self.debug:
                        landmark = pickle.load( open( self.preProcessedDir + landmark, "rb" ) )
                        X, Y = ut.unpackLandmarks(landmark)
                        self.plotLandmarks(img, X, Y, "spec", ifRescale = True)
                        cv2.waitKey(1000)                    

                    x.append(img)
                    y.append(landmark)
                
                index += 1

            x = np.asarray(x)
            y = np.asarray(y)

            print x.shape
            print y.shape

            if self.firstSave:
                self.now = datetime.datetime.now().isoformat()
                self.firstSave = False

            pickle.dump( x, open( self.pFileDir + self.name + self.now + "_x.p", "wb" ) )
            pickle.dump( y, open( self.pFileDir + self.name + self.now + "_y.p", "wb" ) )

        self.firstSave = True

    def splitData(self):
        # split data into train and test sets
        files = os.listdir(self.pFileDir)
        X = []
        Y = []

        for file in files:
            if file != ".DS_Store":
                if file[-3:-2] == "x":
                    print file
                    x = pickle.load( open( self.pFileDir + file, "rb" ) )
                    (num, d1, d2, _) = x.shape
# !!!!!   
                    x = x.reshape((num, d1 * d2))  
                    # x = x.reshape((num, d1, d2, 1)) 
                    X.extend(x)
                elif file[-3:-2] == "y":
                    print file
                    y = pickle.load( open( self.pFileDir + file, "rb" ) )
                    Y.extend(y)

        X = np.asarray(X)
        Y = np.asarray(Y)

        print X.shape
        print Y.shape

        print type(X)
        print type(Y)
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.33, random_state=42)

        self.now = datetime.datetime.now().isoformat()
        pickle.dump( xTrain, open( self.trainTestDir + self.now + "xTrainFlatten.p", "wb" ) )
        pickle.dump( xTest, open( self.trainTestDir + self.now + "xTestFlatten.p", "wb" ) )
        pickle.dump( yTrain, open( self.trainTestDir + self.now + "yTrain.p", "wb" ) )
        pickle.dump( yTest, open( self.trainTestDir + self.now + "yTest.p", "wb" ) )

    def combineData(self):
        dirs = [self.pFileDir + "test/", self.pFileDir + "train/"] 
        tags = ["Test", "Train"]

        for index in range(len(dirs)):
            dr = dirs[index]
            tag = tags[index]

            files = os.listdir(dr)
            X = []
            Y = []

            for file in files:
                if file != ".DS_Store":
                    if file[-3:-2] == "x":
                        print file
                        x = pickle.load( open( dir + file, "rb" ) )
                        (num, d1, d2, _) = x.shape
# !!!!! 
                        x = x.reshape((num, d1 * d2))  
                        # x = x.reshape((num, d1, d2, 1)) 
                        X.extend(x)
                    elif file[-3:-2] == "y":
                        print file
                        y = pickle.load( open( dir + file, "rb" ) )
                        Y.extend(y)

            X = np.asarray(X)
            Y = np.asarray(Y)

            print X.shape
            print Y.shape
            print type(X)
            print type(Y)

            if self.firstSave:
                self.now = datetime.datetime.now().isoformat()
                self.firstSave = False

            pickle.dump( X, open( self.trainTestDir + self.now + "x" + tag + "FlattenSpec" + ".p", "wb" ) )
            pickle.dump( Y, open( self.trainTestDir + self.now  + "y" + tag + "Spec" + ".p", "wb" ) )
            
        self.firstSave = True


    def run(self):
        self.getData()
        self.collectData()
        # self.splitData()
        self.combineData()



if __name__ == '__main__':
    preProcess().run()