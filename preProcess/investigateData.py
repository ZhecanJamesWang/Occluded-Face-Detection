

import pickle
import cv2
import numpy as np
import datetime

class InvestigateData(object):
    def __init__(self):
        
        trainTestDir = "./data/trainTestData/"
        self.xTest = pickle.load( open( trainTestDir + "xTrainFlattenSpec.p", "rb" ) )
        self.yTest = pickle.load( open( trainTestDir + "yTrainSpec.p", "rb" ) )

        # self.suPred = pickle.load( open( trainTestDir + "supervisedTrainPredSpec400Epoch.p", "rb" ) )[:2000]
        # self.unsPred = pickle.load( open( trainTestDir + "unSupervisedTrainPredSpec400Epoch.p", "rb" ) )[:2000]
        
  
        self.output = pickle.load( open( trainTestDir + "firstAEDoutput2016-11-24 17:32:39.113052(30trainData).p", "rb" ) )[:2000]


        # print self.xTest.shape
        # print self.yTest.shape

        # print self.suPred.shape
        # print self.unsPred.shape

        print self.output.shape

    def parseLocation(self, array):
        x = []
        y = []
        for i in range(0, len(array), 2):
            x.append(array[i])
            y.append(array[i + 1])
        return x, y

    def emptyMatrix(self):
        newImg = np.zeros((2500))
        for j in range(len(newImg)):
            newImg[j] = 255
        newImg = newImg.reshape((50, 50))
        return newImg        

    def check(self):

        print len(self.xTest)


        for i in range(0, len(self.xTest)):
            img = self.xTest[i]
            img = img.reshape((50, 50))
            cv2.imshow("face",img)

            output = self.output[i]
            output = output.reshape((50, 50))
            cv2.imshow("recovered",output)
            cv2.waitKey(0)

            # newImg1 = self.emptyMatrix()
            # newImg2 = self.emptyMatrix()
            # newImg3 = self.emptyMatrix()


            # suPredArray = self.suPred[i]
            # suPredX, suPredY = self.parseLocation(suPredArray)

            # unsPredArray = self.unsPred[i]
            # unsPredX, unsPredY = self.parseLocation(unsPredArray)

            # labelArray = self.yTest[i]
            # labelX, labelY = self.parseLocation(labelArray)

            # for index in range(len(labelX)):
            #     cv2.circle(newImg1, (int(labelX[index]*50), int(labelY[index]*50)), 1, (0,0,255), -1)
            # cv2.imshow("label",newImg1)

            # for index in range(len(unsPredX)):
            #     cv2.circle(newImg2, (int(unsPredX[index]*50), int(unsPredY[index]*50)), 1, (0,0,255), -1)
            # cv2.imshow("unsPred",newImg2)

            # for index in range(len(suPredX)):
            #     cv2.circle(newImg3, (int(suPredX[index]*50), int(suPredY[index]*50)), 1, (0,0,255), -1)
            # cv2.imshow("suPred",newImg3)








            # if cv2.waitKey(0)& 0xFF == ord('q'):
            #     print "triggered"
            #     dir = "./data/tmp/"
            #     print dir + str(datetime.datetime.now()) + "img"
            #     cv2.imwrite(dir + str(datetime.datetime.now()) + "img" + '.jpg', img)
            #     cv2.imwrite(dir + str(datetime.datetime.now()) + "newImg1" + '.jpg', newImg1) 
            #     cv2.imwrite(dir + str(datetime.datetime.now()) + "newImg2" + '.jpg', newImg2) 
            #     cv2.imwrite(dir + str(datetime.datetime.now()) + "newImg3" + '.jpg', newImg3) 

            #     cv2.destroyAllWindows()


if __name__ == '__main__':
    InvestigateData().check()

