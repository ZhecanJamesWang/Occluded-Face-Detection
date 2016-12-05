import math
import cv2    
import numpy as np
from PIL import Image, ImageChops
from pylab import array, uint8 

def mirrorImage(image, X, Y):
    (h, w, _) = image.shape
    image = np.fliplr(image)
    X = w - X
    X = np.asarray(X)
    Y = np.asarray(Y)
    return image, X.astype(int), Y.astype(int)


def rotate(image, X, Y):
    originalImage = image.copy()
    (h, w, _) = image.shape
    degree = np.random.uniform(-45, 45)

    M = cv2.getRotationMatrix2D((w/2, h/2),degree,1)
    
    newX = []
    newY = []
    for x, y in zip(X, Y):
        newX.append(M[0, 0] * x + M[0, 1] * y + M[0, 2])
        newY.append(M[1, 0] * x + M[1, 1] * y + M[1, 2])
   
    image = cv2.warpAffine(image,M,(w, h))

    if np.min(newX) < 0 or np.min(newY) < 0 or np.max(newX) > w or np.max(newY) > h:
        image, newX, newY = rotate(originalImage, X, Y) 

    newX = np.asarray(newX)
    newY = np.asarray(newY)
    return image, newX, newY


def contrastBrightess(image, X, Y):
    contrast = np.random.uniform(0.5, 3)
    brightness = np.random.uniform(-50, 50)
    # contrast = 2
    # brightness = 50

    maxIntensity = 255.0 # depends on dtype of image data
    phi = 1
    theta = 1
    image = ((maxIntensity/phi)*(image/(maxIntensity/theta))**contrast) + brightness
    image = np.asarray(image)
    top_index = np.where(image > 255)
    bottom_index = np.where(image < 0)
    image[top_index] = 255
    image[bottom_index] = 0
    image = array(image,dtype=uint8)
    X = np.asarray(X)
    Y = np.asarray(Y)
    return image, X, Y


def plotLandmarks(img, X, Y, name):
    # plot landmarks on original image
    print len(X)
    assert len(X) == len(Y)      
    for index in range(len(X)):
        cv2.circle(img,(int(X[index]), int(Y[index])), 2, (0,0,255), -1)
    cv2.imshow(name,img)


def resize(image, X, Y, random = False, size = (200, 200)):
    originalImage = image
    # resize imgage to determined size maintaing the original ratio
    (yMaxBound, xMaxBound, _) = image.shape
    print (yMaxBound, xMaxBound)

    newX = [x/float(xMaxBound) for x in X]
    newY = [y/float(yMaxBound) for y in Y]


    if random:
        ratio = np.random.uniform(0.5, 1)
        size = (int(xMaxBound*ratio), int(yMaxBound*ratio))

    image = Image.fromarray(np.uint8(image))
    image.thumbnail(size, Image.ANTIALIAS)
    image_size = image.size

    print image_size
    (newXMaxBound, newYMaxBound) = image.size

    newX = [x*float(newXMaxBound) for x in newX]
    newY = [y*float(newYMaxBound) for y in newY]

    thumb = image.crop( (0, 0, size[0], size[1]) )
    image = np.asarray(thumb)

    offset_y = (size[0] - image_size[1]) / 2 
    offset_x = (size[1] - image_size[0]) / 2

    newX = [x + offset_x for x in newX]
    newY = [y + offset_y for y in newY]
    
    thumb = ImageChops.offset(thumb, offset_x, offset_y)



    image = np.asarray(thumb)

    if random:
        newImg = np.zeros_like(originalImage)
        
        offset_y = int((yMaxBound - image_size[1]) / 2)
        offset_x = int((xMaxBound - image_size[0]) / 2)
        other_offset_y = -offset_y if image_size[1] % 2 == 0 else -(offset_y + 1)
        other_offset_x = -offset_x if image_size[0] % 2 == 0 else -(offset_x + 1)

        newX = [x + offset_x for x in newX]
        newY = [y + offset_y for y in newY]
        newImg[offset_y:other_offset_y, offset_x:other_offset_x] = image
        image = newImg

    newX = np.asarray(newX)
    newY = np.asarray(newY)

    return image, newX.astype(int), newY.astype(int)

def translateImage(image, X, Y):
    originalImage = image.copy()
    (h, w, _) = image.shape
    xTransRange, yTransRange = np.random.randint(0, w/5), np.random.randint(0, h/5)

    newImg = np.zeros_like(image)
    newImg[yTransRange:, xTransRange:] = image[:int(h - yTransRange), :int(w - xTransRange)]
    
    newX = X + xTransRange
    newY = Y + yTransRange

    image = newImg

    if np.min(newX) < 0 or np.min(newY) < 0 or np.max(newX) > w or np.max(newY) > h:
        image, newX, newY = translateImage(originalImage, X, Y) 

    newX = np.asarray(newX)
    newY = np.asarray(newY)
    return image, newX, newY

def test():
    dataDir = "./data/ibug/"
    # Load an color image in grayscale
    picName = "image_008_1.jpg"
    ptsName = "image_008_1.pts"


    img = cv2.imread(dataDir + picName,1)
    cv2.imshow("original", img)

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
        else:
            initDataCounter += 1


    X = np.asarray(X)
    Y = np.asarray(Y)
    print X.shape
    print Y.shape
    # cleanImg = np.copy(img)
    cleanImg = img.copy()

    plotLandmarks(img, X, Y, "img")
    while True:

        mirImage, newX, newY = mirrorImage(cleanImg, X, Y)
        mirImage = mirImage.copy()
        cv2.imshow("mirrorClean", mirImage)
        plotLandmarks(mirImage, newX, newY, "mirror")
        print newX.shape
        print newY.shape

        resizeImage, newX, newY = resize(cleanImg, X, Y)

        cv2.imshow("resizeClean", resizeImage)
        cleanResizeImage = np.copy(resizeImage)
        plotLandmarks(resizeImage, newX, newY, "resize")

        resizeImage1, newX, newY = resize(cleanResizeImage, newX, newY, random = True)
        cv2.imshow("resizeClean1", resizeImage1)
        plotLandmarks(resizeImage1, newX, newY, "resize1")

        rotateImage, newX, newY = rotate(cleanImg, X, Y)
        cv2.imshow("rotateClean", rotateImage)
        plotLandmarks(rotateImage, newX, newY, "rotate")

        cbImage, newX, newY = contrastBrightess(cleanImg, X, Y)
        cv2.imshow("contrastBrightnessClean", cbImage)
        plotLandmarks(cbImage, newX, newY, "contrastBrightness")

        transImage, newX, newY = translateImage(cleanImg, X, Y)
        cv2.imshow("transClean", transImage)
        plotLandmarks(transImage, newX, newY, "trans")

        cv2.waitKey(1000)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
