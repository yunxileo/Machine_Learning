"""
Programmer  :   EOF
File        :   init_training_set.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :
    This script file will initialize the image set
and read all images in the directory which is given by
user.

"""
import numpy
import os
import pylab
from matplotlib import pyplot
from matplotlib import image

class Image:
    def __init__(self, fileName = None, label = None):
        self.imgName = fileName
        self.img     = image.imread(fileName)
        self.label   = label

        self.Row     = self.img.shape[0]
        self.Col     = self.img.shape[1]

        self.stdImg  = self.__normalization()

        self.iimg    = self.__integrateImg()

        self.featureTypes = ["I", "II", "III", "IV"]

        self.haarFeatureVal = []

    def __integrateImg(self):
        image = self.stdImg

        #@iImg is integrated image of normalized image @self.stdImg
        iImg = numpy.array([ [0. for i in range(self.Col)] 
                                 for j in range(self.Row)])

        for i in range(0, self.Row):
            for j in range(0, self.Col):
                if j == 0:
                    iImg[i][j] = image[i][j]
                else:
                    iImg[i][j] = iImg[i][j - 1] + image[i][j]

        for j in range(0, self.Col):
            for i in range(1, self.Row):
                iImg[i][j] += iImg[i - 1][j]

        return iImg

    def __normalization(self):
        image = self.img

        #@nImag normalized image
        stdImg = numpy.array([ [0. for i in range(self.Col)] 
                                   for j in range(self.Row)])
        sigma = 0.
        for i in range(self.Row):
            for j in range(self.Col):
                sigma += image[i][j]

        meanVal = sigma / (self.Row * self.Col)

        for i in range(self.Row):
            for j in range(self.Col):
                stdImg[i][j] = (image[i][j] - meanVal) / numpy.std(image)

        return stdImg

    """
    Types of Haar-like rectangle features
      
     --- ---      --- ---
    |   |   |    |   +   |
    | + | - |    |-------|
    |   |   |    |   -   |
     --- ---      -------
       A            B

     -- -- --     -------
    |  |  |  |   |___-___|       
    |- | +| -|   |___+___|
    |  |  |  |   |   -   |
     -- -- --     -------
        C            D

     --- ---
    | - | + |
    |___|___|
    | + | - |
    |___|___|
        E

    For each feature pattern, the start point(x, y) is at 
    the most left-up pixel in that window. The size of that
    window is @width * @height
    """

    def rectangleSum(self, x, y, width, height):
        if x == 0 and y == 0:
            return self.iimg[y + height - 1][ x + width - 1]

        elif x == 0:
            return self.iimg[y + height - 1][x + width - 1] - \
                   self.iimg[y          - 1][x + width - 1]

        elif y == 0:
            return self.iimg[y + height - 1][x + width - 1] - \
                   self.iimg[y + height - 1][x         - 1]

        else:
            return self.iimg[y + height - 1][x + width - 1] + \
                   self.iimg[y          - 1][x         - 1] - \
                   self.iimg[y + height - 1][x         - 1] - \
                   self.iimg[y          - 1][x + width - 1]


    def __featureTypeI(self, x, y, width, height):
        return self.rectangleSum(x, y         , width, height) - \
               self.rectangleSum(x, y + height, width, height)

    def __featureTypeII(self, x, y, width, height):
        return self.rectangleSum(x + width, y, width, height) - \
               self.rectangleSum(x        , y, width, height)

    def __featureTypeIII(self, x, y, width, height):
        return self.rectangleSum(x + width, y, width, height) - \
               self.rectangleSum(x        , y, width, height) - \
               self.rectangleSum(x+2*width, y, width, height)

    def __featureTypeIV(self, x, y, width, height):
        return self.rectangleSum(x + width, y         , width, height) + \
               self.rectangleSum(x        , y + height, width, height) - \
               self.rectangleSum(x        , y         , width, height) - \
               self.rectangleSum(x + width, y + height, width, height)

    def evalFeatures(self, win_Width, win_Height):
        height_Limite = {"I"  : win_Height/2 - 1,
                         "II" : win_Height   - 1,
                         "III": win_Height   - 1,
                         "IV" : win_Height/2 - 1}

        width_Limit  = {"I"   : win_Width   - 1,
                        "II"  : win_Width/2 - 1,
                        "III" : win_Width/3 - 1,
                        "IV"  : win_Width/2 - 1}

        features = []
        for types in self.featureTypes:
            for h in range(1, height_Limite[types]):
                for w in range(1, width_Limit[types]):
                    if types == "I":
                        x_limit = win_Width  - w   
                        y_limit = win_Height - 2*h 
                    elif types == "II":
                        x_limit = win_Width  - 2*w 
                        y_limit = win_Height - h   
                    elif types == "III":
                        x_limit = win_Width  - 3*w 
                        y_limit = win_Height - h   
                    elif types == "IV":
                        x_limit = win_Width  - 2*w 
                        y_limit = win_Height - 2*h 

                    for x in range(1, x_limit):
                        for y in range(1, y_limit):
                            features.append( [types, x, y, w, h])

        return features


    def calFeatures(self, features):
        # features[i] == [types, x, y, w, h]

        for i in range(len(features)):
            (types, x, y, w, h) = [val for val in features[i]]

            if features[i][0] == "I":
                func = self.__featureTypeI
            elif features[i][0] == "II":
                func = self.__featureTypeII
            elif features[i][0] == "III":
                func = self.__featureTypeIII
            elif features[i][0] == "VI":
                func = self.__featureTypeIV
            else:
                raise ValueError("Undefined feature type!")


            self.haarFeatureVal.append( func(x, y, w, h))
            

    def show(self, image = None):
        if image == None:
            image = self.img

        pyplot.imshow(image)
        pylab.show()

class ImageSet:
    def __init__(self, imgDir = None, label = None, sampleNum = None):

        assert isinstance(imgDir, str)

        self.fileList = os.listdir(imgDir)
        self.fileList.sort()

        if sampleNum == None:
            self.sampleNum = len(self.fileList)
        else:
            self.sampleNum = sampleNum

        self.setLabel  = label

        self.images = [None for i in range(self.sampleNum)]
        processed = -10.
        for i in range(self.sampleNum):
            self.images[i] = Image(imgDir + self.fileList[i], label)

            if i % (self.sampleNum / 10) == 0:
                processed += 10.
                print "Loading ", processed, "%"

        print "Loading  100 %\n"
