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

        self.vecImg  = self.iimg.transpose().flatten()

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

        #@stdImag standardized image
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
