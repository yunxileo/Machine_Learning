"""
Programmer  :   EOF
File        :   testing.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
from matplotlib import pyplot
import numpy
import os

from image import ImageSet
from config import *
from adaboost import AdaBoost
from haarFeature import Feature
from getCachedAdaBoost import getCachedAdaBoost

#FEATURE_FILE_TESTING = FEATURE_FILE_TRAINING
#
#TESTING_SAMPLE_NUM = SAMPLE_NUM
#TESTING_NEGATIVE_SAMPLE = NEGATIVE_SAMPLE
#TESTING_POSITIVE_SAMPLE = POSITIVE_SAMPLE

fileObj = open(FEATURE_FILE_TESTING, "a+")

# if that is a empty file
if os.stat(FEATURE_FILE_TESTING).st_size == 0:

    print "First time to load the testing set ..."
    TestSetFace          = ImageSet(TEST_FACE, sampleNum = TESTING_POSITIVE_SAMPLE)
    TestSetNonFace       = ImageSet(TEST_NONFACE, sampleNum = TESTING_NEGATIVE_SAMPLE)

    Row = TestSetFace.images[0].Row
    Col = TestSetFace.images[0].Col

    haar = Feature(SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT, Col, Row)

    Original_Data = [[] for i in range(len(haar.features))]

    dem = 0
    for feature in haar.features:
        (types, x, y, w, h) = feature

        for i in range(TestSetFace.sampleNum):
            if types == "I":
                Original_Data[dem].append(haar.VecFeatureTypeI( TestSetFace.images[i].vecImg, x, y, w, h))
            elif types == "II":
                Original_Data[dem].append(haar.VecFeatureTypeII( TestSetFace.images[i].vecImg, x, y, w, h))
            elif types == "III":
                Original_Data[dem].append(haar.VecFeatureTypeIII( TestSetFace.images[i].vecImg, x, y, w, h))
            elif types == "IV":
                Original_Data[dem].append(haar.VecFeatureTypeIV( TestSetFace.images[i].vecImg, x, y, w, h))

        for i in range(TestSetNonFace.sampleNum):
            if types == "I":
                Original_Data[dem].append(haar.VecFeatureTypeI( TestSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "II":
                Original_Data[dem].append(haar.VecFeatureTypeII( TestSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "III":
                Original_Data[dem].append(haar.VecFeatureTypeIII( TestSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "IV":
                Original_Data[dem].append(haar.VecFeatureTypeIV( TestSetNonFace.images[i].vecImg, x, y, w, h))

        dem += 1

        print "processed: dem= ", dem

    Original_Data = numpy.array(Original_Data)

    for i in range(Original_Data.shape[0]):
        for j in range(Original_Data.shape[1]):
            fileObj.write(str(Original_Data[i][j]) + "\n")

    fileObj.flush()
else:
    print "Haar features have been calculated."
    print "Loading features ..."

    tmp = fileObj.readlines()

    Original_Data = []
    for i in range(FEATURE_NUM):
        haarGroup = []
        for j in range(i * TESTING_SAMPLE_NUM, (i+1) * TESTING_SAMPLE_NUM):
            haarGroup.append(float(tmp[j]))

        Original_Data.append(haarGroup)

    Original_Data = numpy.array(Original_Data)

fileObj.close()

model = getCachedAdaBoost()

output = model.prediction(Original_Data)

print numpy.count_nonzero(output[0:TESTING_POSITIVE_SAMPLE] > 0) * 1./ TESTING_POSITIVE_SAMPLE

print numpy.count_nonzero(output[TESTING_POSITIVE_SAMPLE:TESTING_SAMPLE_NUM] < 0) * 1./ TESTING_NEGATIVE_SAMPLE
