"""
Programmer  :   EOF
File        :   testing.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
from image import ImageSet
from matplotlib import pyplot
import numpy
import os
from config import *
from adaboost import AdaBoost
from haarFeature import Feature

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

    haar = Feature(SEARCH_WIN_HEIGHT, SEARCH_WIN_HEIGHT, Row, Col)

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

fileObj = open(ADABOOST_FILE, "a+")

print "Constructing AdaBoost from existed model data"

tmp = fileObj.readlines()

a = AdaBoost(train = False)

for i in range(0, len(tmp), 4):

    alpha, demention, direction, threshold = None, None, None, None

    for j in range(i, i + 4):
        if (j % 4) == 0:
            alpha = float(tmp[j])
        elif (j % 4) == 1:
            demention = int(tmp[j])
        elif (j % 4) == 2:
            direction = float(tmp[j])
        elif (j % 4) == 3:
            threshold = float(tmp[j])

    classifier = a.Weaker(train = False)
    classifier.constructor(demention, direction, threshold)
    a.G[i/4] = classifier
    a.alpha[i/4] = alpha
    a.N += 1

print "Construction finished"
fileObj.close()

output = a.prediction(Original_Data)

print numpy.count_nonzero(output[0:TESTING_POSITIVE_SAMPLE] > 0) * 1./ TESTING_POSITIVE_SAMPLE

print numpy.count_nonzero(output[TESTING_POSITIVE_SAMPLE:TESTING_SAMPLE_NUM] < 0) * 1./ TESTING_NEGATIVE_SAMPLE
