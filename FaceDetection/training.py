"""
Programmer  :   EOF
File        :   tester.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

"""

from matplotlib import pyplot
import numpy
import os

from config import *
from adaboost import AdaBoost
from image import ImageSet
from haarFeature import Feature
from weakClassifier import WeakClassifier
from vecProduct import VecProduct
"""
function @saveModel save the key data member of AdaBoost 
into a template file @ADABOOST_FILE

Parameter:
    @model : A object of class AdaBoost

"""
def saveModel(model):

    fileObj = open(ADABOOST_FILE, "a+")

    for m in range(model.N):
        fileObj.write(str(model.alpha[m]) + "\n")
        fileObj.write(str(model.G[m].opt_demention) + "\n")
        fileObj.write(str(model.G[m].opt_direction) + "\n")
        fileObj.write(str(model.G[m].opt_threshold) + "\n")

    fileObj.flush()
    fileObj.close()

def saveOriginalDate(Original_Data):
    Original_Data = numpy.array(Original_Data)
    for i in range(Original_Data.shape[0]):
        for j in range(Original_Data.shape[1]):
            fileObj.write(str(Original_Data[i][j]) + "\n")

    fileObj.flush()


fileObj = open(FEATURE_FILE_TRAINING, "a+")

# if that is a empty file
if os.stat(FEATURE_FILE_TRAINING).st_size == 0:

    print "First time to load the training set ..."

    TrainingSetFace      = ImageSet(TRAINING_FACE, 
                                    sampleNum = POSITIVE_SAMPLE)

    TrainingSetNonFace   = ImageSet(TRAINING_NONFACE, 
                                    sampleNum = NEGATIVE_SAMPLE)

    Row = TrainingSetFace.images[0].Row
    Col = TrainingSetFace.images[0].Col

    haar = Feature(SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT,Row, Col)

    Original_Data = [[] for i in range(len(haar.features))]

    dem = 0
    for feature in haar.features:
        (types, x, y, w, h) = feature

        for i in range(TrainingSetFace.sampleNum):
            if types == "I":
                Original_Data[dem].append(haar.VecFeatureTypeI( TrainingSetFace.images[i].vecImg, x, y, w, h))
            elif types == "II":
                Original_Data[dem].append(haar.VecFeatureTypeII( TrainingSetFace.images[i].vecImg, x, y, w, h))
            elif types == "III":
                Original_Data[dem].append(haar.VecFeatureTypeIII( TrainingSetFace.images[i].vecImg, x, y, w, h))
            elif types == "IV":
                Original_Data[dem].append(haar.VecFeatureTypeIV( TrainingSetFace.images[i].vecImg, x, y, w, h))

        for i in range(TrainingSetNonFace.sampleNum):
            if types == "I":
                Original_Data[dem].append(haar.VecFeatureTypeI( TrainingSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "II":
                Original_Data[dem].append(haar.VecFeatureTypeII( TrainingSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "III":
                Original_Data[dem].append(haar.VecFeatureTypeIII( TrainingSetNonFace.images[i].vecImg, x, y, w, h))
            elif types == "IV":
                Original_Data[dem].append(haar.VecFeatureTypeIV( TrainingSetNonFace.images[i].vecImg, x, y, w, h))

        dem += 1

        print "processed: dem = " , dem

    saveOriginalDate(Original_Data)
else:
    print "Haar features have been calculated."
    print "Loading features ..."
    tmp = fileObj.readlines()

    Original_Data = []
    for i in range(FEATURE_NUM):
        haarGroup = []
        for j in range(i * SAMPLE_NUM , (i+1) * SAMPLE_NUM):
            haarGroup.append(float(tmp[j]))

        Original_Data.append(haarGroup)

    Original_Data = numpy.array(Original_Data)


fileObj.close()

SampleNum = Original_Data.shape[1]

assert SampleNum == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)

Label_Face    = [+1 for i in range(POSITIVE_SAMPLE)]
Label_NonFace = [-1 for i in range(NEGATIVE_SAMPLE)]

Label = numpy.array(Label_Face + Label_NonFace)

a = AdaBoost(Original_Data, Label)

try:
    a.train(200)

except KeyboardInterrupt:
    print "You pressed interrupt key. Training process interrupt."

saveModel(a)
