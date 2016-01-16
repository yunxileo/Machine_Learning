"""
Programmer  :   EOF
File        :   tester.py
Date        :   2015.12.29
E-mail      :   jasonleaster@163.com

Description :

"""
from image import ImageSet
from matplotlib import pyplot
import numpy
import os
from config import *
from decisionStump import DecisionStump

fileObj = open(FEATURE_FILE_TRAINING, "a+")

# if that is a empty file
if os.stat(FEATURE_FILE_TRAINING).st_size == 0:

    print "First time to load the training set ..."
    TrainingSetFace      = ImageSet(TRAINING_FACE, sampleNum = POSITIVE_SAMPLE)
    TrainingSetNonFace   = ImageSet(TRAINING_NONFACE, sampleNum = NEGATIVE_SAMPLE)

    Original_Data_Face = [
         TrainingSetFace.images[i].haarA + 
         TrainingSetFace.images[i].haarB + 
         TrainingSetFace.images[i].haarC +
         TrainingSetFace.images[i].haarD
        for i in range(TrainingSetFace.sampleNum)
        ]

    Original_Data_NonFace =[ 
         TrainingSetNonFace.images[i].haarA +
         TrainingSetNonFace.images[i].haarB +
         TrainingSetNonFace.images[i].haarC +
         TrainingSetNonFace.images[i].haarD
        for i in range(TrainingSetNonFace.sampleNum)
        ]

    Original_Data = numpy.array(Original_Data_Face + \
                        Original_Data_NonFace).transpose()

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
        for j in range(i, i + POSITIVE_SAMPLE + NEGATIVE_SAMPLE):
            haarGroup.append(float(tmp[j]))

        Original_Data.append(haarGroup)

    Original_Data = numpy.array(Original_Data)


fileObj.close()

SampleDem = Original_Data.shape[0]
SampleNum = Original_Data.shape[1]

assert SampleNum == (POSITIVE_SAMPLE + NEGATIVE_SAMPLE)
assert SampleDem == FEATURE_NUM

Label_Face    = [+1 for i in range(POSITIVE_SAMPLE)]
Label_NonFace = [-1 for i in range(NEGATIVE_SAMPLE)]

Label = numpy.array(Label_Face + Label_NonFace)

linkedList = []

processing = 0.
for i in range(Original_Data.shape[0]):
    classifier = DecisionStump([Original_Data[i, :]], Label)
    classifier.train()

    linkedList.append(classifier.opt_errorRate)

    if i % (Original_Data.shape[0] / 10) == 0:
        print "processing %", processing
        processing += 10.

print linkedList.index(min(linkedList))
