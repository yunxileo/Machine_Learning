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
from adaboost import AdaBoost

TrainingSetFace      = ImageSet("./newtraining/face/")
TrainingSetNonFace   = ImageSet("./newtraining/non-face/")

haarVal1 = abs(numpy.array(TrainingSetFace.images[2].haarD))
haarVal2 = abs(numpy.array(TrainingSetNonFace.images[2].haarD))

#haarVal1 = TrainingSetFace.images[0].haarD
#haarVal2 = TrainingSetNonFace.images[0].haarD

haarVal1.sort()
haarVal2.sort()

print "face:", numpy.count_nonzero(haarVal1 > 0.12)* 1./len(haarVal1)
print "Nonface:", numpy.count_nonzero(haarVal2 > 0.12)* 1./len(haarVal2)

for i in range(0, len(haarVal1), 100):
    pyplot.plot(i, haarVal1[i], "or")
    pyplot.plot(i, haarVal2[i], "ob")

pyplot.show()

"""
haarBFace = []
haarBNonFace = []

for i in range(TrainingSetFace.sampleNum):
    haarBFace.append(abs(sum(TrainingSetFace.images[i].haarB))/ 
                     len(TrainingSetFace.images[i].haarB))

for i in range(TrainingSetNonFace.sampleNum):
    haarBNonFace.append(abs(sum(TrainingSetNonFace.images[i].haarB))/
                        len(TrainingSetNonFace.images[i].haarB))

haarBFace.sort()
haarBNonFace.sort()

for i in range(TrainingSetFace.sampleNum):
    pyplot.plot(i, haarBFace[i], "or")

for i in range(TrainingSetNonFace.sampleNum):
    pyplot.plot(i, haarBNonFace[i], "ob")

pyplot.show()
"""
