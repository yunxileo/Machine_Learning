"""
Programmer  :   EOF
Date        :   2015.11.22
File        :   weakclassifier.py

"""

import numpy
from matplotlib import pyplot
from config import *

class WeakClassifier:

    def __init__(self, Mat = None, Tag = None, W = None, train = True):
        if train == True:
            self._Mat = numpy.array(Mat)
            self._Tag = numpy.array(Tag)

            self.SampleNum = self._Mat.shape[1]
            self.SampleDem = self._Mat.shape[0]

            self.NumPos = numpy.count_nonzero(self._Tag == LABEL_POSITIVE)
            self.NumNeg = numpy.count_nonzero(self._Tag == LABEL_NEGATIVE)

            if W == None:
                pos_W = [1.0/(2 * self.NumPos) for i in range(self.NumPos)]
                            
                neg_W = [1.0/(2 * self.NumNeg) for i in range(self.NumNeg)]
                self.weight = pos_W + neg_W

            else:
                self.weight = numpy.array(W)

            self.opt_errorRate = 1.
            self.opt_demention = 0
            self.opt_threshold = None
            self.opt_direction = 0
            

    def optimal(self, d):
        sumPos = 0.
        sumNeg = 0.

        for i in range(self.SampleNum):
            if self._Tag[i] == LABEL_POSITIVE:
                sumPos += self.weight[i] * self._Mat[d][i]
            else:
                sumNeg += self.weight[i] * self._Mat[d][i]
                
        miuPos = sumPos / self.NumPos
        miuNeg = sumNeg / self.NumNeg

        direction = None
        if miuPos < miuNeg:
            direction = +1
        else:
            direction = -1

        threshold = (miuPos + miuNeg)/2

        output = [0 for i in range(self.SampleNum)]

        for i in range(self.SampleNum):
            if self._Mat[d][i] < threshold:
                output[i] = +1
            else:
                output[i] = -1

        errorRate = 0.
        for i in range(self.SampleNum):
            if output[i] != self._Tag[i]:
                errorRate += self.weight[i]

        return errorRate, threshold, direction

    def train(self):

        for dem in range(self.SampleDem):
            err, threshold, direction = self.optimal(dem)
            if err < self.opt_errorRate:
                self.opt_errorRate = err
                self.opt_demention = dem
                self.opt_threshold = threshold
                self.opt_direction = direction

        assert self.opt_errorRate < 0.5

        return self.opt_errorRate

    def prediction(self, Mat):
        SampleNum = Mat.shape[1]

        dem       = self.opt_demention
        threshold = self.opt_threshold
        direction = self.opt_direction

        output = numpy.array([0 for i in range(SampleNum)])

        for i in range(SampleNum):
            if direction * Mat[dem][i] < direction * threshold:
                output[i] = +1
            else:
                output[i] = -1

        return output

    def show(self):

        dem = self.opt_demention

        N = 10
        MaxVal = numpy.max(self._Mat[dem]) 
        MinVal = numpy.min(self._Mat[dem])

        scope = (MaxVal - MinVal) / N

        centers = [ (MinVal - scope/2)+ scope*i for i in range(N)]
        counter = [ [0, 0] for i in range(N)]

        for j in range(N):
            for i in range(self.SampleNum):
                if abs(self._Mat[dem][i] - centers[j]) < scope/2:
                    if self._Tag[i] == LABEL_POSITIVE:
                        counter[j][1] += 1
                    else:
                        counter[j][0] += 1

        posVal, negVal = [], []

        for i in range(N):
            posVal.append(counter[i][1])
            negVal.append(counter[i][0])

        pyplot.plot(centers, posVal, "r-")
        pyplot.plot(centers, negVal, "b-")

        pyplot.show()

    def __str__(self):

        string  = "opt_errorRate:" + str(self.opt_errorRate) + "\n"
        string += "opt_threshold:" + str(self.opt_threshold) + "\n"
        string += "opt_demention:" + str(self.opt_demention) + "\n"
        string += "opt_direction:" + str(self.opt_direction) + "\n"
        string += "weights      :" + str(self.weight)        + "\n"
        return string

    def constructor(self, demention, direction, threshold):
        self.opt_demention = demention
        self.opt_threshold = threshold
        self.opt_direction = direction

        return self
