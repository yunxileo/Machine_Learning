"""
Programmer  :   EOF
Cooperator  :   Wei Chen.
Date        :   2015.11.22
File        :   adaboost.py

File Description:
	AdaBoost is a machine learning meta-algorithm. 
That is the short for "Adaptive Boosting".

Thanks Wei Chen. Without him, I can't understand AdaBoost in this short time. We help each other and learn this algorithm.

"""
import numpy

from config import *
from weakClassifier import WeakClassifier

if DEBUG_MODEL == True:
    import time
    from matplotlib import pyplot
    from haarFeature import Feature


class AdaBoost:

    def __init__(self, Mat = None, Tag = None, WeakerClassifier = None, train = True):
        """
        self._Mat: A matrix which store the samples. Every column 
                   vector in this matrix is a point of sample.
        self._Tag: 
    	self.W: A vecter which is the weight of weaker classifier
    	self.N: A number which descripte how many weaker classifier
    		is enough for solution.
	"""
        if train == True:
            self._Mat = numpy.array(Mat) * 1.0
            self._Tag = numpy.array(Tag) * 1.0

            self.SamplesDem = self._Mat.shape[0]
            self.SamplesNum = self._Mat.shape[1]

            # Make sure that the inputed data's demention is right.
            assert self.SamplesNum == self._Tag.size


            # Initialization of weight
            pos_W = [1.0/(2 * POSITIVE_SAMPLE) for i in range(POSITIVE_SAMPLE)]

            neg_W = [1.0/(2 * NEGATIVE_SAMPLE) for i in range(NEGATIVE_SAMPLE)]
            self.W = pos_W + neg_W

            self.accuracy = []

        self.Weaker = WeakClassifier

        self.G = {}
        self.alpha = {}
        self.N = 0
        self.detectionRate = 0.

        # true positive rate
        self.tpr = 0.
        # false positive rate
        self.fpr = 0.


    def is_good_enough(self):

        output = self.prediction(self._Mat)

        correct = numpy.count_nonzero(output == self._Tag)/(self.SamplesNum*1.) 
        self.accuracy.append( correct)

        self.detectionRate = numpy.count_nonzero(output[0:POSITIVE_SAMPLE] == 1) * 1./ POSITIVE_SAMPLE

        if self.accuracy[self.N-1] > AB_EXPECTED_DETECTION_RATE\
            and self.detectionRate > AB_EXPECTED_ACCURACY:
            return True

    def train(self, M = 4):
	"""
	function @train() is the main process which run 
	AdaBoost algorithm.

	@M : Upper bound weaker classifier. How many weaker 
        classifier will be used to construct a strong 
	classifier.
	"""

        if DEBUG_MODEL == True:
            adaboost_start_time = time.time()

        for m in range(M):
            self.N += 1

            if DEBUG_MODEL == True:
                weaker_start_time = time.time()

            self.G[m] = self.Weaker(self._Mat, self._Tag, self.W)
            
            errorRate = self.G[m].train()

            if DEBUG_MODEL == True:
                print "Time for training WeakClassifier:", \
                        time.time() - weaker_start_time

            self.alpha[m] = numpy.log((1-errorRate)/errorRate)

            output = self.G[m].prediction(self._Mat)
            
            if self.is_good_enough():
                print (self.N) ," weak classifier is enough to ",
                print "meet the request which given by user."
                print "Training Done :)"
                break

            Z = 0.0
            for i in range(self.SamplesNum):
                Z += self.W[i] * numpy.exp(-self.alpha[m] * self._Tag[i] * output[i])

            for i in range(self.SamplesNum):
                self.W[i] = (self.W[i] / Z) * numpy.exp(-self.alpha[m] * self._Tag[i] * output[i])

            if DEBUG_MODEL == DEBUG_MODEL:
                print "weakClassifier:", self.N
                print "errorRate     :", errorRate
                print "accuracy      :", self.accuracy[-1]
                print "detectionRate :", self.detectionRate


        if DEBUG_MODEL == True:
            self.showErrRates()
            print "The time cost of training this AdaBoost model:",\
                    time.time() - adaboost_start_time


    def prediction(self, Mat, threshold = 0):

        Mat = numpy.array(Mat)

        output = numpy.array([ 0. for i in range(Mat.shape[1])])

        for i in range(self.N):
            output += self.G[i].prediction(Mat) * self.alpha[i]

        for i in range(len(output)):
            if output[i] > threshold:
                output[i] = LABEL_POSITIVE
            else:
                output[i] = LABEL_NEGATIVE

        return output

    def showErrRates(self):
        pyplot.title("The changes of accuracy (Figure by Jason Leaster)")
        pyplot.xlabel("Iteration times")
        pyplot.ylabel("Accuracy of Prediction")
        pyplot.plot([i for i in range(self.N)], self.accuracy, '-.', label = "Accuracy * 100%")

        pyplot.show()

    def showROC(self):
        tprs = []
        fprs = []
        for t in numpy.arange(AB_TH_MIN, AB_TH_MAX, 0.01):
            output = self.prediction(self._Mat, t)

            Num_tp = 0 # Number of true positive
            Num_fn = 0 # Number of false negative
            Num_tn = 0 # Number of true negative
            Num_fp = 0 # Number of false positive
            for i in range(self.SamplesNum):
                if self._Tag[i] == LABEL_POSITIVE:
                    if output[i] == LABEL_POSITIVE:
                        Num_tp += 1
                    else:
                        Num_fn += 1
                else:
                    if output[i] == LABEL_POSITIVE:
                        Num_fp += 1
                    else:
                        Num_tn += 1

            tprs.append(Num_tp * 1./(Num_tp + Num_fn))
            fprs.append(Num_fp * 1./(Num_tn + Num_fp))


        pyplot.title("The ROC curve")
        pyplot.plot(fprs, tprs, "-r", linewidth = 3)
        pyplot.axis([-0.02, 1.1, 0, 1.1])
        pyplot.show()

    def makeClassifierPic(self):
        IMG_WIDTH  = 19
        IMG_HEIGHT = 19

        haar = Feature(SEARCH_WIN_WIDTH, SEARCH_WIN_HEIGHT, IMG_WIDTH, IMG_HEIGHT)

        featuresAll = haar.features
        selFeatures = [] # selected features

        for n in range(self.N):
            selFeatures.append(featuresAll[self.G[n].opt_demention])

        
        classifierPic = numpy.array([[0 for i in range(IMG_WIDTH)] for j in range(IMG_HEIGHT)])

        for n in range(self.N):
            feature   = featuresAll[n]
            alpha     = self.alpha[n]
            direction = self.direction[n]

            (types, x, y, width, height) = [val for val in feature]

            image = numpy.array([[0 for i in range(IMG_WIDTH)] for j in range(IMG_HEIGHT)])

            assert x >= 0 and x < self.Row
            assert y >= 0 and y < self.Col
            assert width > 0 and height > 0

            if types == "I":
                for i in range(y, y + height * 2):
                    for j in range(x, x + width):
                        if i < y + height:
                            image[i][j] = 255
                        else:
                            image[i][j] = 0

            elif types == "II":
                for i in range(y, y + height):
                    for j in range(x, x + width * 2):
                        if j < x + width:
                            image[i][j] = 0
                        else:
                            image[i][j] = 255

            elif types == "III":
                for i in range(y, y + height):
                    for j in range(x, x + width * 3):
                        if j >= (x + width) and j < (x + width * 2):
                            image[i][j] = 255
                        else:
                            image[i][j] = 0

            elif types == "IV":
                for i in range(y, y + height * 2):
                    for j in range(x, x + width * 2):
                        if (j < x + width and i < y + height) or\
                           (j >= x + width and i >= y + height):
                            image[i][j] = 0
                        else:
                            image[i][j] = 255

            classifierPic += image * alpha * direction

        pylab.matshow(classifierPic, cmap = "Greys")
        pylab.show()
            
            
