from config import *
from adaboost import AdaBoost

def getCachedAdaBoost():

    fileObj = open(ADABOOST_FILE, "a+")

    print "Constructing AdaBoost from existed model data"

    tmp = fileObj.readlines()

    if len(tmp) == 0:
        raise ValueError("There is no cached AdaBoost model")

    model = AdaBoost(train = False)

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

        classifier = model.Weaker(train = False)
        classifier.constructor(demention, direction, threshold)
        model.G[i/4] = classifier
        model.alpha[i/4] = alpha
        model.N += 1

    print "Construction finished"
    fileObj.close()

    return model

