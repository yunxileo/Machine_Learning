"""
Programmer  :   EOF
File        :   config.py
Date        :   2016.01.06
E-mail      :   jasonleaster@163.com

Description :
    This is a configure file for this project.

"""

DEBUG_MODEL = True

TRAINING_FACE    = "./newtraining/face/"
TRAINING_NONFACE = "./newtraining/non-face/"
TEST_FACE        = "./newtraining/test/face/"
TEST_NONFACE     = "./newtraining/test/non-face/"

TEST_IMG         = "./Test/jasonleaster.jpg"

FEATURE_FILE_TRAINING = "./features/features_train.tmp"
FEATURE_FILE_TESTING  = "./features/features_test.tmp"

ADABOOST_FILE    = "./model/model.tmp"

SEARCH_WIN_HEIGHT = 19
SEARCH_WIN_WIDTH  = 19

FEATURE_TYPE_NUM = 2
FEATURE_NUM = 32746

POSITIVE_SAMPLE  = 100
NEGATIVE_SAMPLE  = 100

SAMPLE_NUM = POSITIVE_SAMPLE + NEGATIVE_SAMPLE

TESTING_POSITIVE_SAMPLE = 10
TESTING_NEGATIVE_SAMPLE = 10

LABEL_POSITIVE = +1
LABEL_NEGATIVE = -1

TESTING_SAMPLE_NUM = TESTING_NEGATIVE_SAMPLE + TESTING_POSITIVE_SAMPLE

# the expected rate of detection in adaboost model
AB_EXPECTED_DETECTION_RATE = 0.90
# the expected accuracy in adaboost model
AB_EXPECTED_ACCURACY = 0.90

# the threshold range of adaboost. (from -inf to +inf)
AB_TH_MIN = -20
AB_TH_MAX = +20

