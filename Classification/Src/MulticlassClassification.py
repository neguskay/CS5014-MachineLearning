"""
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P2 - CLASSIFICATION OF OBJECT COLOUR USING OPTICAL SPECTROSCOPY DATA

STUDENT ID: 170027939

@ MulticlassClassification.py
-Extracts data from CSV as pandas data frame
-Separates data into Training, Validation and testing data-sets
-Computes Multiclass Classification

"""


import Src.LogisticClassifier as logistic
import Src.NeuralNetClassifier as nn
import pip as pp
import numpy as np
import csv
import pandas as pd
import collections
import matplotlib
import matplotlib.pyplot as plt


import seaborn as sb
import sklearn

# Classification == Logistic Reg only
from sklearn.linear_model import LinearRegression, LogisticRegression
#from sklearn.multiclass import

# Import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# Import Data
# Do not assume first row = header, it isn't.
multi_wavelength_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/multiclass/Wavelength.csv', header=None)
multi_x_training_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/multiclass/X.csv', header=None)
multi_y_training_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/multiclass/y.csv', header=None)
multi_x_test_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/multiclass/XToClassify.csv', header=None)

# Model 1 Data
model1_multi_x_train = multi_x_training_data
model1_multi_x_test = multi_x_test_data


# Model 2 Data
model2_multi_x_train = pd.DataFrame(multi_x_training_data.mean(axis=1))
model2_multi_x_test = pd.DataFrame(multi_x_test_data.mean(axis=1))


def _compute_multiclass_classifications():
    # Compute Logistic Models
    print("*************************************")
    print("///COMPUTING MULTICLASS LOGISTIC: Model 1")
    logistic._compute_logistic_model(model1_multi_x_train, multi_y_training_data, model1_multi_x_test)

    print("*************************************")
    print("///COMPUTING MULTICLASS LOGISTIC: Model 2")
    logistic._compute_logistic_model(model2_multi_x_train, multi_y_training_data, model2_multi_x_test)

    # Compute Neural Networks Models
    print("*************************************")
    print("///COMPUTING MULTICLASS NEURAL NET: Model 1")
    nn._compute_nn_model(model1_multi_x_train, multi_y_training_data, model1_multi_x_test)

    print("*************************************")
    print("///COMPUTING MULTICLASS NEURAL NET: Model 2")
    nn._compute_nn_model(model2_multi_x_train, multi_y_training_data, model2_multi_x_test)


