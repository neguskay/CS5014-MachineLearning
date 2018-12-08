'''
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P2 - CLASSIFICATION OF OBJECT COLOUR USING OPTICAL SPECTROSCOPY DATA

STUDENT ID: 170027939

@ Binary Classification.py
-Extracts data from CSV as pandas data frame
-Separates data into Training, Validation and testing data-sets
-Computes Binary Classification

'''
import Src.LogisticClassifier as logistic
import Src.NeuralNetClassifier as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Classification == Logistic Reg only
from sklearn.linear_model import LogisticRegression

# Import metrics
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC



'''
Import/Load the Data
'''

# Import Data
# Do not assume first row = header, it isn't.
binary_wavelength_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/binary/Wavelength.csv', header=None)
binary_x_training_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/binary/X.csv', header=None)
binary_y_training_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/binary/y.csv', header=None)
binary_x_test_data = pd.read_csv('/Users/negus/PycharmProjects/CS5014/P2/binary/XToClassify.csv', header=None)

# print(test_data_x.values[0, 0])
# print("Wavelength")
# print(len(wavelength_data))
# print(wavelength_data.shape)
# print("Train x")
# print(len(training_data_x))
# print(training_data_x.shape)
# print("Train y")
# print(len(training_data_y))
# print(training_data_y.shape)
# print("Test x")
# print(len(test_data_x))
# print(test_data_x.shape)

#x_training_data = pd.DataFrame(x_training_data.mean(axis=1))
#    x_test_data = pd.DataFrame(x_test_data.mean(axis=1))


# Model 1 Data
model1_x_train = binary_x_training_data
model1_x_test = binary_x_test_data


# Model 2 Data
model2_x_train = pd.DataFrame(binary_x_training_data.mean(axis=1))
model2_x_test = pd.DataFrame(binary_x_test_data.mean(axis=1))



def _compute_binary_classifications():
    # Compute logistic models
    print("*************************************")
    print("///COMPUTING BINARY LOGISTIC: Model 1")
    logistic._compute_logistic_model(model1_x_train, binary_y_training_data, model1_x_test)

    print("*************************************")
    print("///COMPUTING BINARY LOGISTIC: Model 2")
    logistic._compute_logistic_model(model2_x_train, binary_y_training_data, model2_x_test)



    # Compute neural net models
    print("*************************************")
    print("///COMPUTING BINARY NEURAL NET: Model 1")
    nn._compute_nn_model(model1_x_train, binary_y_training_data, model1_x_test)

    print("*************************************")
    print("///COMPUTING BINARY NEURAL NET: Model 2")
    nn._compute_nn_model(model2_x_train, binary_y_training_data, model2_x_test)
