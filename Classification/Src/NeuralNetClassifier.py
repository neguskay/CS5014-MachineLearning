'''
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P2 - CLASSIFICATION OF OBJECT COLOUR USING OPTICAL SPECTROSCOPY DATA

STUDENT ID: 170027939

@ Neural Network Classifier.py
-Plots and saves neural network charts
-Classifies data using a neural network multilayer perceptron
-Saves predicted outputs to CSV File format
-Computes accuracy score, log loss, confusion matrix and a classification report


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Classification == Logistic Reg only
from sklearn.neural_network import MLPClassifier

# Import metrics
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.svm import SVC



# Plot simple scatter charts
def _plot_and_save_nn_chart(x_list, y_list, x_label, y_label, title, marker):
    plt.scatter(x_list, y_list, label=title, color='k', s=100, marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.legend()
    plt.show()
    # Add syntax for chart saving


def _print_current_nn_state(list, current_state):
    print("\n\r")
    print("\n\rSTATUS: ", str(current_state))
    print(list)


# Computes Logistic Models
def _compute_nn_model(x_training_data, y_training_data, x_test_data):
    model = MLPClassifier()
    model.fit(x_training_data, y_training_data.values.ravel())
    print("\n\r")
    print("Simple Neural Network Model created")

    # Compute decision function [May or may not be used/ better using the probability decision matrix]
    #model_decision_function = model.decision_function(x_test_data)
    #_print_current_nn_state(model_decision_function, "Decision Function")

    # Computer model accuracy and score
    # test model using training data
    model_training_prediction_probability_matrix = model.predict_proba(x_training_data)
    _print_current_nn_state(model_training_prediction_probability_matrix, "Training Prediction Probability ")
    y_training_predicted_data = model.predict(x_training_data)
    _print_current_nn_state(y_training_predicted_data, "Testing on Training Set")
    model_training_accuracy = accuracy_score(y_training_data, y_training_predicted_data)
    _print_current_nn_state(model_training_accuracy, "Training Accuracy Score")
    model__training_score = model.score(x_training_data, y_training_data)


    # Compute prediction on test data
    y_predicted_data = model.predict(x_test_data)
    _print_current_nn_state(y_predicted_data, "Predicting actual outputs based Given Test Data")


    # Compute final score of model on test data
    # Used only in supervised classification
    model_output_prediction_matrix = model.predict_proba(x_test_data)
    model_output_score = model.score(x_test_data, y_predicted_data)


    # Plot Charts
    x_test_data_mean = pd.DataFrame(x_test_data.mean(axis=1))
    _print_current_nn_state(None,"Plotting Charts")
    print(x_test_data_mean)
    _plot_and_save_nn_chart(x_test_data_mean, y_predicted_data, "Average Optical Reflectance", "Colour Class", "Neural", "o")
    _print_current_nn_state(classification_report(y_training_data, model.predict(x_training_data)),
                         "Printing Cassification Report")
    _print_current_nn_state(log_loss(y_training_data, model.predict_proba(x_training_data)), "Computing Log Loss")
    _print_current_nn_state(confusion_matrix(y_training_data, model.predict(x_training_data)), "Computing Confusion Matrix")


    # Save Dataframe to CSV
    y_predicted_data = pd.DataFrame(y_predicted_data)
    y_predicted_data.to_csv("/Users/negus/PycharmProjects/CS5014/P2/Neural_PredictedClasses.csv", sep='\t', encoding='utf-8')