'''
UNIVERSITY OF ST ANDREWS
CS5014 - MACHINE LEARNING

PRACTICAL P2 - CLASSIFICATION OF OBJECT COLOUR USING OPTICAL SPECTROSCOPY DATA

STUDENT ID: 170027939

@ LogisticClassifier.py
-Plots and saves logistic model charts
-Classifies data using a logistic regression with a linear model
-Saves predicted outputs to CSV File format
-Computes accuracy score, log loss, confusion matrix and a classification report


'''
import pip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
# Classification == Logistic Reg only
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import _logistic_loss
# Import metrics
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.svm import SVC


# Calculate loss at x
def loss_model(x):
    return (1/(1 + np.exp(-x)))


# Plot simple scatter charts
def _plot_and_save_chart(x_list, y_list, x_label, y_label, title, marker, logistic_model):
    plt.scatter(x_list, y_list, label=title, color='k', s=100, marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #_print_current_state(loss, "Computing Loss")
    #plt.scatter(x_test_list, loss, color='red', linewidth=1, marker=marker)

    plt.show()
    # Add syntax for chart saving


def _plot_logistic_loss_function(x_test_list, logistic_model, ):


    plt.show()

def _print_current_state(list, current_state):
    print("\n\r")
    print("\n\rSTATUS: ", str(current_state))
    print(list)


# Computes Logistic Models
def _compute_logistic_model(x_training_data, y_training_data, x_test_data):
    model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
    model.fit(x_training_data, y_training_data.values.ravel())

    # Look into refactoring
    x_try = pd.DataFrame(x_training_data).mean(axis=1)
    print(x_try.shape)
    print(y_training_data[0].shape)
    newreg = sb.regplot(x_try, y_training_data[0])
    newreg.set_title("Optical Reflectance V Colour Classification with Regression")
    newreg.set_xlabel("Average Optical Reflectance")
    newreg.set_ylabel("Colour Class")

    plt.show()

    # Compute decision function [May or may not be used/ better using the probability decision matrix]
    model_decision_function = model.decision_function(x_test_data)
    _print_current_state(model_decision_function, "Decision Function")


    # Computer model accuracy and score
    # test model using training data
    model_training_prediction_probability_matrix = model.predict_proba(x_training_data)
    _print_current_state(model_training_prediction_probability_matrix, "Training Prediction Probability ")
    y_training_predicted_data = model.predict(x_training_data)
    _print_current_state(y_training_predicted_data, "Testing on Training Set")
    model_training_accuracy = accuracy_score(y_training_data, y_training_predicted_data)
    _print_current_state(model_training_accuracy, "Training Accuracy Score")
    model__training_score = model.score(x_training_data, y_training_data)


    # Compute prediction on test data
    y_predicted_data = model.predict(x_test_data)

    #print(type(y_actual_prob))
    #y_actual_prob = pd.DataFrame(y_actual_prob)
    _print_current_state(y_predicted_data, "Predicting actual outputs based Given Test Data")

    # Compute log loss
    y_actual_prob = model.predict_proba(x_test_data)
    _print_current_state(y_actual_prob, "actual loss")

    # Compute final score of model on test data
    # Used only in supervised classification
    model_output_prediction_matrix = model.predict_proba(x_test_data)
    model_output_score = model.score(x_test_data, y_predicted_data)


    # Plot Charts
    x_test_data_mean = pd.DataFrame(x_test_data.mean(axis=1))
    _print_current_state(None,"Plotting Charts")
    _plot_and_save_chart(x_test_data_mean, y_predicted_data, "Average Optical Reflectance", "Colour Class", "XvY", "x", model)
    #_plot_and_save_chart(x_test_data_mean, y_predicted_data, "Average X", "Average Y", "XvY", "x", model)

    _print_current_state(classification_report(y_training_predicted_data, y_training_data), "Printing Cassification Report")
    _print_current_state(log_loss(y_training_data, model.predict_proba(x_training_data)), "Printing log loss")
    _print_current_state(confusion_matrix(y_training_data, model.predict(x_training_data)),
                            "Computing Confusion Matrix")


    # Save Dataframe to CSV
    y_predicted_data = pd.DataFrame(y_predicted_data)
    y_predicted_data.to_csv("/Users/negus/PycharmProjects/CS5014/P2/Logistic_PredictedClasses_.csv", sep='\t', encoding='utf-8')













