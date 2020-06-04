#Importing required packages
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

#Importing iris data
data_folder = Path("/opt/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data")
filei = data_folder / "iris.csv"
iris = pd.read_csv(filei,sep=',')

X = iris.drop(['virginica'], axis = 1)
y = iris['virginica']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

bold_start = '\033[1m'
bold_end   = '\033[0m'

#Random Forest Classifier Classification Report and Confusion Matrix
pipeRfc = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 200))
pipeRfc.fit(X_train, y_train)
pred_rfc = pipeRfc.predict(X_test)
print(bold_start, "Random Forest Classifier", bold_end)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
print("")

#Support Vector Model Classifier Classification Report and Confusion Matrix
pipeClf = make_pipeline(StandardScaler(), svm.SVC())
pipeClf.fit(X_train, y_train)
pred_clf = pipeClf.predict(X_test)
print(bold_start, "Support Vector Model Classifier", bold_end)
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))
print("")

#Multi-layer Perceptron Classifier Classification Report and Confusion Matrix
pipeMlpc = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes = (11,11,11), max_iter = 1500))
pipeMlpc.fit(X_train, y_train)
pred_mlpc = pipeMlpc.predict(X_test)
print(bold_start, "Multi-layer Perceptron Classifier", bold_end)
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))

#AdaBoost Classifier Classification Report and Confusion Matrix
pipeAda = make_pipeline(StandardScaler(), AdaBoostClassifier())
pipeAda.fit(X_train, y_train)
pred_Ada = pipeAda.predict(X_test)
print(bold_start, "AdaBoost Classifier", bold_end)
print(classification_report(y_test, pred_Ada))
print(confusion_matrix(y_test, pred_Ada))
print("")

#Naive Bayes Classifier Classification Report and Confusion Matrix
pipeNb = make_pipeline(StandardScaler(), GaussianNB())
pipeNb.fit(X_train, y_train)
pred_Nb = pipeNb.predict(X_test)
print(bold_start, "Naive Bayes Classifier", bold_end)
print(classification_report(y_test, pred_Nb))
print(confusion_matrix(y_test, pred_Nb))
print("")
