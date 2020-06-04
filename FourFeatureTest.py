#Importing required packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

#Importing iris data
data_folder = Path("/opt/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data")
filei = data_folder / "iris.csv"
iris = pd.read_csv(filei,sep=',')

X = iris.drop(['virginica'], axis = 1)
y = iris['virginica']

#Print Naive Bayes Classifier Classification Report and Confusion Matrix
pipeNb = make_pipeline(StandardScaler(), GaussianNB())
pipeNb.fit(X, y)
pred_Nb = pipeNb.predict(X)
print(classification_report(y, pred_Nb))
print(confusion_matrix(y, pred_Nb))
print("")

#Plot and Show Naive Bayes Classifier Confusion Matrix Visual
plot_confusion_matrix(pipeNb, X, y, display_labels = ['Setosa','Versicolour','Virginica'], cmap='Reds')
plt.show()
