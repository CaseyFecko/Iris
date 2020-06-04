#Importing required packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import GaussianNB

#Import iris Data
data_folder = Path("/opt/anaconda3/lib/python3.7/site-packages/sklearn/datasets/data")
filei = data_folder / "iris.csv"
iris = pd.read_csv(filei,sep=',')

X = iris.drop(['virginica'], axis = 1)
y = iris['virginica']
pipeNb = make_pipeline(StandardScaler(), GaussianNB())
fig, axs = plt.subplots(2,2)

#Removing Sepal Length
X = iris.drop(['virginica', '150'], axis = 1)
y = iris['virginica']
pipeNb.fit(X, y)
pred_Nb = pipeNb.predict(X)
print(classification_report(y, pred_Nb))
print(confusion_matrix(y, pred_Nb))
print("")
plot_confusion_matrix(pipeNb, X, y, display_labels = ['Setosa','Versicolour','Virginica'], cmap='Reds', ax = axs[0,0])

axs[0,0].set_title("Removing Sepal Length")



#Removing Sepal Width
X = iris.drop(['virginica', '4'], axis = 1)
y = iris['virginica']
pipeNb.fit(X, y)
pred_Nb = pipeNb.predict(X)
print(classification_report(y, pred_Nb))
print(confusion_matrix(y, pred_Nb))
print("")
plot_confusion_matrix(pipeNb, X, y, display_labels = ['Setosa','Versicolour','Virginica'], cmap='Reds', ax = axs[0,1])
axs[0,1].set_title("Removing Sepal Width")

#Removing Petal Length
X = iris.drop(['virginica', 'setosa'], axis = 1)
y = iris['virginica']
pipeNb.fit(X, y)
pred_Nb = pipeNb.predict(X)
print(classification_report(y, pred_Nb))
print(confusion_matrix(y, pred_Nb))
print("")
plot_confusion_matrix(pipeNb, X, y, display_labels = ['Setosa','Versicolour','Virginica'], cmap='Reds', ax = axs[1,0])
axs[1,0].set_title("Removing Petal Length")

#Removing Petal Width
X = iris.drop(['virginica', 'versicolor'], axis = 1)
y = iris['virginica']
pipeNb.fit(X, y)
pred_Nb = pipeNb.predict(X)
print(classification_report(y, pred_Nb))
print(confusion_matrix(y, pred_Nb))
print("")
plot_confusion_matrix(pipeNb, X, y, display_labels = ['Setosa','Versicolour','Virginica'], cmap='Reds', ax = axs[1,1])
axs[1,1].set_title("Removing Petal Width")


plt.show()

