#Importing required packages
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

#Import iris data
iris = datasets.load_iris()
X = iris.data[:, :4]
y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot Sepal Length and Sepal Width
a = plt.figure(1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
a.show()

# Plot Sepal Length and Petal Length
b = plt.figure(2)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
plt.scatter(X[:, 0], X[:, 2], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
b.show()

# Plot Sepal Length and Petal Width
c = plt.figure(3)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
plt.scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Sepal Length')
plt.ylabel('Petal Width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
c.show()

# Plot Sepal Width and Petal Length
d = plt.figure(4)
x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
y_min, y_max = X[:, 2].min() - .5, X[:, 2].max() + .5
plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Sepal Width')
plt.ylabel('Petal Length')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
d.show()

# Plot Sepal Width and Petal Width
e = plt.figure(5)
x_min, x_max = X[:, 1].min() - .5, X[:, 1].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
plt.scatter(X[:, 1], X[:, 3], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
e.show()

# Plot Petal Length and Petal Width
f = plt.figure(6)
x_min, x_max = X[:, 2].min() - .5, X[:, 2].max() + .5
y_min, y_max = X[:, 3].min() - .5, X[:, 3].max() + .5
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.colorbar()
plt.xlabel('Petal Lengh')
plt.ylabel('Petal Width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
f.show()
input()

