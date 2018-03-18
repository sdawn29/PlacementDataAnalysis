import numpy as np
import pylab as pl
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB


def plot_classification_results(clf, X, y, title, f1, f2):
    # Divide dataset into training and testing parts
    # X_train, X_test, y_train, y_test = train_test_split(
    # X, y, test_size=0.2)

    # Fit the data with classifier.
    clf.fit(X, y)

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    h = .02  # step size in the mesh
    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure()
    pl.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.title(title)
    pl.xlabel('12th Marks')
    pl.ylabel('CGPA')

    pred = clf.predict([[int(f1),int(f2)]])
    if pred == 1:
        score = "Placed"
    else:
        score = "Not Placed"
    pl.show()
    return score

xs = pd.read_csv('data_final.csv',sep=",", header=0, usecols=[0,1])
xs = xs.as_matrix()
ys = pd.read_csv('data_final.csv',sep=",", header=0, usecols=[2])
ys = ys.values
ys = ys.ravel()

clf = GaussianNB()

feature1 = input("Enter 12th Marks: ")
feature2 = input("Enter CGPA: ")
score = plot_classification_results(clf, xs, ys, "Scatter Plot !2th Marks v/s CGPA", feature1, feature2)
print("Placement Outcome: ", score)