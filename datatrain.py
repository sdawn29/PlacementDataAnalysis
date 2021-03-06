import numpy as np
import pylab as pl
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.naive_bayes import GaussianNB

import io
import base64
import time
from flask import Flask, request
app = Flask(__name__)

def plot_classification_results(clf, X, y, f1, f2):
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
    pl.title("Scatter Plot 12th Marks v/s CGPA")
    pl.xlabel('12th Marks')
    pl.ylabel('CGPA')

    png = io.BytesIO()
    pl.savefig(png, format='png')

    pred = clf.predict([[int(f1),int(f2)]])
    return png, pred

xs = pd.read_csv('data_final.csv',sep=",", header=0, usecols=[0,1])
xs = xs.as_matrix()
ys = pd.read_csv('data_final.csv',sep=",", header=0, usecols=[2])
ys = ys.values
ys = ys.ravel()

clf = GaussianNB()

@app.route("/plot")
def plot_image():
    twelve_r = request.args.get('twelve')
    cgpa_r = request.args.get('cgpa')

    twelve = float(twelve_r)
    cgpa = float(cgpa_r)

    png, pred = plot_classification_results(clf, xs, ys, twelve, cgpa)

    png.seek(0)
    encoded = base64.b64encode(png.getvalue()).decode()
    template = """
<center>
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">

    <title>Source Code</title>
  </head>
  <body>
    <h3 class="display-5">{}</h3>
    <br>
    <br>
    <b>Training Model</b>
    <br>
    {}
    <br>
    <br>
    <button class="btn btn-danger" onclick="window.close();"> Close Window </button>
    </body>
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
  </html>
</center>
"""
    res = "NOT Placed" if pred == 0 else "Placed"
    t = '<img src="data:image/png;base64,{}">'.format(encoded)
    r = template.format(res, t)

    print(pred)
    return r

if __name__ == '__main__':
    app.debug = True
    app.run()
