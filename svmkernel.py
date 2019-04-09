import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

#create a mesh of points to plot in
def make_meshgrid(x,y,h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

#Plot the decision boundaries for a classifier
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    out =  ax.contourf(xx, yy, Z, **params)
    return out

#import some data to model
iris = datasets.load_iris()
#Take the firs two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
Y = iris.target

# We create an instance of SVM and fit our data.
# Data is not scaled because support vectors will be plotted
C = 1.0 # SVM regularization parameter

# Setting up the different kernels stored in the models array
models = (svm.SVC(kernel = 'linear', C = C),
          svm.LinearSVC(C = C),
          svm.SVC(kernel = 'rbf', gamma = 0.7, C = C),
          svm.SVC(kernel = 'poly', degree = 3, C = C))

# Loop through and train
models = (clf.fit(X, Y) for clf in models)

# Add plot titles
titles = ('SVC with Linear kernel','LinearSVC (linear kernel)','SVC with RBF kernel','SVc with polynomial (degree3) kernel')

# set up 2x2 grid for plotting
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace = 0.4, hspace = 0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy, cmap = plt.cm.coolwarm, alpha = 0.8)
    ax.scatter(X0, X1, c = 'y', cmap = plt.cm.coolwarm, s = 20, edgecolors = 'k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
