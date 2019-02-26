# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 09:54:25 2016

@author: edouard.duchesnay@cea.fr
"""

'''
SVM & Kernel methods
====================
'''
import numpy as np
from numpy.linalg import norm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
#%matplotlib inline
#%matplotlib qt



class KernDensity:
    def __init__(self, sigma=1):
        self.sigma = sigma

    def fit(self, X, y, alphas=None):
        self.X = X
        self.y = y
        if alphas is None:
            alphas = np.ones(X.shape[0])
        self.alphas = alphas

    def predict(self, X):
        y_pred = np.zeros((X.shape[0]))
        for j, x in enumerate(X):
            for i in range(self.X.shape[0]):
                #print(j, i, x)
                y_pred[j] += self.alphas[i] * self.y[i] * np.exp( - (norm(self.X[i, :] - x) ** 2) / (2 * self.sigma ** 2))
        return(y_pred)


## Plot 3D
def plot3d(coord_x, coord_y, coord_z, points, y, zlim=None, ax=None, fig=None, xylabelsize=33):
    # Plot
    from matplotlib import cm
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    z_min = np.min(coord_z) - np.max(coord_z) * 2
    ax.plot_surface(coord_x, coord_y, coord_z, rstride=2, cstride=2,
                    #vmin=Z.min(), vmax=Z.max(),
                    cmap=cm.coolwarm,
                    linewidth=1, antialiased=True)
    cset = ax.contourf(coord_x, coord_y, coord_z, zdir='z', offset=z_min-10,
                       cmap=cm.coolwarm)
    argmin = coord_x.ravel()[coord_z.argmin()], coord_y.ravel()[coord_z.argmin()]
    print("argmin", argmin)
    # add point and cross at defined point
    colors = {-1:'b', 1:'r'}
    for lev in np.unique(y):
        pts = points[y==lev, :]
        ax.plot(pts[:, 0], pts[:, 1], 'o', color=colors[lev], zs=[z_min]*pts.shape[0], ms=10)
    ax.set_xlabel(r'$x^0$', size=xylabelsize)
    ax.set_ylabel(r'$x^1$', size=xylabelsize)
    #ax.set_zlabel(r'$Kernel density$', size=xylabelsize)
    ax.set_zlim(z_min, np.max(coord_z))
    return ax, z_min, argmin


## Dataset
##########

im = np.array(
      [[ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
       [ 1.,  1.,  1.,  1.,  1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
       [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.]])

x0, y0 = np.where(im == 0)       
x1, y1 = np.where(im == 1)

X = np.column_stack([
    np.concatenate([x0, x1]),
    np.concatenate([y0, y1])])
y = np.array([-1] * len(x0) + [1] * len(x1))

xmin, xmax, ymin, ymax = 0, im.shape[0]-1, 0, im.shape[1]-1
coord_x, coord_y = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]
XX = np.column_stack([coord_x.ravel(), coord_y.ravel()])


# Kernel mapping
################

self = KernDensity(sigma=.2)
self.fit(X, y)
y_pred_kde = self.predict(XX)
coord_z_kde = y_pred_kde.reshape(coord_x.shape)
points=X

# View 2D
if False:
    plt.imshow(np.rot90(coord_z_kde), cmap=plt.cm.coolwarm, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    plt.plot(X[y==1, 0], X[y==1, 1], 'o', color='r')#, zs=[z_min], ms=20)
    plt.plot(X[y==-1, 0], X[y==-1, 1], 'o', color='b')#, zs=[z_min], ms=20)


fig = plt.figure(figsize=(30, 15)) 

ax=fig.add_subplot(121, projection='3d')
ax, z_min, argmin = plot3d(coord_x, coord_y, coord_z_kde, points=X, y=y, ax=ax, fig=fig)
plt.title(r'$x \rightarrow K(x_i, x) = \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)$', size=33)
# set camera to fixed point of view
print(ax.azim, ax.elev, ax.dist)
#(-152.49214958606902, 21.717791411042867, 10)
#ax.view_init(azim=-152, elev=21) #Reproduce view
#ax.view_init(azim=-14.1935483871, elev=29.6875, dist=10)

# SV
#####

from sklearn.svm import SVC
#1.0 / X.shape[1] 0.5
#(1/(2 *.2)) : 2.5
clf = SVC(kernel='rbf')#, gamma=1)
clf.fit(X, y) 
clf.support_vectors_.shape

print(clf.support_.shape)

np.all(X[clf.support_,:] == clf.support_vectors_)

Xsv = clf.support_vectors_
y_sv = y[clf.support_]

y_pred_svm = clf.predict(XX)
#self = KernDensity(sigma=.2)
#self.fit(X, y)
#y_pred = self.predict(XX)
coord_z_svm = y_pred_svm.reshape(coord_x.shape)

# View 2D
if False:
    plt.imshow(np.rot90(coord_z_svm), cmap=plt.cm.coolwarm, extent=[xmin, xmax, ymin, ymax], aspect='auto')
    plt.plot(Xsv[y_sv==1, 0], Xsv[y_sv==1, 1], 'o', color='r')#, zs=[z_min], ms=20)
    plt.plot(Xsv[y_sv==-1, 0], Xsv[y_sv==-1, 1], 'o', color='b')#, zs=[z_min], ms=20)



#fig = plt.figure(figsize=(15, 15)) 
ax=fig.add_subplot(122, projection='3d')
ax, z_min, argmin = plot3d(coord_x, coord_y, coord_z_svm, points=Xsv, y=y_sv, ax=ax, fig=fig)
plt.title(r'$f(x) = sign \left(\sum_{i \in SV}\alpha_i y_i \exp\left(-\frac{||x_i - x_j||^2}{2\sigma^2}\right)\right)$', size=33)
# set camera to fixed point of view
#ax.azim, ax.elev, ax.dist
#(-152.49214958606902, 21.717791411042867, 10)
#ax.view_init(azim=-152, elev=21) #Reproduce view

############

import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt

# dataset
X, y = datasets.make_classification(n_samples=10, n_features=2,n_redundant=0,
                                    n_classes=2,
                                    random_state=1,
                                    shuffle=False)
clf = SVC(kernel='rbf')#, gamma=1)
clf.fit(X, y)
print("#Errors: %i" % np.sum(y != clf.predict(X)))

clf.decision_function(X)

# Usefull internals:
# Array of support vectors
clf.support_vectors_

# indices of support vectors within original X
np.all(X[clf.support_,:] == clf.support_vectors_)


########################


from sklearn.ensemble import RandomForestClassifier 

forest = RandomForestClassifier(n_estimators = 100)
forest.fit(X, y)

print("#Errors: %i" % np.sum(y != forest.predict(X)))


