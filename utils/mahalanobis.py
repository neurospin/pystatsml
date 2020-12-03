# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:09:56 2016

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

'''
Mahalanobis distance
====================
'''

from matplotlib.patches import Ellipse
def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)

x = np.array([2, 2])

plt.scatter(X0[:, 0], X0[:, 1], color='b')
plt.scatter(X1[:, 0], X1[:, 1], color='r')
plt.scatter(mean0[0], mean0[1], color='b', s=200, label="m0")
plt.scatter(mean1[0], mean1[1], color='r', s=200, label="m2")
plt.scatter(x[0], x[1], color='k', s=200, label="x")
plot_cov_ellipse(Cov, pos=mean0, facecolor='none', linewidth=2, edgecolor='b')
plot_cov_ellipse(Cov, pos=mean1, facecolor='none', linewidth=2, edgecolor='r')
plt.legend(loc='upper left')

#
d2_m0x = scipy.spatial.distance.euclidean(mean0, x)
d2_m0m2 = scipy.spatial.distance.euclidean(mean0, mean1)

Covi = scipy.linalg.inv(Cov)
dm_m0x = scipy.spatial.distance.mahalanobis(mean0, x, Covi)
dm_m0m2 = scipy.spatial.distance.mahalanobis(mean0, mean1, Covi)

print('Euclidean dist(m0, x)=%.2f > dist(m0, m2)=%.2f' % (d2_m0x, d2_m0m2))
print('Mahalanobis dist(m0, x)=%.2f < dist(m0, m2)=%.2f' % (dm_m0x, dm_m0m2))


'''
## Exercise

- Write a function `euclidean(a, b)` that compute the euclidean distance
- Write a function `mahalanobis(a, b, Covi)` that compute the euclidean
  distance, with the inverse of the covariance matrix. Use `scipy.linalg.inv(Cov)`
  to invert your matrix.
'''
def euclidian(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def mahalanobis(a, b, cov_inv):
    return np.sqrt(np.dot(np.dot((a - b), cov_inv),  (a - b).T))

assert mahalanobis(mean0, mean1, Covi) == dm_m0m2
assert euclidian(mean0, mean1)  == d2_m0m2

mahalanobis(X0, mean0, Covi)
X = X0
mean = mean0
covi= Covi

np.sqrt(np.dot(np.dot((X - mean), covi),  (X - mean).T))

def mahalanobis(X, mean, covi):
    """
    from scipy.spatial.distance import mahalanobis
    d2= np.array([mahalanobis(X[i], mean, covi) for i in range(X.shape[0])])
    np.all(mahalanobis(X, mean, covi) == d2)
    """
    return np.sqrt(np.sum(np.dot((X - mean), covi) *  (X - mean), axis=1))

