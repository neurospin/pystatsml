# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:23:23 2016

@author: edouard.duchesnay@cea.fr
"""

'''
Fisher's linear discriminant
============================
'''

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

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
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, label="toto", **kwargs)

    ax.add_artist(ellip)
    return ellip


def fisher_lda(X, y):
    mean0_hat, mean1_hat = X[y == 0].mean(axis=0),  X[y == 1].mean(axis=0)
    Xcentered = np.vstack([(X[y == 0] - mean0_hat), (X[y == 1] - mean1_hat)])
    Cov_hat = np.cov(Xcentered.T)
    beta = np.dot(np.linalg.inv(Cov_hat), (mean1 - mean0))
    beta /= np.linalg.norm(beta)
    thres = 1 / 2 * np.dot(beta, (mean1 - mean0))
    return beta, thres, mean0_hat, mean1_hat, Cov_hat

def plot_linear_disc(beta, thres, X, y, Cov_hat=None):
    # Threshold coordinate. xy of the point equi-distant to m0, m1
    thres_xy = thres * beta
    # vector supporting the seprating hyperplane 
    sep_vec = np.array([beta[1], -beta[0]])
    # Equation of seprating hyperplane
    a = np.arctan(sep_vec[1] / sep_vec[0])
    b = thres_xy[1] - a * thres_xy[0]
    xmin, xmax = np.min(X, axis=0)[0], np.max(X, axis=0)[0]
    ymin = a * xmin + b
    ymax = a * xmax + b
    sep_p1_xy = [xmin, ymin]
    sep_p2_xy = [xmax, ymax]
    # Plot
    err = plt.scatter(X[errors, 0], X[errors, 1], color='k', marker="x", s=100, lw=2)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color=palette[0])
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color=palette[1])
    m1 = plt.scatter(mean0_hat[0], mean0_hat[1], color=palette[0], s=200, label="m0")
    m2 = plt.scatter(mean1_hat[0], mean1_hat[1], color=palette[1], s=200, label="m2")
    plot_cov_ellipse(Cov_hat, pos=mean0_hat, facecolor='none', linewidth=2, edgecolor=palette[3], ls='-')
    Sw = plot_cov_ellipse(Cov_hat, pos=mean1_hat, facecolor='none', linewidth=2, edgecolor=palette[3], ls='-')
    # Projection vector
    proj = plt.arrow(thres_xy[0], thres_xy[1], beta[0], beta[1], fc="k", ec="k", head_width=0.2, head_length=0.2, linewidth=2)
    # Points along the separating hyperplance
    hyper = plt.plot([sep_p1_xy[0], sep_p2_xy[0]], [sep_p1_xy[1], sep_p2_xy[1]], color='k', linewidth=4, ls='--')
    plt.axis('equal')
    #plt.legend([m1, m2, Sw, proj, err], ['$\mu_0$', '$\mu_1$', '$S_W$', "$w$", 'Errors'], loc='lower right', fontsize=18)

# Dataset
n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

# Fisher LDA
beta, thres, mean0_hat, mean1_hat, Cov_hat = fisher_lda(X, y)

y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred != y 
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred)))

#%matplotlib inline
%matplotlib qt

from matplotlib import rc
plt.rc('text', usetex=True)
font = {'family' : 'serif'}
plt.rc('font', **font)
palette = sns.color_palette()

fig = plt.figure(figsize=(7, 7))
plot_linear_disc(beta, thres, X, y)

# RGBA S_W 8172b2ff
# RGBA S_B c44e52ff
proj = np.dot(X, beta)

# Fisher projection
plt.figure(figsize=(np.sqrt(2 * 7 ** 2), 2))
for lab in np.unique(y_true):
    sns.distplot(proj.ravel()[y == lab], label=str(lab))

plt.figure(figsize=(7, 2))
for lab in np.unique(y_true):
    sns.distplot(X[y == lab, 0], label=str(lab))

plt.figure(figsize=(7, 2))
for lab in np.unique(y_true):
    sns.distplot(X[y == lab, 1], label=str(lab))

'''
Linear discriminant analysis (LDA)
==================================
'''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Dataset
n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

# LDA with scikit-learn
lda = LDA()
proj = lda.fit(X, y).transform(X)
y_pred = lda.predict(X)

errors =  y_pred != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred)))

# Use pandas & seaborn for convinience
data = pd.DataFrame(dict(x0=X[:, 0], x1=X[:, 1], y=["c"+str(v) for v in y]))
plt.figure()
g = sns.PairGrid(data, hue="y")
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
g.add_legend()



plt.figure()
for lab in np.unique(y):
    sns.distplot(proj.ravel()[y == lab], label=str(lab))

plt.legend()
plt.title("Distribution of projected data using LDA")


'''
Logistic regression
'''
from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=1e8)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, y)
y_pred_logreg = logreg.predict(X)

errors =  y_pred_logreg != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_logreg)))
print(logreg.coef_)

'''
Ridge Fisher's linear discriminant
==================================
'''
#%matplotlib inline
#%matplotlib qt

# Dataset
n_samples, n_features = 5, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(45)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
# modify X1 to distrub the etimation of cov 
X1[2, :] = [2, -4]

X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

def ridge_fisher_lda(X, y, lambda_):
    mean0_hat, mean1_hat = X[y == 0].mean(axis=0),  X[y == 1].mean(axis=0)
    Xcentered = np.vstack([(X[y == 0] - mean0_hat), (X[y == 1] - mean1_hat)])
    Cov_hat = np.cov(Xcentered.T) + lambda_ * np.identity(2)
    beta = np.dot(np.linalg.inv(Cov_hat), (mean1 - mean0))
    beta /= np.linalg.norm(beta)
    thres = 1 / 2 * np.dot(beta, (mean1 - mean0))
    return beta, thres, mean0_hat, mean1_hat, Cov_hat

plt.figure(figsize=(15, 5)) 

# Fisher LDA
plt.subplot(131)
beta, thres, mean0_hat, mean1_hat, Cov_hat = fisher_lda(X, y)
y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred != y
plot_linear_disc(beta, thres, X, y, Cov_hat=Cov_hat/np.linalg.norm(Cov_hat))
plt.title("Fisher ($\lambda=%.1f$)" % 0)

# Fisher Ridge
plt.subplot(132)
beta, thres, mean0_hat, mean1_hat, Cov_hat = ridge_fisher_lda(X, y, 1)
y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred != y
plot_linear_disc(beta, thres, X, y, Cov_hat=Cov_hat/np.linalg.norm(Cov_hat))
plt.title("Ridge Fisher ($\lambda=%.1f$)" % 1)

# Fisher Ridge
plt.subplot(133)
beta, thres, mean0_hat, mean1_hat, Cov_hat = ridge_fisher_lda(X, y, 10)
y_proj = np.dot(X, beta)
y_pred = np.asarray(y_proj > thres, dtype=int)
errors = y_pred != y 
plot_linear_disc(beta, thres, X, y, Cov_hat=Cov_hat/np.linalg.norm(Cov_hat))
plt.title("Ridge Fisher ($\lambda=%.1f$)" % 10)

'''
Penalized Logistic regression
=============================
'''

# Dataset
# Build a classification task using 3 informative features
from sklearn import datasets

X, y = datasets.make_classification(n_samples=100,
                           n_features=20,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)

'''
Ridge
-----
'''
from sklearn import linear_model
lr = linear_model.LogisticRegression(C=1)
# This class implements regularized logistic regression. C is the Inverse of regularization strength.
# Large value => no regularization.

lr.fit(X, y)
y_pred_lr = lr.predict(X)

errors =  y_pred_lr != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y)))
print(lr.coef_)

'''
Lasso
-----
'''
from sklearn import linear_model
lrl1 = linear_model.LogisticRegression(penalty='l1')
# This class implements regularized logistic regression. C is the Inverse of regularization strength.
# Large value => no regularization.

lrl1.fit(X, y)
y_pred_lrl1 = lrl1.predict(X)

errors =  y_pred_lrl1 != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_lrl1)))
print(lrl1.coef_)


'''
Linear SVM
=========
'''

'''
Ridge
-----
'''
from sklearn import svm

svmlin = svm.LinearSVC()
# Remark: by default LinearSVC uses squared_hinge as loss
svmlin.fit(X, y)
y_pred_svmlin = svmlin.predict(X)

errors =  y_pred_svmlin != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlin)))
print(svmlin.coef_)

'''
Lasso
-----
'''

from sklearn import svm

svmlinl1 = svm.LinearSVC(penalty='l1', dual=False)
# Remark: by default LinearSVC uses squared_hinge as loss

svmlinl1.fit(X, y)
y_pred_svmlinl1 = svmlinl1.predict(X)

errors =  y_pred_svmlinl1 != y
print("Nb errors=%i, error rate=%.2f" % (errors.sum(), errors.sum() / len(y_pred_svmlinl1)))
print(svmlinl1.coef_)

'''
Compare predictions of Logistic regression (LR) and their SVM counterparts, ie.: L2 LR vs L2 SVM and L1 LR vs L1 SVM
- Compute the correlation between pairs of weights vectors.
- Compare the predictions of two classifiers using their decision function: 
    * Provide the generic form of the decision function for a linear classifier, assuming that their is no intercept.
    * Compute the correlation decision function.
    * Plot the pairwise decision function of the classifiers.
- Conclude on the differences between Linear SVM and logistic regression.

'''

print(np.corrcoef(lr.coef_, svmlin.coef_))
print(np.corrcoef(lrl1.coef_, svmlinl1.coef_))
# The weights vectors are highly correlated

print(np.corrcoef(lr.decision_function(X), svmlin.decision_function(X)))
print(np.corrcoef(lrl1.decision_function(X), svmlinl1.decision_function(X)))
# The decision function are highly correlated

plt.plot(lr.decision_function(X), svmlin.decision_function(X), "o")
plt.plot(lrl1.decision_function(X), svmlinl1.decision_function(X), "o")


'''
Imbalanced classes
==================
'''
import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn import metrics
import matplotlib.pyplot as plt

# dataset
X, y = datasets.make_classification(n_samples=500,
                           n_features=5,
                           n_informative=2,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=1,
                           shuffle=False)

print(*["#samples of class %i = %i;" % (lev, np.sum(y == lev)) for lev in np.unique(y)])

# With intercept
lr_inter = linear_model.LogisticRegression(C=1, fit_intercept=True)  # default value
lr_inter.fit(X, y)
p, r, f, s = metrics.precision_recall_fscore_support(y, lr_inter.predict(X))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('# The predictions are balanced in sensitivity and specificity')

# No intercept
lr_nointer = linear_model.LogisticRegression(C=1, fit_intercept=False)
lr_nointer.fit(X, y)
p, r, f, s = metrics.precision_recall_fscore_support(y, lr_nointer.predict(X))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('# specificity ~ sensitivity')

plt.plot(lr_inter.decision_function(X), lr_nointer.decision_function(X), "o")
print('# The decision function is highly correlated. The intercept has biased upwardly the decision function.')

# Create imbalanced dataset, by subsampling sample of calss 0: keep only 10% of classe 0's samples and all classe 1's samples.
n0 = int(np.rint(np.sum(y == 0) / 20))
subsample_idx = np.concatenate((np.where(y == 0)[0][:n0], np.where(y == 1)[0]))
Ximb = X[subsample_idx, :]
yimb = y[subsample_idx]
print(*["#samples of class %i = %i;" % (lev, np.sum(yimb == lev)) for lev in np.unique(yimb)])

# With intercept
lr_inter = linear_model.LogisticRegression(C=1, fit_intercept=True)  # default value
lr_inter.fit(Ximb, yimb)
p, r, f, s = metrics.precision_recall_fscore_support(yimb, lr_inter.predict(Ximb))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('# sensitivity >> specificity')

# No intercept
lr_nointer = linear_model.LogisticRegression(C=1, fit_intercept=False)
lr_nointer.fit(Ximb, yimb)
p, r, f, s = metrics.precision_recall_fscore_support(yimb, lr_nointer.predict(Ximb))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('''# Specificity ~ sensitivity. Nevertheless the prediction
disequilibrium has been reduced.
This sugest that intercept should not be used with impbalced training dataset
when we explicitely want balanced prediction.
''')


plt.plot(lr_inter.decision_function(X), lr_nointer.decision_function(X), "o")
print('''# The decision function is no more highly correlated. 
The intercept has largly modified the learning process it is no more a 
simple upward bias on the decision function.''')

# Class reweighting + intercept
lr_inter_reweight = linear_model.LogisticRegression(C=1, fit_intercept=True,
                                                    class_weight="balanced")
lr_inter_reweight.fit(Ximb, yimb)
p, r, f, s = metrics.precision_recall_fscore_support(yimb, lr_inter_reweight.predict(Ximb))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('# The predictions are balanced in sensitivity and specificity')

# Class reweighting no intercept
lr_nointer_reweight = linear_model.LogisticRegression(C=1,  fit_intercept=False,
                                                    class_weight="balanced")
lr_nointer_reweight.fit(Ximb, yimb)
p, r, f, s = metrics.precision_recall_fscore_support(yimb, lr_nointer_reweight.predict(Ximb))
print("SPC: %.3f; SEN: %.3f" % tuple(r))
print('# The predictions are balanced in sensitivity and specificity')
