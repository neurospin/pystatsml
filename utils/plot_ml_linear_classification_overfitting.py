#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 23:25:38 2020

@author: ed203246
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # nicer plots
import sklearn.metrics as metrics
import sklearn.linear_model as lm

import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

# %% Plot train/test with inreasing size

def logistic(x): return 1 / (1 + np.exp(-x))

def fit_on_increasing_size(model):
    n_samples = 100
    n_features_ = np.arange(20, 2000, 100)
    bacc_train, bacc_test = [], []
    for n_features in n_features_:
        n_features_info = int(n_features / 10)
        X, y = datasets.make_classification(n_samples=n_samples * 2, n_features=n_features,
                                     n_informative=n_features_info, n_redundant=int(n_features_info / 2),
                                     n_classes=2,
                                     n_clusters_per_class=1,
                                     weights=None, flip_y=0.01,
                                     class_sep=.5,
                                     hypercube=True, shift=0.0, scale=1.0, shuffle=True,
                                     random_state=1)
        """
        # Sample the dataset (* 2 nb of samples)
        n_features_info = int(n_features / 10)
        np.random.seed(27)  # Make reproducible 27
        X = np.random.randn(n_samples * 2, n_features)
        beta = np.zeros(n_features)
        beta[:n_features_info] = 1
        Xbeta = np.dot(X, beta)
        eps = np.random.randn(n_samples * 2)
        proba = logistic(Xbeta + eps)
        y =  (proba >= 0.5).astype(int)
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
        # fit/predict
        mod.fit(X_train, y_train)
        y_pred_train = mod.predict(X_train)
        y_pred_test = mod.predict(X_test)
        #snr.append(Xbeta.std() / eps.std())
        bacc_train.append(metrics.balanced_accuracy_score(y_train, y_pred_train))
        bacc_test.append(metrics.balanced_accuracy_score(y_test, y_pred_test))
    return n_features_, np.array(bacc_train), np.array(bacc_test)

def plot_bacc(n_features_, bacc_train, bacc_test, xvline, ax, title):
    """
    Two scales plot. Left y-axis: train test r-squared. Right y-axis SNR.
    """
    ax.plot(n_features_, bacc_train, label="Train Acc", linewidth=2, color=sns.color_palette()[0])
    ax.plot(n_features_, bacc_test, label="Test Acc", linewidth=2, color=sns.color_palette()[1])
    ax.axvline(x=xvline, linewidth=2, color='k', ls='--')
    ax.fill_between(n_features_, bacc_test, 0.5, alpha=.3, color=sns.color_palette()[1])
    ax.fill_between(n_features_, bacc_test, bacc_train, alpha=.3, color=sns.color_palette()[0])
    ax.axhline(y=0.5, linewidth=1, color='k', ls='--')
    ax.set_ylim(0.3, 1.1)
    ax.set_ylabel("r2", fontsize=16)
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title(title, fontsize=20)

# plot
fig, axis = plt.subplots(4, 1, figsize=(9, 12), sharex=True)


# %% No regularization

#fig, axis = plt.subplots(1, 1, figsize=(9, 12), sharex=True)
#fig, axis = plt.subplots(1, 1, figsize=(9, 9), sharex=True)

#mod = lm.LogisticRegression(penalty='none')
mod = lm.LogisticRegression(penalty='l2', C=.1e16) # lambda = 1 / C!

n_features, bacc_train, bacc_test = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(bacc_test)]
plot_bacc(n_features, bacc_train, bacc_test, argmax, axis[0], 'Regression')

# %% L2 regularization

mod = lm.LogisticRegression(penalty='l2', C=1e-2) # lambda = 1 / C!
n_features, bacc_train, bacc_test = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(bacc_test)]
plot_bacc(n_features, bacc_train, bacc_test, argmax, axis[1], 'Ridge')

# %% L1 regularization

mod = lm.LogisticRegression(penalty='l1', C=.1, solver='saga') # lambda = 1 / C!
n_features, bacc_train, bacc_test = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(bacc_test)]
plot_bacc(n_features, bacc_train, bacc_test, argmax, axis[2], 'Lasso')


# %% L1-L2 regularization

mod = lm.LogisticRegression(penalty='elasticnet',  C=.1, l1_ratio=0.5, solver='saga')
n_features, bacc_train, bacc_test = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(bacc_test)]
plot_bacc(n_features, bacc_train, bacc_test, argmax, axis[3], 'ElasticNet')



plt.tight_layout()
axis[3].set_xlabel("Number of input features", fontsize=16)
#plt.savefig("/home/ed203246/git/pystatsml/images/linear_classification_penalties.png")

# %% Codes examples:

if False:
    from sklearn import datasets
    import  sklearn.linear_model as lm

    X, y = datasets.make_regression(n_features=5, n_informative=2, random_state=0)

    lr = lm.LinearRegression().fit(X, y)

    l2 = lm.Ridge(alpha=10).fit(X, y)  # lambda is alpha!
    print(l2.coef_)

    l1 = lm.Lasso(alpha=1).fit(X, y)  # lambda is alpha !
    print(l1.coef_)

    l1l2 = lm.ElasticNet(alpha=1, l1_ratio=.9).fit(X, y)
