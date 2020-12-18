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

# %% Plot train/test with inreasing size

def fit_on_increasing_size(model):
    n_samples = 100
    n_features_ = np.arange(10, 350, 20)
    r2_train, r2_test, snr = [], [], []
    for n_features in n_features_:
        # Sample the dataset (* 2 nb of samples)
        n_features_info = int(n_features / 10)
        np.random.seed(27)  # Make reproducible 27
        X = np.random.randn(n_samples * 2, n_features)
        beta = np.zeros(n_features)
        beta[:n_features_info] = .7
        Xbeta = np.dot(X, beta)
        eps = np.random.randn(n_samples * 2)
        y =  Xbeta + eps
        # Split the dataset into train and test sample
        Xtrain, Xtest = X[:n_samples, :], X[n_samples:, :]
        ytrain, ytest = y[:n_samples], y[n_samples:]
        # fit/predict
        lr = model.fit(Xtrain, ytrain)
        y_pred_train = lr.predict(Xtrain)
        y_pred_test = lr.predict(Xtest)
        snr.append(Xbeta.std() / eps.std())
        r2_train.append(metrics.r2_score(ytrain, y_pred_train))
        r2_test.append(metrics.r2_score(ytest, y_pred_test))
    return n_features_, np.array(r2_train), np.array(r2_test), np.array(snr)

def plot_r2_snr(n_features_, r2_train, r2_test, xvline, snr, ax, title):
    """
    Two scales plot. Left y-axis: train test r-squared. Right y-axis SNR.
    """
    ax.plot(n_features_, r2_train, label="Train r2", linewidth=2, color=sns.color_palette()[0])
    ax.plot(n_features_, r2_test, label="Test r2", linewidth=2, color=sns.color_palette()[1])
    ax.axvline(x=xvline, linewidth=2, color='k', ls='--')
    ax.fill_between(n_features_, r2_test, 0, alpha=.3, color=sns.color_palette()[1])
    ax.fill_between(n_features_, r2_test, r2_train, alpha=.3, color=sns.color_palette()[0])
    ax.axhline(y=0, linewidth=1, color='k', ls='--')
    ax.set_ylim(-0.2, 1.1)
    ax.set_ylabel("r2", fontsize=16)
    ax.legend(loc='best')
    ax.grid(True)
    ax.set_title(title, fontsize=20)
    ax_right = ax.twinx()
    ax_right.plot(n_features_, snr, '--', color='gray', label="SNR", linewidth=1)
    ax_right.set_ylabel("SNR", color='gray')
    for tl in ax_right.get_yticklabels():
        tl.set_color('gray')

# plot
fig, axis = plt.subplots(4, 1, figsize=(9, 12), sharex=True)


# %% No regularization

mod = lm.LinearRegression()
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(r2_test)]
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0], 'Regression')

# %% L2 regularization

mod = lm.Ridge(alpha=10)  # lambda is alpha!
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(r2_test)]
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[1], 'Ridge')

# %% L1 regularization

mod = lm.Lasso(alpha=.1)  # lambda is alpha !
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(r2_test)]
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[2], 'Lasso')


# %% L1-L2 regularization

mod = lm.ElasticNet(alpha=.5, l1_ratio=.5)
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)
argmax = n_features[np.argmax(r2_test)]
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[3], 'ElasticNet')



plt.tight_layout()
axis[3].set_xlabel("Number of input features", fontsize=16)
plt.savefig("/home/ed203246/git/pystatsml/images/linear_regression_penalties.png")

# %% Codes examples:

from sklearn import datasets
import  sklearn.linear_model as lm

X, y = datasets.make_regression(n_features=5, n_informative=2, random_state=0)

lr = lm.LinearRegression().fit(X, y)

l2 = lm.Ridge(alpha=10).fit(X, y)  # lambda is alpha!
print(l2.coef_)

l1 = lm.Lasso(alpha=1).fit(X, y)  # lambda is alpha !
print(l1.coef_)

l1l2 = lm.ElasticNet(alpha=1, l1_ratio=.9).fit(X, y)
