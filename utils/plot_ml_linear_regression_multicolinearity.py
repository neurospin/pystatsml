#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 10:23:37 2020

@author: ed203246
"""

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
import sklearn.linear_model as lm
import sklearn.metrics as metrics

from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=2)
pd.set_option('precision', 2)

# %% Plot linear regression plan (in 2d)

# Fit Ordinary Least Squares: OLS
csv = pd.read_csv('https://github.com/duchesnay/pystatsml/raw/master/datasets/Advertising.csv', index_col=0)
X = csv[['TV', 'Radio']]
y = csv['Sales']

lr = lm.LinearRegression().fit(X, y)
y_pred = lr.predict(X)
print("R-squared =", metrics.r2_score(y, y_pred))

print("Coefficients =", lr.coef_, lr.intercept_)

# Plot
fig = plt.figure(figsize=(9, 9))
#fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(csv['TV'], csv['Radio'], csv['Sales'], c='r', marker='o')

xx1, xx2 = np.meshgrid(
    np.linspace(csv['TV'].min(), csv['TV'].max(), num=10),
    np.linspace(csv['Radio'].min(), csv['Radio'].max(), num=10))

XX = np.column_stack([xx1.ravel(), xx2.ravel()])

yy = lr.predict(XX)
ax.plot_wireframe(xx1, xx2, yy.reshape(xx1.shape))
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
_ = ax.set_zlabel('Sales')

plt.savefig("/home/ed203246/git/pystatsml/images/linear_regression_plan.png")
