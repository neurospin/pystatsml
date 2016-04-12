# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 14:44:13 2016

@author: edouard.duchesnay@cea.fr
"""


'''
Regression
==========
'''

'''
Do it yourself
--------------
'''
import numpy as np
X = np.random.randn(n_samples * 2, n_features)
beta = np.zeros(n_features)
beta[:n_features_info] = 1
Xbeta = np.dot(X, beta)
eps = np.random.randn(n_samples * 2)
y = Xbeta + eps

'''
sklearn
-------
'''
from sklearn import datasets
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from sklearn.cross_validation import KFold

X, y = datasets.make_regression(n_samples=100, n_features=100, 
                         n_informative=10, random_state=42)



'''
Classification
==============
'''

'''
Do it yourself
--------------
'''
import numpy as np

# Dataset
n_samples, n_features = 100, 2
mean0, mean1 = np.array([0, 0]), np.array([0, 2])
Cov = np.array([[1, .8],[.8, 1]])
np.random.seed(42)
X0 = np.random.multivariate_normal(mean0, Cov, n_samples)
X1 = np.random.multivariate_normal(mean1, Cov, n_samples)
X = np.vstack([X0, X1])
y = np.array([0] * X0.shape[0] + [1] * X1.shape[0])

n_samples, n_features,  = 100, 2

np.random.randn()


'''
sklearn
-------
'''
from sklearn import datasets
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from sklearn.cross_validation import StratifiedKFold

X, y = datasets.make_classification(n_samples=100, n_features=100, 
                         n_informative=10, random_state=42)



