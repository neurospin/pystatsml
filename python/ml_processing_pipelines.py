# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:40:35 2016

@author: edoaurd.duchesnay@cea.fr
"""
import numpy as np
from sklearn import datasets
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.cross_validation import KFold

'''
Regression pipelines
====================
'''
import numpy as np
from sklearn import datasets
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.cross_validation import KFold

# Datasets
n_samples, n_features, noise_sd = 100, 100, 20
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=n_features, noise=noise_sd,
                         n_informative=5, random_state=42, coef=True)
 
# Use this to tune the noise parameter such that snr < 5
print("SNR:", np.std(np.dot(X, coef)) / noise_sd)

print("=============================")
print("== Basic linear regression ==")
print("=============================")

scores = cross_val_score(estimator=lm.LinearRegression(), X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())

print("==============================================")
print("== Scaler + anova filter + ridge regression ==")
print("==============================================")

anova_ridge = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('selectkbest', SelectKBest(f_regression)),
    ('ridge', lm.Ridge())
])
param_grid = {'selectkbest__k':np.arange(10, 110, 10), 
              'ridge__alpha':[.001, .01, .1, 1, 10, 100] }

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid, n_jobs=-1)
scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())

print("=====================================")
print("== Scaler + Elastic-net regression ==")
print("=====================================")

enet = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('enet', lm.ElasticNet(max_iter=10000)),
])
param_grid = {'enet__alpha': [.001, .01, .1, 1, 10, 100],
              'enet__l1_ratio':[.1, .5, .9]}
enet_cv = GridSearchCV(enet, cv=5,  param_grid=param_grid, n_jobs=-1)
scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())

'''
Classification pipelines
========================
'''
import numpy as np
from sklearn import datasets
import sklearn.linear_model as lm
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classification
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics
from sklearn.cross_validation import KFold

# Datasets
n_samples, n_features, noise_sd = 100, 100, 20
X, y, coef = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                         n_informative=5, random_state=42)
 
print("=============================")
print("== Basic linear regression ==")
print("=============================")

scores = cross_val_score(estimator=lm.LinearRegression(), X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())

print("==============================================")
print("== Scaler + anova filter + ridge regression ==")
print("==============================================")

anova_ridge = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('selectkbest', SelectKBest(f_classification)),
    ('ridge', lm.Ridge())
])
param_grid = {'selectkbest__k':np.arange(10, 110, 10), 
              'ridge__alpha':[.001, .01, .1, 1, 10, 100] }

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid, n_jobs=-1)
scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())

print("=====================================")
print("== Scaler + Elastic-net regression ==")
print("=====================================")

enet = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('enet', lm.ElasticNet(max_iter=10000)),
])
param_grid = {'enet__alpha': [.001, .01, .1, 1, 10, 100],
              'enet__l1_ratio':[.1, .5, .9]}
enet_cv = GridSearchCV(enet, cv=5,  param_grid=param_grid, n_jobs=-1)
scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5)
print("Test  r2:%.2f" % scores.mean())
