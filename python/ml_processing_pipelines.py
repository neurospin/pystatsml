# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:40:35 2016

@author: edoaurd.duchesnay@cea.fr
"""
from sklearn import preprocessing
preprocessing.OneHotEncoder


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

# Datasets
n_samples, n_features, noise_sd = 100, 100, 20
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=n_features, 
                                      noise=noise_sd, n_informative=5,
                                      random_state=42, coef=True)
 
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

# Expect execution in ipython, for python remove the %time
print("----------------------------")
print("-- Parallelize inner loop --")
print("----------------------------")

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid, n_jobs=-1)
%time scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5)
print("Test r2:%.2f" % scores.mean())

print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid)
%time scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5, n_jobs=-1)
print("Test r2:%.2f" % scores.mean())


print("=====================================")
print("== Scaler + Elastic-net regression ==")
print("=====================================")

alphas = [.0001, .001, .01, .1, 1, 10, 100, 1000] 
l1_ratio = [.1, .5, .9]

print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

enet = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('enet', lm.ElasticNet(max_iter=10000)),
])
param_grid = {'enet__alpha':alphas ,
              'enet__l1_ratio':l1_ratio}
enet_cv = GridSearchCV(enet, cv=5,  param_grid=param_grid)
%time scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5, n_jobs=-1)
print("Test r2:%.2f" % scores.mean())

print("-----------------------------------------------")
print("-- Parallelize outer loop + built-in CV      --")
print("-- Remark: scaler is only done on outer loop --")
print("-----------------------------------------------")

enet_cv = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('enet', lm.ElasticNetCV(max_iter=10000, l1_ratio=l1_ratio, alphas=alphas)),
])

%time scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5)
print("Test r2:%.2f" % scores.mean())

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
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

# Datasets
n_samples, n_features, noise_sd = 100, 100, 20
X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                         n_informative=5, random_state=42)


def balanced_acc(estimator, X, y):
    '''
    Balanced acuracy scorer
    '''
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

print("=============================")
print("== Basic logistic regression ==")
print("=============================")

scores = cross_val_score(estimator=lm.LogisticRegression(C=1e8, class_weight='balanced'),
                         X=X, y=y, cv=5, scoring=balanced_acc)
print("Test  bACC:%.2f" % scores.mean())

print("=======================================================")
print("== Scaler + anova filter + ridge logistic regression ==")
print("=======================================================")

anova_ridge = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('selectkbest', SelectKBest(f_classif)),
    ('ridge', lm.LogisticRegression(penalty='l2', class_weight='balanced'))
])
param_grid = {'selectkbest__k':np.arange(10, 110, 10), 
              'ridge__C':[.0001, .001, .01, .1, 1, 10, 100, 1000, 10000]}


# Expect execution in ipython, for python remove the %time
print("----------------------------")
print("-- Parallelize inner loop --")
print("----------------------------")

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid, 
                              scoring=balanced_acc, n_jobs=-1)
%time scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5,\
                               scoring=balanced_acc)
print("Test bACC:%.2f" % scores.mean())

print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid,
                              scoring=balanced_acc)
%time scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5,\
                               scoring=balanced_acc, n_jobs=-1)
print("Test bACC:%.2f" % scores.mean())


print("========================================")
print("== Scaler + lasso logistic regression ==")
print("========================================")

Cs = np.array([.0001, .001, .01, .1, 1, 10, 100, 1000, 10000])
alphas = 1 / Cs
l1_ratio = [.1, .5, .9]

print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

lasso = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('lasso', lm.LogisticRegression(penalty='l1', class_weight='balanced')),
])
param_grid = {'lasso__C':Cs}
enet_cv = GridSearchCV(lasso, cv=5,  param_grid=param_grid, scoring=balanced_acc)
%time scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5,\
                               scoring=balanced_acc, n_jobs=-1)
print("Test bACC:%.2f" % scores.mean())


print("-----------------------------------------------")
print("-- Parallelize outer loop + built-in CV      --")
print("-- Remark: scaler is only done on outer loop --")
print("-----------------------------------------------")

lasso_cv = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('lasso', lm.LogisticRegressionCV(Cs=Cs, scoring=balanced_acc)),
])

%time scores = cross_val_score(estimator=lasso_cv, X=X, y=y, cv=5)
print("Test bACC:%.2f" % scores.mean())


print("=============================================")
print("== Scaler + Elasticnet logistic regression ==")
print("=============================================")

print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

enet = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('enet', lm.SGDClassifier(loss="log", penalty="elasticnet",
                            alpha=0.0001, l1_ratio=0.15, class_weight='balanced')),
])

param_grid = {'enet__alpha':alphas,
              'enet__l1_ratio':l1_ratio}

enet_cv = GridSearchCV(enet, cv=5,  param_grid=param_grid, scoring=balanced_acc)
%time scores = cross_val_score(estimator=enet_cv, X=X, y=y, cv=5,\
    scoring=balanced_acc, n_jobs=-1)
print("Test bACC:%.2f" % scores.mean())
