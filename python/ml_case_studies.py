'''
Case studies of ML
==================
'''

'''

Default of credit card clients Data Set
---------------------------------------

Sources: 

http://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

Data Set Information:
~~~~~~~~~~~~~~~~~~~
This research aimed at the case of customers default payments in Taiwan.

Attribute Information:

This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:

- X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.

- X2: Gender (1 = male; 2 = female).

- X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).

- X4: Marital status (1 = married; 2 = single; 3 = others).

- X5: Age (year).

- X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005;...;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months;...; 8 = payment delay for eight months; 9 = payment delay for nine months and above.

- X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005;...; X17 = amount of bill statement in April, 2005.

- X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005;...;X23 = amount paid in April, 2005.

'''

'''
Read dataset
~~~~~~~~~~~~
'''

from __future__ import print_function

import pandas as pd
import numpy as np

url = 'https://raw.github.com/neurospin/pystatsml/master/data/default%20of%20credit%20card%20clients.xls'
data = pd.read_excel(url, skiprows=1, sheetname='Data')

df = data.copy()
target = 'default payment next month'
print(df.columns)

#Index(['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0',
#       'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
#       'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
#       'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
#       'default payment next month'],
#      dtype='object')


'''
Data recoding of categorial factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Categorial factors with 2 levels are kept
- Categorial that are ordinal are kept
- Undocumented values are replaced with NaN
'''
def describe_factor(x):
    ret = dict()
    for lvl in x.unique():
        if pd.isnull(lvl):
            ret["NaN"] = x.isnull().sum()
        else:
           ret[lvl] = np.sum(x==lvl)
    return ret

print('Sex')
print(describe_factor(df["SEX"]))
# {1: 11888, 2: 18112}

print('Education is ordinnal Keep it, but set, others to NA')
print(describe_factor(df["EDUCATION"]))
# {0: 14, 1: 10585, 2: 14030, 3: 4917, 4: 123, 5: 280, 6: 51}

# remap unknown with NaN
df["EDUCATION"] = df["EDUCATION"].map({0: np.NaN, 1:1, 2:2, 3:3, 4:np.NaN, 
    5: np.NaN, 6: np.NaN})
print(describe_factor(df["EDUCATION"]))
# {1.0: 10585, 2.0: 14030, 3.0: 4917, 'NaN': 468}

print('MARRIAGE 0,3=>NA')
print(describe_factor(df["MARRIAGE"]))
# {0: 54, 1: 13659, 2: 15964, 3: 323}

df.MARRIAGE = df.MARRIAGE.map({0:np.NaN, 1:1, 2:0, 3:np.NaN})
print(describe_factor(df.MARRIAGE))
# {0.0: 15964, 1.0: 13659, 'NaN': 377}

print("Others are quantitative and presents")

print(df.describe())
#                 ID       LIMIT_BAL           SEX     EDUCATION      MARRIAGE  \
#count  30000.000000    30000.000000  30000.000000  29532.000000  29623.000000   
#mean   15000.500000   167484.322667      1.603733      1.808073      0.461094   
#std     8660.398374   129747.661567      0.489129      0.698643      0.498492   
#min        1.000000    10000.000000      1.000000      1.000000      0.000000   
#25%     7500.750000    50000.000000      1.000000      1.000000      0.000000   
#50%    15000.500000   140000.000000      2.000000      2.000000      0.000000   
#75%    22500.250000   240000.000000      2.000000      2.000000      1.000000   
#max    30000.000000  1000000.000000      2.000000      3.000000      1.000000   
#
#                AGE         PAY_0         PAY_2         PAY_3         PAY_4  \
#count  30000.000000  30000.000000  30000.000000  30000.000000  30000.000000   
#mean      35.485500     -0.016700     -0.133767     -0.166200     -0.220667   
#std        9.217904      1.123802      1.197186      1.196868      1.169139   
#min       21.000000     -2.000000     -2.000000     -2.000000     -2.000000   
#25%       28.000000     -1.000000     -1.000000     -1.000000     -1.000000   
#50%       34.000000      0.000000      0.000000      0.000000      0.000000   
#75%       41.000000      0.000000      0.000000      0.000000      0.000000   
#max       79.000000      8.000000      8.000000      8.000000      8.000000   
#
#                 BILL_AMT4      BILL_AMT5  \
#count             ...               30000.000000   30000.000000   
#mean              ...               43262.948967   40311.400967   
#std               ...               64332.856134   60797.155770   
#min               ...             -170000.000000  -81334.000000   
#25%               ...                2326.750000    1763.000000   
#50%               ...               19052.000000   18104.500000   
#75%               ...               54506.000000   50190.500000   
#max               ...              891586.000000  927171.000000   
#
#           BILL_AMT6       PAY_AMT1        PAY_AMT2      PAY_AMT3  \
#count   30000.000000   30000.000000    30000.000000   30000.00000   
#mean    38871.760400    5663.580500     5921.163500    5225.68150   
#std     59554.107537   16563.280354    23040.870402   17606.96147   
#min   -339603.000000       0.000000        0.000000       0.00000   
#25%      1256.000000    1000.000000      833.000000     390.00000   
#50%     17071.000000    2100.000000     2009.000000    1800.00000   
#75%     49198.250000    5006.000000     5000.000000    4505.00000   
#max    961664.000000  873552.000000  1684259.000000  896040.00000   
#
#            PAY_AMT4       PAY_AMT5       PAY_AMT6  default payment next month  
#count   30000.000000   30000.000000   30000.000000                30000.000000  
#mean     4826.076867    4799.387633    5215.502567                    0.221200  
#std     15666.159744   15278.305679   17777.465775                    0.415062  
#min         0.000000       0.000000       0.000000                    0.000000  
#25%       296.000000     252.500000     117.750000                    0.000000  
#50%      1500.000000    1500.000000    1500.000000                    0.000000  
#75%      4013.250000    4031.500000    4000.000000                    0.000000  
#max    621000.000000  426529.000000  528666.000000                    1.000000  

''' 
Missing data
~~~~~~~~~~~~
'''

print(df.isnull().sum())
#ID                              0
#LIMIT_BAL                       0
#SEX                             0
#EDUCATION                     468
#MARRIAGE                      377
#AGE                             0
#PAY_0                           0
#PAY_2                           0
#PAY_3                           0
#PAY_4                           0
#PAY_5                           0
#PAY_6                           0
#BILL_AMT1                       0
#BILL_AMT2                       0
#BILL_AMT3                       0
#BILL_AMT4                       0
#BILL_AMT5                       0
#BILL_AMT6                       0
#PAY_AMT1                        0
#PAY_AMT2                        0
#PAY_AMT3                        0
#PAY_AMT4                        0
#PAY_AMT5                        0
#PAY_AMT6                        0
#default payment next month      0
#dtype: int64

df.ix[df["EDUCATION"].isnull(), "EDUCATION"] = df["EDUCATION"].mean()
df.ix[df["MARRIAGE"].isnull(), "MARRIAGE"] = df["MARRIAGE"].mean()
print(df.isnull().sum().sum())
# O

describe_factor(df[target])
{0: 23364, 1: 6636}

'''
Prepare Data set
~~~~~~~~~~~~~~~~
'''

predictors = df.columns.drop(['ID', target])
X = np.asarray(df[predictors])
y = np.asarray(df[target])

'''
Univariate analysis
~~~~~~~~~~~~~~~~~~~
'''

'''
Machine Learning with SVM
~~~~~~~~~~~~~~~~~~~~~~~~~

On this large dataset, we can afford to set aside some test samples. This will
also save computation time. However we will have to do some manual work. 
'''

import numpy as np
from sklearn import datasets
import sklearn.svm as svm
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import sklearn.metrics as metrics

def balanced_acc(estimator, X, y):
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

print("===============================================")
print("== Put aside half of the samples as test set ==")
print("===============================================")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0, stratify=y)

print("=================================")
print("== Scale trainin and test data ==")
print("=================================")

scaler = preprocessing.StandardScaler()
Xtrs = scaler.fit(Xtr).transform(Xtr)
Xtes = scaler.transform(Xte)

print("=========")
print("== SVM ==")
print("=========")

svc = svm.LinearSVC(class_weight='balanced', dual=False)
%time scores = cross_val_score(estimator=svc,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#CPU times: user 1.01 s, sys: 39.7 ms, total: 1.05 s
#Wall time: 112 ms
#Validation  bACC:0.67

svc_rbf = svm.SVC(kernel='rbf', class_weight='balanced')
%time scores = cross_val_score(estimator=svc_rbf,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#CPU times: user 10.2 s, sys: 136 ms, total: 10.3 s
#Wall time: 10.3 s
#Test  bACC:0.71

svc_lasso = svm.LinearSVC(class_weight='balanced', penalty='l1', dual=False)
%time scores = cross_val_score(estimator=svc_lasso,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#CPU times: user 4.51 s, sys: 168 ms, total: 4.68 s
#Wall time: 544 ms
#Test  bACC:0.67

print("========================")
print("== SVM CV Grid search ==")
print("========================")
Cs = [0.001, .01, .1, 1, 10, 100, 1000]
param_grid = {'C':Cs}

print("-------------------")
print("-- SVM Linear L2 --")
print("-------------------")

svc_cv = GridSearchCV(svc, cv=3,  param_grid=param_grid, scoring=balanced_acc,
                      n_jobs=-1)
# What are the best parameters ?
%time svc_cv.fit(Xtrs, ytr).best_params_
#CPU times: user 211 ms, sys: 209 ms, total: 421 ms
#Wall time: 1.07 s
#{'C': 0.01}
scores = cross_val_score(estimator=svc_cv,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#Validation bACC:0.67

print("-------------")
print("-- SVM RBF --")
print("-------------")

svc_rbf_cv = GridSearchCV(svc_rbf, cv=3,  param_grid=param_grid,
                          scoring=balanced_acc, n_jobs=-1)
# What are the best parameters ?
%time svc_rbf_cv.fit(Xtrs, ytr).best_params_
#Wall time: 1min 10s
#Out[6]: {'C': 1}

# reduce the grid search
svc_rbf_cv.param_grid={'C': [0.1, 1, 10]}
scores = cross_val_score(estimator=svc_rbf_cv,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#Validation bACC:0.71

print("-------------------")
print("-- SVM Linear L1 --")
print("-------------------")

svc_lasso_cv = GridSearchCV(svc_lasso, cv=3,  param_grid=param_grid,
                            scoring=balanced_acc, n_jobs=-1)
# What are the best parameters ?
%time svc_lasso_cv.fit(Xtrs, ytr).best_params_
#CPU times: user 514 ms, sys: 181 ms, total: 695 ms
#Wall time: 2.07 s
#Out[10]: {'C': 0.1}

# reduce the grid search
svc_lasso_cv.param_grid={'C': [0.1, 1, 10]}

scores = cross_val_score(estimator=svc_lasso_cv,\
                         X=Xtrs, y=ytr, cv=2, scoring=balanced_acc)
print("Validation bACC:%.2f" % scores.mean())
#Validation bACC:0.67



print("SVM-RBF, test bACC:%.2f" % balanced_acc(svc_rbf_cv, Xtes, yte))
# SVM-RBF, test bACC:0.70

print("SVM-Lasso, test bACC:%.2f" % balanced_acc(svc_lasso_cv, Xtes, yte))
# SVM-Lasso, test bACC:0.67


## SKIP
###############################################################################
'''
Machine Learning: Logistic regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

def balanced_acc(estimator, X, y):
    """
    Balanced acuracy scorer
    """
    return metrics.recall_score(y, estimator.predict(X), average=None).mean()

print("===============================")
print("== Basic logistic regression ==")
print("===============================")

scores = cross_val_score(estimator=lm.LogisticRegression(C=1e8, class_weight='balanced'),
                         X=X, y=y, cv=5, scoring=balanced_acc)
print("Test  bACC:%.2f" % scores.mean())
# Test  bACC:0.65

print("=======================================================")
print("== Scaler + anova filter + ridge logistic regression ==")
print("=======================================================")

anova_ridge = Pipeline([
    ('standardscaler', preprocessing.StandardScaler()),
    ('selectkbest', SelectKBest(f_classif)),
    ('ridge', lm.LogisticRegression(penalty='l2', class_weight='balanced'))
])
param_grid = {'selectkbest__k':np.arange(2, X.shape[1]+1, 2), 
              'ridge__C':[.0001, .001, .01, .1, 1, 10, 100, 1000, 10000]}


# Expect execution in ipython, for python remove the %time
print("----------------------------")
print("-- Parallelize outer loop --")
print("----------------------------")

anova_ridge_cv = GridSearchCV(anova_ridge, cv=5,  param_grid=param_grid,
                              scoring=balanced_acc)
%time scores = cross_val_score(estimator=anova_ridge_cv, X=X, y=y, cv=5,\
                               scoring=balanced_acc, n_jobs=-1)
print("Test bACC:%.2f" % scores.mean())
#CPU times: user 225 ms, sys: 388 ms, total: 613 ms
#Wall time: 1min 19s
#Test bACC:0.69


print("========================================")
print("== Scaler + lasso logistic regression ==")
print("========================================")

Cs = np.array([.0001, .001, .01, .1, 1, 10, 100, 1000, 10000])
alphas = 1 / Cs
l1_ratio = [.1, .5, .9]

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

#CPU times: user 1min 25s, sys: 2.71 s, total: 1min 28s
#Wall time: 5.57 s
#Test bACC:0.81

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

###############################################################################
## SKIP




