# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 13:52:52 2016

@author: ed203246
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

np.random.seed(42)

# dataset
n_samples = 100
experience = np.random.normal(size=n_samples)
salary = 1500 + experience + np.random.normal(size=n_samples, scale=.5)
X = np.column_stack([experience, salary])


'''
Write a class `BasicPCA` with two methods `fit(X)` that estimates the data mean
and principal components directions. `transform(X)` that project a new the data
into the principal components.

Check that your `BasicPCA` pfermed simillarly than the one from sklearn:
`from sklearn.decomposition import PCA`
'''

import numpy as np
from sklearn.decomposition import PCA


class BasicPCA():
    def fit(self, X):
        # U : Unitary matrix having left singular vectors as columns.
        #     Of shape (n_samples,n_samples) or (n_samples,n_comps), depending on
        #     full_matrices.
        #
        # s : The singular values, sorted in non-increasing order. Of shape (n_comps,), 
        #     with n_comps = min(n_samples, n_features).
        #
        # Vh: Unitary matrix having right singular vectors as rows. 
        #     Of shape (n_features, n_features) or (n_comps, n_features) depending on full_matrices.
        self.mean = X.mean(axis=0)
        Xc = X - self.mean  # Centering is required
        U, s, V = scipy.linalg.svd(Xc, full_matrices=False)
        self.explained_variance_ = (s ** 2) / n_samples
        self.explained_variance_ratio_ = (self.explained_variance_ /
                                 self.explained_variance_.sum())
        self.princ_comp_dir = V

    def transform(self, X):
        Xc = X - self.mean
        return(np.dot(Xc, self.princ_comp_dir.T))


X = np.column_stack([experience, salary])
pca = PCA(n_components=2)
pca.fit(X)

basic_pca = BasicPCA()
basic_pca.fit(X)

print(pca.explained_variance_ratio_)
assert np.all(basic_pca.transform(X) == pca.transform(X))


'''
Apply your sklearn PCA on `salary` and `experience` features of the salary
dataset available at: 
'https://raw.github.com/duchesnay/pylearn-doc/master/data/salary_table.csv'

How many components do you need to explain 95% of the variance ?

Discuss the contribution of the input features in the first component.
Are experience and salary positively correlated ?
'''
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

try:
    salary = pd.read_csv('data/salary_table.csv')
except:
    url = 'https://raw.github.com/duchesnay/pylearn-doc/master/data/salary_table.csv'
    salary = pd.read_csv(url)

X = salary[['salary', 'experience']]
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)
# [  9.99999147e-01   8.53068390e-07]
# Answer: 1

print(pca.components_)
#[[  9.99999825e-01   5.90856391e-04]
# [ -5.90856391e-04   9.99999825e-01]]

# Both salary and experience contribute in the same way to the first component.
# experience and salary are positively correlated