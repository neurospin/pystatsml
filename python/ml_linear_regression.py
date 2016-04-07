# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 12:28:00 2016

@author: edouard.duchesnay@cea.fr
"""

'''
sklearn
=======
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as metrics
#%matplotlib inline

# Fit Ordinary Least Squares: OLS
csv = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

X = csv[['TV', 'Radio']]
y =  csv['Sales']

lr = lm.LinearRegression().fit(X, y)
y_pred = lr.predict(X)

print("R-squared=", metrics.r2_score(y, y_pred))


# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(csv['TV'], csv['Radio'], csv['Sales'], c='r', marker='o')

xx1, xx2 = np.meshgrid(
    np.linspace(csv['TV'].min(), csv['TV'].max(), num=10),
    np.linspace(csv['Radio'].min(), csv['Radio'].max(), num=10))


XX = np.column_stack([xx1.ravel(), xx2.ravel()])

yy = lr.predict(XX)
ax.plot_surface(xx1, xx2, yy.reshape(xx1.shape), color='None')
ax.set_xlabel('TV')
ax.set_ylabel('Radio')
ax.set_zlabel('Sales')

'''
Overfitting
===========
'''

'''
High dimensionality
-------------------
'''

def fit_on_increasing_size(model):
    n_samples = 100
    n_features_ = np.arange(10, 800, 20)
    r2_train, r2_test, snr = [], [], []
    for n_features in n_features_:
        # Sample the dataset (* 2 nb of samples)
        n_features_info = int(n_features/10)
        np.random.seed(42)  # Make reproducible
        X = np.random.randn(n_samples * 2, n_features)
        beta = np.zeros(n_features)
        beta[:n_features_info] = 1
        Xbeta = np.dot(X, beta)
        eps = np.random.randn(n_samples * 2)
        y =  Xbeta + eps
        # Split the dataset into train and test sample
        Xtrain, Xtest = X[:n_samples, :], X[n_samples:, :], 
        ytrain, ytest = y[:n_samples], y[n_samples:]
        # fit/predict
        lr = model.fit(Xtrain, ytrain)
        y_pred_train = lr.predict(Xtrain)
        y_pred_test = lr.predict(Xtest)
        snr.append(Xbeta.std() / eps.std())
        r2_train.append(metrics.r2_score(ytrain, y_pred_train))
        r2_test.append(metrics.r2_score(ytest, y_pred_test))
    return n_features_, np.array(r2_train), np.array(r2_test), np.array(snr)

def plot_r2_snr(n_features_, r2_train, r2_test, xvline, snr, ax):
    """
    Two scales plot. Left y-axis: train test r-squared. Right y-axis SNR.
    """
    ax.plot(n_features_, r2_train, label="Train r-squared", linewidth=2)
    ax.plot(n_features_, r2_test, label="Test r-squared", linewidth=2)
    ax.axvline(x=xvline, linewidth=2, color='k', ls='--')
    ax.axhline(y=0, linewidth=1, color='k', ls='--')
    ax.set_ylim(-0.2, 1.1)
    ax.set_xlabel("Number of input features")
    ax.set_ylabel("r-squared")
    ax.legend(loc='best')
    ax.set_title("Prediction perf.")
    ax_right = ax.twinx()
    ax_right.plot(n_features_, snr, 'r-', label="SNR", linewidth=1)
    ax_right.set_ylabel("SNR", color='r')
    for tl in ax_right.get_yticklabels():
        tl.set_color('r')

'''
No penalization
~~~~~~~~~~~~~~~
'''

# Model = linear regression
lr = lm.LinearRegression()

# Fit models on dataset
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=lr)

argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 100 first features
plot_r2_snr(n_features[n_features <= 100], 
            r2_train[n_features <= 100], r2_test[n_features <= 100],
            argmax,
            snr[n_features <= 100],
            axis[1])
plt.tight_layout()

'''
L2 penalization
~~~~~~~~~~~~~~~
'''

# Model = linear regression
ridge = lm.Ridge(alpha=10)

# Fit models on dataset
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=ridge)

argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 100 first features
plot_r2_snr(n_features[n_features <= 100], 
            r2_train[n_features <= 100], r2_test[n_features <= 100],
            argmax,
            snr[n_features <= 100],
            axis[1])
plt.tight_layout()


'''
Multicollinearity
-----------------
'''
## Dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

bv = np.array([10, 20, 30, 40, 50])             # buisness volumes indicator
tax  = .2 * bv                                  # Tax
bp = .1 * bv + np.array([-.1, .2, .1, -.2, .1]) # business potential

X = np.column_stack([bv, tax])
beta_star = np.array([.1, 0])  # true solution

'''
Since tax and b are correlated, there is an infinite number of linear combinations
leading to the same prediction.
'''
 
# 10 times the bv then subtract it 9 times using the tax variable: 
beta_medium = np.array([.1 * 10, -.1 * 9 * (1/.2)])
# 100 times the bv then subtract it 99 times using the tax variable: 
beta_large = np.array([.1 * 100, -.1 * 99 * (1/.2)])

# Check that all model lead to the same result
assert np.all(np.dot(X, beta_star) == np.dot(X, beta_medium))
assert np.all(np.dot(X, beta_star) == np.dot(X, beta_large))

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xx1, xx2 = np.meshgrid(
    np.linspace(X[:, 0].min(), X[:, 0].max(), num=10),
    np.linspace(X[:, 1].min(), X[:, 1].max(), num=10))

XX = np.column_stack([xx1.ravel(), xx2.ravel()])
yy_1 = np.dot(XX, beta_star)
yy_2 = np.dot(XX, beta_medium)
yy_3 = np.dot(XX, beta_large)

ax.plot_wireframe(xx1, xx2, yy_1.reshape(xx1.shape), color="blue",
                  label=r"$||\beta||_2^2=%.2f$" % np.sum(beta_star ** 2))
ax.plot_wireframe(xx1, xx2, yy_2.reshape(xx1.shape), color="green",
                  label=r"$||\beta||_2^2=%.2f$" % np.sum(beta_medium ** 2))
ax.plot_wireframe(xx1, xx2, yy_3.reshape(xx1.shape), color="red", 
                  label=r"$||\beta||_2^2=%.2f$" % np.sum(beta_large ** 2))

ax.scatter(X[:, 0], X[:, 1], zs=bp, c='k', marker='o', s=100, depthshade=True)

plt.legend()
ax.set_xlabel('Buisness volumes')
ax.set_ylabel('Tax')
ax.set_zlabel('Business potential')
ax.set_zlim((-100, + 100))

'''
L2 & L1 Penalization
====================
'''

## Dataset ##
# Plot]
#n_samples = 3
X = np.array([[1.0, 1.1], [0, 0], [-1.0, -1.1]])
n_samples = X.shape[0]
print(np.cov(X.T))
beta_s = np.array([15, 0])
beta_m = np.array([2, -1])
beta_l = np.array([3, -2])

noise = np.array([1, -3, 2]) / 10 # np.random.randn(n_samples)
y =  np.dot(X, beta_s) + noise

def l1_max_linear_loss(X, y, mean=True):
    n = float(X.shape[0])
    scale = 1.0 / n if mean else 1.
    l1_max = scale * np.abs(np.dot(X.T, y)).max()
    return l1_max

l1_max_linear_loss(X, y)
#10

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
# latex
from matplotlib import rc
plt.rc('text', usetex=True)
font = {'family' : 'serif', 'size':22}
plt.rc('font', **font)




def plot3d(X, Y, Z, point, zlim=None, ax=None, fig=None, xylabelsize=33):
    # Plot
    from matplotlib import cm
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(111, projection='3d')
    z_min = np.min(Z) - np.max(Z)/2
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10,
                    #vmin=Z.min(), vmax=Z.max(),
                    cmap=cm.coolwarm,
                    linewidth=1, antialiased=True)
    cset = ax.contourf(X, Y, Z, zdir='z', offset=z_min,
                       #norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                       cmap=cm.coolwarm)
    argmin = X.ravel()[Z.argmin()], Y.ravel()[Z.argmin()]
    print("argmin", argmin)
    # add point and cross at defined point
    ax.plot([point[0]], [point[1]], 'wo', zs=[z_min], ms=20)
    ax.plot([X.min(), X.max()], [point[1], point[1]], '--w', zs=[z_min, z_min], linewidth=2.0)
    ax.plot([point[0], point[0]], [Y.min(), Y.max()], '--w', zs=[z_min, z_min], linewidth=2.0)
    # add point and cross at argmin
    ax.plot([argmin[0]], [argmin[1]], 'o', color='k', zs=[z_min], ms=20)
    ax.plot([X.min(), X.max()], [argmin[1], argmin[1]], '--', color='k', zs=[z_min, z_min], linewidth=2.0)
    ax.plot([argmin[0], argmin[0]], [Y.min(), Y.max()], '--', color='k', zs=[z_min, z_min], linewidth=2.0)
    #ax.text(argmin[0], argmin[1], z_min, ".  (%.3f, %.3f)" % argmin)
    ax.set_xlabel(r'$\beta_1$', size=xylabelsize)
    ax.set_ylabel(r'$\beta_2$', size=xylabelsize)
    #ax.set_zlabel(r'Error', size=xylabelsize)
    ax.set_zlim(z_min, np.max(Z))
    return ax, z_min, argmin


import statsmodels.api as sm
## Fit and summary:
model = sm.OLS(y, X).fit()
print(model.summary())

dx = dy = 50
beta1, beta2 = np.meshgrid(
    np.linspace(-dx, dx, num=100),
    np.linspace(-dy, dy, num=100))


# Make sure |beta(0, 0)| = 0 is sampled
Betas = np.column_stack([beta1.ravel(), beta2.ravel()])
L2 = (Betas ** 2).sum(axis=1)
beta1.ravel()[L2.argmin()] = 0
beta2.ravel()[L2.argmin()] = 0
Betas = np.column_stack([beta1.ravel(), beta2.ravel()])
L2 = (Betas ** 2).sum(axis=1)
assert L2[L2.argmin()] == 0


fig = plt.figure(figsize=(9, 3)) 

# OLS
ax=fig.add_subplot(231, projection='3d')
#ax=fig.add_subplot(111, projection='3d')
MSE = 1 / n_samples * np.sum((y - (np.dot(X, Betas.T).T)) ** 2, axis=1)
#MSE = np.log(MSE)
print(MSE.min(), MSE.mean(), MSE.max())
ax, z_min, argmin = plot3d(beta1, beta2, MSE.reshape(beta1.shape), point=beta_s, ax=ax, fig=fig)
plt.title(r'$\frac{1}{N}||y- X \beta ||_2^2$')


# L2
ax=fig.add_subplot(232, projection='3d')
L2 = (Betas ** 2).sum(axis=1)
print(L2.min(), L2.mean(), L2.max())
plot3d(beta1, beta2, L2.reshape(beta1.shape), point=beta_s, ax=ax, fig=fig)
plt.title(r'$||\beta ||_2^2$')


# Ridge
ax=fig.add_subplot(233, projection='3d')
Ridge = MSE + L2
plot3d(beta1, beta2, Ridge.reshape(beta1.shape), point=beta_s, ax=ax, fig=fig)
plt.title(r'$\frac{1}{N}||y- X \beta ||_2^2 + \lambda ||\beta ||_2^2$')

plt.tight_layout()


# L1
ax=fig.add_subplot(235, projection='3d')
L1 = (np.abs(Betas)).sum(axis=1)
print(L1.min(), L1.mean(), L1.max())
plot3d(beta1, beta2, L1.reshape(beta1.shape), point=beta_s, ax=ax, fig=fig)
plt.title(r'$||\beta ||_1$')


# Lasso
ax=fig.add_subplot(236, projection='3d')
Lasso = MSE + 100 * L1
#print(Lasso.min(), Lasso.mean(), Lasso.max())
plot3d(beta1, beta2, Lasso.reshape(beta1.shape), point=beta_s, ax=ax, fig=fig)
plt.title(r'$\frac{1}{N}||y- X \beta ||_2^2 + \lambda ||\beta ||_1$')

'''
Lasso
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as metrics

# Model = linear regression
# lambda is alpha !
lasso = lm.Lasso(alpha=.1)

# Fit models on dataset
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=lasso)

argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 200 first features
plot_r2_snr(n_features[n_features <= 200], 
            r2_train[n_features <= 200], r2_test[n_features <= 200],
            argmax,
            snr[n_features <= 200],
            axis[1])
plt.tight_layout()

'''
L1 L2 plot, L1 sparse
'''
import matplotlib.pyplot as plt
import numpy as np

n_samples = 1000
X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, .5],[.5, 1]], size=n_samples)
#X = np.array([[1.0, 1.1], [0, 0], [-1.0, -1.1]])
n_samples = X.shape[0]
print(np.cov(X.T))
beta_s = np.array([3, -3])

noise = np.random.randn(n_samples)
y =  np.dot(X, beta_s) + noise

np.std(np.dot(X, beta_s)) / np.std(noise)

def l1_max_linear_loss(X, y, mean=True):
    n = float(X.shape[0])
    scale = 1.0 / n if mean else 1.
    l1_max = scale * np.abs(np.dot(X.T, y)).max()
    return l1_max

l1_max_linear_loss(X, y)
#10
dx = dy = 5
beta1, beta2 = np.meshgrid(
    np.linspace(-dx, dx, num=100),
    np.linspace(-dy, dy, num=100))


# Make sure |beta(0, 0)| = 0 is sampled
Betas = np.column_stack([beta1.ravel(), beta2.ravel()])
L2 = (Betas ** 2).sum(axis=1)

MSE = 1 / n_samples * np.sum((y - (np.dot(X, Betas.T).T)) ** 2, axis=1)


from matplotlib.colors import LogNorm
cax = plt.matshow(MSE.reshape(beta1.shape),
                  norm=LogNorm(vmin=MSE.min(), vmax=MSE.max()),
                  cmap=plt.cm.coolwarm)
frame = plt.gca()
frame.get_xaxis().set_visible(False)
frame.get_yaxis().set_visible(False)
 
plt.savefig("/tmp/toto.svg")
plt.close()


'''
Elastic-net
'''
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as lm

# Model = linear regression
# lambda is alpha !
mod = lm.ElasticNet(alpha=.5, l1_ratio=.5)

# Fit models on dataset
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)

argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 100 first features
plot_r2_snr(n_features[n_features <= 100], 
            r2_train[n_features <= 100], r2_test[n_features <= 100],
            argmax,
            snr[n_features <= 100],
            axis[1])
plt.tight_layout()


'''
Realistic dataset for lm comparison: Work in Progress
=====================================================
'''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import seaborn

var, cov = 1, 1
#n_features = 3
Cov = np.ones((n_features, n_features)) * cov

def fit_on_increasing_size(model):
    n_samples = 50
    n_features_ = np.arange(10, 390, 20)
    r2_train, r2_test, snr = [], [], []
    for n_features in n_features_:
        # Sample the dataset (* 2 nb of samples)
        n_features_info = int(n_features/10)
        #print(n_features_info)
        mean = np.zeros(n_features)
        #Cov = np.ones((n_features, n_features)) * cov
        #Cov[np.diag_indices_from(Cov)] = var
        Cov = np.identity(n_features) * var
        diag_indices = np.arange(n_features)
        Cov[((diag_indices + 1)[:-1], diag_indices[:-1])] = cov
        Cov[(diag_indices[:-1], (diag_indices + 1)[:-1])] = cov
        np.random.seed(43)  # Make reproducible
        X = np.random.multivariate_normal(mean=mean, cov=Cov, size=n_samples * 2)
        #X = np.random.randn(n_samples * 2, n_features)
        beta = np.zeros(n_features)
        beta[:n_features_info] = 1
        Xbeta = np.dot(X, beta)
        eps = np.random.randn(n_samples * 2) * 3.
        y =  Xbeta + eps
        # Split the dataset into train and test sample
        Xtrain, Xtest = X[:n_samples, :], X[n_samples:, :], 
        ytrain, ytest = y[:n_samples], y[n_samples:]
        # fit/predict
        #try:
        #    mod.set_params(alpha=np.sqrt(n_features))
        #    print(mod.alpha)
        #except:
        #    pass
        lr = model.fit(Xtrain, ytrain)
        y_pred_train = lr.predict(Xtrain)
        y_pred_test = lr.predict(Xtest)
        snr.append(Xbeta.std() / eps.std())
        r2_train.append(metrics.r2_score(ytrain, y_pred_train))
        r2_test.append(metrics.r2_score(ytest, y_pred_test))
    return n_features_, np.array(r2_train), np.array(r2_test), np.array(snr)


# Model = linear regression
mod = lm.LinearRegression(fit_intercept=False)
# lambda is alpha !
mod = lm.Ridge(alpha=10)
mod = lm.Lasso(alpha=.1)
mod = lm.ElasticNet(alpha=1, l1_ratio=.1)

# Fit models on dataset
n_features, r2_train, r2_test, snr = fit_on_increasing_size(model=mod)

argmax = n_features[np.argmax(r2_test)]

# plot
fig, axis = plt.subplots(1, 2, figsize=(9, 3))

# Left pane: all features
plot_r2_snr(n_features, r2_train, r2_test, argmax, snr, axis[0])

# Right pane: Zoom on 200 first features
plot_r2_snr(n_features[n_features <= 200], 
            r2_train[n_features <= 200], r2_test[n_features <= 200],
            argmax,
            snr[n_features <= 200],
            axis[1])
plt.tight_layout()
