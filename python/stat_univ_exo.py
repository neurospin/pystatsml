'''
Univariate statistics exercises
===============================
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(seed=42)  # make the example reproducible


'''
Estimator of main statistical measures
--------------------------------------

- Generate 2 ramdom samples $x \sim(1.78, 0.1)$, $y \sim(1.66, 0.1)$ both of size 10.

- Compute xbar $\bar{x}, \sigma_x, \sigma_{xy}$ using only `np.sum()` operation. 
Explore `np.` module to find out the numpy functions that does the same 
computations and compare them (using `assert`) with your previous results.
'''
n = 10
x  = np.random.normal(loc=1.78, scale=.1, size=n)
y  = np.random.normal(loc=1.66, scale=.1, size=n)

xbar = np.sum(x) / x.shape[0]
assert np.mean(x) == xbar

xvar = np.sum((x - xbar) ** 2) / (n - 1)
assert np.var(x, ddof=1) == xvar

ybar = np.sum(y) / n
xycov = np.sum((x - xbar) * (y - ybar)) / (n - 1)

xy = np.vstack((x, y))
Cov = np.cov(xy, ddof=1)  # or bias = True is the default behavior 
assert Cov[0, 0] == xvar
assert Cov[0, 1] == xycov
assert np.all(np.cov(xy, ddof=1) == np.cov(xy))

'''
One sample t-test
-----------------

- 
Given the following samples, test whether its true mean is 1.75.
Warning, when computing the std or the variance set ddof=1. The default
value 0, leads to the biased estimator of the variance.

'''
import scipy.stats as stats
n = 100
x = np.random.normal(loc=1.78, scale=.1, size=n)

'''
- Compute the t-value (tval)

- Plot the T(n-1) distribution for 100 tvalues values within [0, 10]. Draw P(T(n-1)>tval) 
  ie. color the surface defined by x values larger than tval below the T(n-1).
  Using the code.

- Compute the p-value: P(T(n-1)>tval).

- The p-value is one-sided: a two-sided test would test P(T(n-1) > tval)
  and P(T(n-1) < -tval). What would be the two sided p-value ?
  
- Compare the two-sided p-value with the one obtained by stats.ttest_1samp
using `assert np.allclose(arr1, arr2)`
'''


xbar, s, xmu, = np.mean(x), np.std(x, ddof=1), 1.75

tval = (xbar - xmu) / (s / np.sqrt(n))

tvalues = np.linspace(-10, 10, 100)
plt.plot(tvalues, stats.t.pdf(tvalues, n-1), 'b-', label="T(n-1)")
upper_tval_tvalues = tvalues[tvalues > tval]
plt.fill_between(upper_tval_tvalues, 0, stats.t.pdf(upper_tval_tvalues, n-1), alpha=.8)
plt.legend()

# Survival function (1 - `cdf`)
pval = stats.t.sf(tval, n - 1)

pval2sided = pval * 2
# do it with sicpy
assert np.allclose((tval, pval2sided), stats.ttest_1samp(x, xmu))


'''
Two sample t-test  (quantitative ~ categorial (2 levels))
---------------------------------------------------------

Given the following two sample, test whether their means are equals.
'''

import scipy.stats as stats
nx, ny = 50, 25
x = np.random.normal(loc=1.76, scale=.1, size=nx)
y = np.random.normal(loc=1.70, scale=.12, size=ny)

# Compute with scipy
tval, pval = stats.ttest_ind(x, y, equal_var=False)

'''
- Compute the t-value.
'''

xbar, ybar = np.mean(x), np.mean(y)
xvar, yvar = np.var(x, ddof=1), np.var(y, ddof=1)

'''
equal variance
~~~~~~~~~~~~~~~~
'''

sigma = np.sqrt((xvar * (nx - 1) + yvar * (ny - 1)) / (nx + ny - 2))

se = sigma * np.sqrt(1 / nx + 1 / ny)

tval = (xbar - ybar) / se

df = nx + ny - 2

'''
- Compute the p-value. The p-value is one-sided: a two-sided test would test P(T > tval) and P(T < -tval). What would be the two sided p-value ?
'''

pval = stats.t.sf(tval, df)
pval2sided = pval * 2

assert np.allclose((tval, pval2sided),
                   stats.ttest_ind(x, y, equal_var=True))

'''
unequal variance
~~~~~~~~~~~~~~~~
'''

se = np.sqrt(xvar / nx + yvar / ny)

tval = (xbar - ybar) / se

'''
Use the following function to approximate the df needed for the p-value
'''

def unequal_var_ttest_df(v1, n1, v2, n2):
    vn1 = v1 / n1
    vn2 = v2 / n2
    df = (vn1 + vn2)**2 / (vn1**2 / (n1 - 1) + vn2**2 / (n2 - 1))
    return df

df = unequal_var_ttest_df(xvar, nx, yvar, ny)

'''
- Compute the p-value.

- The p-value is one-sided: a two-sided test would test P(T > tval)
  and P(T < -tval). What would be the two sided p-value ?
'''

pval = stats.t.sf(tval, df)
pval2sided = pval * 2


'''
- Compare the two-sided p-value with the one obtained by `stats.ttest_ind`
using `assert np.allclose(arr1, arr2)`
'''
# do it with sicpy
assert np.allclose((tval, pval2sided), stats.ttest_ind(x, y, equal_var=False))

'''
Plot of the two sample t-test
'''
xjitter = np.random.normal(loc=-1, size=len(x), scale=.01)
yjitter = np.random.normal(loc=+1, size=len(y), scale=.01)
plt.plot(xjitter, x, "ob", alpha=.5)
plt.plot(yjitter, y, "ob", alpha=.5)
plt.plot([-1, +1], [xbar, ybar], "or", markersize=15)

#left, left + width, bottom, bottom + height
#plt.bar(left=0, height=se, width=0.1, bottom=ybar-se/2)
## effect size error bar
plt.errorbar(-.1, ybar + (xbar - ybar) / 2, yerr=(xbar - ybar) / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='r')

plt.errorbar([-.8, .8], [xbar, ybar], yerr=np.sqrt([xvar, yvar]) / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='b')

plt.errorbar(.1, ybar, yerr=se / 2, 
             elinewidth=3, capsize=5, markeredgewidth=3,
             color='b')

plt.savefig("/tmp/two_samples_ttest.svg")
plt.clf()

'''
Anova F-test (quantitative ~ categorial (>2 levels))
----------------------------------------------------

Perform an Anova on the following dataset.
- Compute between and within variances
- Compute fval
- Compare the p-value with the one obtained by `stats.f_oneway`
using `assert np.allclose(arr1, arr2)`
'''
import scipy.stats as stats

# dataset
mu_k = np.array([1, 2, 3])    # means of 3 samples
sd_k = np.array([1, 1, 1])    # sd of 3 samples
n_k = np.array([10, 20, 30])  # sizes of 3 samples
grp = [0, 1, 2]               # group labels
n = np.sum(n_k)
label = np.hstack([[k] * n_k[k] for k in [0, 1, 2]])

y = np.zeros(n)
for k in grp:
    y[label == k] = np.random.normal(mu_k[k], sd_k[k], n_k[k])

# Compute with scipy
fval, pval = stats.f_oneway(y[label == 0], y[label == 1], y[label == 2])


# estimate parameters
ybar_k = np.zeros(3)

ybar = y.mean()
for k in grp:
    ybar_k[k] = np.mean(y[label == k])


betweenvar = np.sum([n_k[k] * (ybar_k[k] - ybar) ** 2 for k in grp]) / (len(grp) - 1)
withinvar = np.sum([np.sum((y[label==k] - ybar_k[k]) ** 2) for k in grp]) / (n - len(grp))

fval = betweenvar / withinvar
# Survival function (1 - `cdf`)
pval = stats.f.sf(fval, (len(grp) - 1), n - len(grp))

assert np.allclose((fval, pval), 
                   stats.f_oneway(y[label == 0], y[label == 1], y[label == 2]))



'''
Simple linear regression (one continuous independant variable (IV))
-------------------------------------------------------------------
'''


url = 'https://raw.github.com/neurospin/pystatsml/master/data/salary_table.csv'
salary = pd.read_csv(url)
salary.E = salary.E.map({1:'Bachelor', 2:'Master', 3:'Ph.D'})
salary.M = salary.M.map({0:'N', 1:'Y'})

## Outcome
## S: salaries for IT staff in a corporation.

## Predictors:
## X: experience (years)
## E: education (1=Bachelor's, 2=Master's, 3=Ph.D)
## M: management (1=management, 0=not management)


from scipy.stats as stats
import numpy as np
y, x = salary.S, salary.X
beta, beta0, r_value, p_value, std_err = stats.linregress(x,y)

print("y=%f x + %f  r:%f, r-squared:%f, p-value:%f, std_err:%f" % (beta, beta0, r_value, r_value**2, p_value, std_err))

# plotting the line
yhat = beta * x  +  beta0 # regression line
plt.plot(x, yhat, 'r-', x, y,'o')
plt.xlabel('Experience (years)')
plt.ylabel('Salary')
plt.show()

## Exercise partition of variance formula.

## Compute:
## $\bar{y}$ `y_mu`

y_mu = np.mean(y)

## $SS_\text{tot}$: `ss_tot`

ss_tot = np.sum((y - y_mu) ** 2)

## $SS_\text{reg}$: `ss_reg`
ss_reg = np.sum((yhat - y_mu) ** 2)

## $SS_\text{res}$: `ss_res`
ss_res = np.sum((y - yhat) ** 2)

## Check partition of variance formula based on SS using `assert np.allclose(val1, val2, atol=1e-05)`
assert np.allclose(ss_tot - (ss_reg + ss_res), 0, atol=1e-05)

## What np.allclose does ?

## What assert does

## What is it worth for ?

## Compute $R^2$ and compare with `r_value` above
r2 = ss_reg / ss_tot

assert np.sqrt(r2) == r_value

## Compute F score
n = y.size
fval = ss_reg / (ss_res / (n - 2))

'''
- Compute the p-value:
  * Plot the F(1,n) distribution for 100 f values within [10, 25]. Draw P(F(1,n)>F) ie. color the surface defined by x values larger than F below the F(1,n).
  * P(F(1,n)>F) is the p-value, compute it.
'''

fvalues = np.linspace(10, 25, 100)

plt.plot(fvalues, f.pdf(fvalues, 1, 30), 'b-', label="F(1, 30)")

upper_fval_fvalues = fvalues[fvalues > fval]
plt.fill_between(upper_fval_fvalues, 0, f.pdf(upper_fval_fvalues, 1, 30), alpha=.8)

# pdf(x, df1, df2): Probability density function at x of the given RV.
plt.legend()


# Survival function (1 - `cdf`)
pval = stats.f.sf(fval, 1, n - 2)


## With statmodels
from statsmodels.formula.api import ols
model = ols('S ~ X', salary)
results = model.fit()
print(results.summary())

## sklearn
import sklearn.feature_selection
#sklearn.feature_selection.f_regression??
sklearn.feature_selection.f_regression(x.reshape((n, 1)), y)


'''
Multiple regression
-------------------
'''
import numpy as np
import scipy
np.random.seed(seed=42)  # make the example reproducible

# Dataset
N, P = 50, 4
X = np.random.normal(size= N * P).reshape((N, P))
## Our model needs an intercept so we add a column of 1s:
X[:, 0] = 1
print(X[:5, :])

betastar = np.array([10, 1., .5, 0.1])
e = np.random.normal(size=N)
y = np.dot(X, betastar) + e

# Estimate the parameters
Xpinv = scipy.linalg.pinv2(X)
betahat = np.dot(Xpinv, y)
print("Estimated beta:\n", betahat)

'''
1. What are the dimensions of pinv$(X)$ ?

((P x N) (N x P))^1 (P x N)
P x N
'''
print(Xpinv.shape)


'''
2. Compute the MSE between the predicted values and the true values.
'''

yhat = np.dot(X, betahat)

mse = np.sum((y - yhat) ** 2) / N
print("MSE =", mse)

import scipy.stats as stats
