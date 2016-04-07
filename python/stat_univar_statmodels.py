from __future__ import print_function

#import matplotlib.pyplot as plt
#from statsmodels.sandbox.regression.predstd import wls_prediction_std

np.random.seed(42)

'''
Ordinary Least Squares
======================
'''

'''
Numpy
-----
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
Linear model with statsmodel
----------------------------
'''

'''
Interfacing with numpy
~~~~~~~~~~~~~~~~~~~~~~
'''
import statsmodels.api as sm

## Fit and summary:
model = sm.OLS(y, X).fit()
print(model.summary())

# prediction of new values
ypred = model.predict(X)

# residuals + prediction == true values
assert np.all(ypred + model.resid == y)

'''
Interfacing with Pandas
~~~~~~~~~~~~~~~~~~~~~~
'''
import statsmodels.formula.api as smfrmla
# Build a dataframe excluding the intercept
df = pd.DataFrame(np.column_stack([X[:, 1:], y]), columns=['x1','x2', 'x3', 'y'])


## Fit and summary:
model = smfrmla.ols("y ~ x1 + x2 + x2", df).fit()
print(model.summary())



oneway = smfrmla.ols('salary ~ management + experience', salary).fit()

twoway = smfrmla.ols('salary ~ education + management + experience', salary).fit()

sm.stats.anova_lm(oneway, twoway)
twoway.compare_f_test(oneway)

oneway = smfrmla.ols('salary ~ management + experience', salary).fit()
oneway.model.data.param_names
oneway.model.data.exog

print(twoway.model.data.param_names)
print(twoway.model.data.exog[:10, :])

ttest_exp = oneway.t_test([0, 0, 1])
ttest_exp.pvalue, ttest_exp.tvalue
print(ttest_exp)

# Alternatively, you can specify the hypothesis tests using a string
oneway.t_test('experience')

'''
multiple comparison
'''

import numpy as np
np.random.seed(seed=42)  # make example reproducible

# Dataset
import numpy as np
np.random.seed(seed=42)  # make example reproducible


# Dataset
n_samples, n_features = 100, 1000
n_info = int(n_features/10) # number of features with information
n1, n2 = int(n_samples/2), n_samples - int(n_samples/2)
snr = .5
Y = np.random.randn(n_samples, n_features)
grp = np.array(["g1"] * n1 + ["g2"] * n2)

# Add some group effect for Pinfo features
Y[grp=="g1", :n_info] += snr

# 
import scipy.stats as stats
import matplotlib.pyplot as plt
tvals, pvals = np.full(n_features, np.NAN), np.full(n_features, np.NAN)
for j in range(n_features):
    tvals[j], pvals[j] = stats.ttest_ind(Y[grp=="g1", j], Y[grp=="g2", j], equal_var=True)

fig, axis = plt.subplots(3, 1)#, sharex='col')

axis[0].plot(range(n_features), tvals, 'o')
axis[0].set_ylabel("t-value")

axis[1].plot(range(n_features), pvals, 'o')
axis[1].axhline(y=0.05, color='red', linewidth=3, label="p-value=0.05")
#axis[1].axhline(y=0.05, label="toto", color='red')
axis[1].set_ylabel("p-value")
axis[1].legend()

axis[2].hist([pvals[n_info:], pvals[:n_info]], 
    stacked=True, bins=100, label=["Negatives", "Positives"])
axis[2].set_xlabel("p-value histogram")
axis[2].set_ylabel("density")
axis[2].legend()

plt.tight_layout()



'''
No correction
'''
P, N = n_info,  n_features - n_info # Positives, Negatives
TP = np.sum(pvals[:n_info ] < 0.05)  # True Positives
FP = np.sum(pvals[n_info: ] < 0.05)  # False Positives
print("No correction, FP: %i (exepected: %.2f), TP: %i" % (FP, N * 0.05, TP))


'''
False negative rate (FNR)
    FNR} = FN} / (TP} + FN}) = 1-TPR}
'''
FNR = 
print("No correction, false positives: %i (exepected value: %i)" % (FP, 0.05 * (n_features - TP)))



## Bonferoni
import statsmodels.sandbox.stats.multicomp as multicomp
_, pvals_fwer, _, _  = multicomp.multipletests(pvals, alpha=0.05, 
                                               method='bonferroni')
TP = np.sum(pvals_fwer[:n_info ] < 0.05)  # True Positives
FP = np.sum(pvals_fwer[n_info: ] < 0.05)  # False Positives
print("FWER correction, FP: %i, TP: %i" % (FP, TP))


## FDR
import statsmodels.sandbox.stats.multicomp as multicomp
_, pvals_fdr, _, _  = multicomp.multipletests(pvals, alpha=0.05, 
                                               method='fdr_bh')
TP = np.sum(pvals_fdr[:n_info ] < 0.05)  # True Positives
FP = np.sum(pvals_fdr[n_info: ] < 0.05)  # False Positives

print("FDR correction, FP: %i, TP: %i" % (FP, TP))

'''
Binary classif measures:

- **Sensitivity** or **true positive rate (TPR)**, eqv. with hit rate, recall:

    TPR = TP / P = TP / (TP+FN)
    
- specificity (SPC) or true negative rate

    SPC = TN / N = TN / (TN+FP) 

- precision or positive predictive value (PPV)

    PPV = TP / (TP + FP)

- negative predictive value (NPV)

    NPV = TN / (TN + FN)

- fall-out or **false positive rate (FPR)** 

    FPR = FP / N = FP / (FP + TN) = 1-SPC
    
    
- false negative rate (FNR)

    FNR = FN / (TP + FN) = 1-TPR

- false discovery rate (FDR)

    FDR = FP / (TP + FP) = 1 - PPV 

'''