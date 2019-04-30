# -*- coding: utf-8 -*-
"""
Brain volumes study
===================

The study provides the brain volumes of grey matter (gm), white matter (wm)
and cerebrospinal fluid) (csf) of 808 anatomical MRI scans.
"""

###############################################################################
# Manipulate data
# ---------------

###############################################################################
# Set the working directory within a directory called "brainvol"
#
# Create 2 subdirectories: `data` that will contain downloaded data and `reports` for results of the analysis.

import os
import os.path
import pandas as pd
import tempfile
import urllib.request

WD = os.path.join(tempfile.gettempdir(), "brainvol")
os.makedirs(WD, exist_ok=True)
os.chdir(WD)

# use cookiecutter file organization
# https://drivendata.github.io/cookiecutter-data-science/
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

###############################################################################
# **Fetch data**
#
# * _Demographic data_ `demo.csv` (columns: `participant_id`, `site`, `group`, `age`, `sex`) and tissue volume data:
#    `group` is Control or Patient.
#    `site` is the recruiting site.
#
# * _Gray matter volume_ `gm.csv` (columns: `participant_id`, `session`, `gm_vol`)
#
# * _White matter volume_ `wm.csv` (columns: `participant_id`, `session`, `wm_vol`)
#
# * _Cerebrospinal Fluid_ `csf.csv` (columns: `participant_id`, `session`, `csf_vol`)

base_url = 'https://raw.github.com/neurospin/pystatsml/master/datasets/brain_volumes/%s'
data = dict()
for file in ["demo.csv", "gm.csv", "wm.csv", "csf.csv"]:
    urllib.request.urlretrieve(base_url % file, "data/%s" % file)

demo = pd.read_csv("data/%s" % "demo.csv")
gm = pd.read_csv("data/%s" % "gm.csv")
wm = pd.read_csv("data/%s" % "wm.csv")
csf = pd.read_csv("data/%s" % "csf.csv")

print("tables can be merge using shared columns")
print(demo.head())
print(gm.head())

###############################################################################
# **Merge tables** according to `participant_id`. Drop row with missing values.

brain_vol = pd.merge(pd.merge(pd.merge(demo, gm), wm), csf)
assert brain_vol.shape == (808, 9)
brain_vol = brain_vol.dropna()

###############################################################################
# **Compute Total Intra-cranial volume**
# `tiv_vol` = `gm_vol` + `csf_vol` + `wm_vol`.

brain_vol["tiv_vol"] = brain_vol["gm_vol"] + brain_vol["wm_vol"] + brain_vol["csf_vol"]

###############################################################################
# ** Compute tissue fractions**
# `gm_f = gm_vol / tiv_vol`, `wm_f  = wm_vol / tiv_vol`.

brain_vol["gm_f"] = brain_vol["gm_vol"] / brain_vol["tiv_vol"]
brain_vol["wm_f"] = brain_vol["wm_vol"] / brain_vol["tiv_vol"]

###############################################################################
# **Save in a excel file `brain_vol.xlsx`**

brain_vol.to_excel("brain_vol.xlsx", sheet_name='data')


###############################################################################
# Statistics
# ----------

###############################################################################
# Load excel file `brain_vol.xlsx`

import os
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smfrmla
import statsmodels.api as sm

brain_vol = pd.read_excel("brain_vol.xlsx", sheet_name='data')


###############################################################################
# **Descriptive statistics**
# Most of participants have several MRI sessions (column `session`)
# Select on rows from session one `"ses-01"

brain_vol1 = brain_vol[brain_vol.session == "ses-01"]

###############################################################################
# Global descriptives statistics of numerical variables

desc_glob_num = brain_vol1.describe()
print(desc_glob_num)

###############################################################################
# Global Descriptive statistics of categorical variable

desc_glob_cat = brain_vol1[["site", "group", "sex"]].describe(include='all')
print(desc_glob_cat)

# I prefer to get count by level
desc_glob_cat = pd.DataFrame({col:brain_vol1[col].value_counts().to_dict() for col in ["site", "group", "sex"]})
print(desc_glob_cat)

###############################################################################
# Remove the single participant from site 6

brain_vol = brain_vol[brain_vol.site != "S6"]
brain_vol1 = brain_vol[brain_vol.session == "ses-01"]
desc_glob_cat = pd.DataFrame({col:brain_vol1[col].value_counts().to_dict() for col in ["site", "group", "sex"]})
print(desc_glob_cat)

###############################################################################
#

###############################################################################
# Visualize site effect of gm ratio using violin plot: site $\times$ gm.

###############################################################################
# Visualize age effect of gm ratio using scatter plot : age $\times$ gm.
#
#8. Linear model (Ancova): gm_ratio ~ age + group + site).



# Global Descriptive statistics of numerical variable





brain_vol1[["site", "group", "sex"]].value_counts()

#
grp_desc = brain_vol1[["site", "group", "sex"]].groupby("site").describe(include='all')

# 5. Descriptive analysis per site in excel file.
with pd.ExcelWriter(os.path.join("reports", "stats_descriptive.xlsx")) as writer:
    glob_desc.to_excel(writer, sheet_name='glob_desc')
    grp_desc.to_excel(writer, sheet_name='grp_desc')

# 4. Visualize site effect of gm ratio using violin plot: site $\times$ gm.
sns.violinplot("site", "gm_f", hue="group", data=brain_vol)

# 5. Visualize age effect of gm ratio using scatter plot : age $\times$ gm.
sns.lmplot("age", "gm_f", hue="group", data=brain_vol[brain_vol.group.notnull()])

# 6. Linear model (Ancova): gm_f ~ age + group + site
twoway = smfrmla.ols('gm_f ~ age + group + site', brain_vol).fit()
aov = sm.stats.anova_lm(twoway, typ=2) # Type 2 ANOVA DataFrame

print("= Anova =")
print(aov)

print("= Parameters =")
print(twoway.params)

print("%.2f%% of grey matter loss per year (almost %.0f%% per decade)" %\
      (twoway.params.age * 100, twoway.params.age * 100 * 10))