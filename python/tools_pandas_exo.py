# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:26:05 2016

@author: edouard.duchesnay@cea.fr
"""

from __future__ import print_function

'''

Exercises
---------

Data Frame
~~~~~~~~~~

1. Read the iris dataset at 'https://raw.github.com/neurospin/pystatsml/master/data/iris.csv'

2. Print column names

3. Get numerical columns

4. For each species compute the mean of numerical columns and store it in  a ``stats`` table like:

::

          species  sepal_length  sepal_width  petal_length  petal_width
    0      setosa         5.006        3.428         1.462        0.246
    1  versicolor         5.936        2.770         4.260        1.326
    2   virginica         6.588        2.974         5.552        2.026


'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://raw.github.com/neurospin/pystatsml/master/data/iris.csv'
df = pd.read_csv(url)

num_cols = df._get_numeric_data().columns

stats = list()

for grp, d in df.groupby("species"):
    stats.append( [grp] + d.ix[:, num_cols].mean(axis=0).tolist())

stats = pd.DataFrame(stats, columns=["species"] + num_cols.tolist())
print(stats)
