#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 13:52:03 2019

@author: edouard
"""

## TODO rewrite test function
"""
Manual check, run command line

nb=tests/test_notebook.ipynb
rst=tests/test_notebook.rst

# Run notebook
jupyter nbconvert --to notebook --execute $nb --output $(basename $nb)

# Convert to rst
jupyter nbconvert --to rst --stdout $nb
jupyter nbconvert --to rst --stdout $nb | bin/filter_fix_rst.py > $rst
"""