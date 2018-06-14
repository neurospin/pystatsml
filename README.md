Statistics and Machine Learning in Python
=========================================

This is a draft version !!

Structure
---------

Courses are available in three formats:

1. Jupyter notebooks.

2. Python files using sphinx-gallery.

3. ReStructuredText files.

All notebooks and python files are converted into `rst` format and then assembled together using sphinx.

Build
-----

After pulling the repository execute Jupyter notebooks (outputs are expected to be removed before git submission).
```
make exe
```

Build the pdf file:
```
make pdf
```

Build the html files:
```
make html
```

Dependencies
------------
The easier is to install Anaconda at https://www.continuum.io with python >= 3. Anaconda provides

- python 3
- ipython
- Jupyter
- pandoc


Then install:

1. [sphinx-gallery](https://sphinx-gallery.readthedocs.io)

```
pip install sphinx-gallery
```

2. [nbstripout](https://github.com/kynan/nbstripout)

```
conda install -c conda-forge nbstripout
```

Configure your git repository with nbstripout pre-commit hook for users who don't want to track output in VCS.

```
cd pystatsml
nbstripout --install
```

