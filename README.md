Statistics and Machine Learning in Python
=========================================

This is a draft version !!

Structure
---------

Courses are avalaible in three formats:

1. Python files in the [scientific_python/examples](https://github.com/neurospin/pystatsml/tree/master/scientific_python/examples). The sphinx-gallery is used to transform python files into rst documents.

2. Ipython notebooks files in the  in the [notebooks](https://github.com/neurospin/pystatsml/tree/master/notebooks) directory.

3. ReStructuredText files in the [rst](https://github.com/neurospin/pystatsml/tree/master/rst) directory.

All notebooks and python files are converted into `rst` format and then assembled together using sphinx.

Build
-----
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

- [sphinx-gallery](https://sphinx-gallery.readthedocs.io)

```
pip install sphinx-gallery
```

- [nbstripout](https://github.com/kynan/nbstripout)

```
conda install -c conda-forge nbstripout
```

Configure your git repository with nbstripout pre-commit hook for users who don't want to track output in VCS.

```
cd pystatsml
nbstripout --install
```

