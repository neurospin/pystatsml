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

- python 3
- ipython
- Jupyter

The easier is to install Anaconda at https://www.continuum.io with python >= 3

- sphinx-gallery
```
pip install sphinx-gallery
```

- pandoc

For Linux (gnome based distributions):

Install pandoc:
```
sudo apt-get install pandoc
```
If you don't already have pip:
```
sudo apt-get install python3-pip
```
Install Jupyter:
```
sudo -H pip3 install jupyter
```

Others
------

sudo apt-get install latexmk texlive-latex-extra

