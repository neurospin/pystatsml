gh-pages
--------

TODO: do it with: circleci

- https://circleci.com/blog/deploying-documentation-to-github-pages-with-continuous-integration/
- https://github.com/jklukas/docs-on-gh-pages


Publishing sphinx-generated docs on github:

https://daler.github.io/sphinxdoc-test/includeme.html



Upload to github
----------------


"$WD/build/html" contains the pystsamsl website. Now we start to upload to github server. Clone from github to a temporary directory, and checkout gh-pages branch

```
WD=~/git/pystatsml
cd /tmp
git clone git@github.com:duchesnay/pystatsml.git pystatsml_doc
cd pystatsml_doc
git symbolic-ref HEAD refs/heads/gh-pages
rm .git/index
git clean -fdx
cp -r $WD/build/html/* ./
cp -r $WD/auto_gallery ./
git add .
git add -f auto_gallery
git add -f _sources
git add -f _static
git add -f _images
touch .nojekyll
git commit -am "gh-pages First commit"
git push origin gh-pages
firefox  $WD/build/html/index.html
```

Then
```
gedit index.html

Replace:
```
  <div class="section" id="phantom">
<h1>Phantom<a class="headerlink" href="#phantom" title="Permalink to this headline">¶</a></h1>
</div>
```
by

```
<div class="section" id="phantom">
<h1 style="font-weight:bold;">Statistics and Machine Learning in
Python<a class="headerlink" href="#phantom" title="Permalink to this headline">¶</a></h1>
</div>

<hr>

<p><a href="https://duchesnay.github.io/">Edouard Duchesnay</a>, <a href="https://www.umu.se/en/staff/toklot02/">Tommy Löfstedt</a>, Feki Younes</p>

Links:
<ul>
  <li><a href="https://github.com/duchesnay/pystatsml">Github</a></li>
  <li><a href="ftp://ftp.cea.fr/pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPython.pdf">Pdf</a></li>
</ul>
```

Then

```
git commit -am "Title and authors"
git push origin gh-pages
firefox  $WD/build/html/index.html
```

Now, you can visit your updated website at https://duchesnay.github.io/pystatsml.


ML Resources
------------

- **Practical Machine Learning Course Notes (in R)**
    https://sux13.github.io/DataScienceSpCourseNotes/8_PREDMACHLEARN/Practical_Machine_Learning_Course_Notes.html

- **Computational Statistics in Python**
    https://people.duke.edu/~ccc14/sta-663/index.html

- **scipy-lectures**

    https://github.com/scipy-lectures/scipy-lecture-notes

- **Scientific Python & Software engineering best practices**
    https://github.com/paris-saclay-cds/python-workshop

- **Deep Learning course in python**
    https://github.com/m2dsupsdlclass/lectures-labs

- **Others**
    https://github.com/justmarkham/DAT4

    http://statweb.stanford.edu/~jtaylo/courses/stats202/index.html

    http://www.dataschool.io/

    https://onlinecourses.science.psu.edu/stat857/node/141

    https://github.com/rasbt/python-machine-learning-book

    https://onlinecourses.science.psu.edu/stat505/

    http://www.kdnuggets.com/2016/04/top-10-ipython-nb-tutorials.html


A gallery of interesting IPython Notebooks
------------------------------------------

https://github.com/ipython/ipython/wiki/A-gallery-of-interesting-IPython-Notebooks

IPython notebooks
-----------------

- https://ipython.org/ipython-doc/3/notebook/index.html

- http://nbviewer.ipython.org/github/ipython/ipython/tree/1.x/examples/

- http://www.kevinsheppard.com/Python_Course/IPython_Notebooks

- Shortcuts: http://johnlaudun.org/20131228-ipython-notebook-keyboard-shortcuts/

Markdown
--------
http://daringfireball.net/projects/markdown/basics

R with Jupyther
~~~~~~~~~~~~~~~

conda install -c r r-essentials

Sphinx
------

http://sphinx-doc.org/

IPython notebooks + Sphinx
--------------------------

http://sphinx-ipynb.readthedocs.org/en/latest/howto.html


nbsphinx: Jupyter Notebook Tools for Sphinx

https://nbsphinx.readthedocs.io/en/0.3.3/

nbsphinx is a Sphinx extension that provides a source parser for *.ipynb files. Custom Sphinx directives are used to show Jupyter Notebook code cells (and of course their results) in both HTML and LaTeX output. Un-evaluated notebooks – i.e. notebooks without stored output cells – will be automatically executed during the Sphinx build process.

conda install -c conda-forge nbsphinx

sphinx-gallery
--------------

https://sphinx-gallery.readthedocs.io/en/latest/

``pip install sphinx-gallery``

http://www.scipy-lectures.org

https://github.com/scipy-lectures/scipy-lecture-notes

strip jupyter output before submission
--------------------------------------

https://github.com/kynan/nbstripout

``conda install -c conda-forge nbstripout``

Set up the git filter and attributes as described in the manual installation instructions below:

``cd pystatsml``
``nbstripout --install``


rst
---

http://docutils.sourceforge.net/rst.html
http://docutils.sourceforge.net/docs/ref/rst/



R vs Python
-----------

https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis
http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html

Mail to share the course
------------------------

Please find the link to my Machine Learning course in Python, it is a draft version:
ftp://ftp.cea.fr//pub/unati/people/educhesnay/pystatml/StatisticsMachineLearningPython.pdf

Below the link to github:
https://github.com/duchesnay/pystatsml


git clone https://github.com/duchesnay/pystatsml.git


Basically, it uses Jupyter notebook and pure python, everything is converted to rst and assembled to html or pdf using sphynx.

It is a draft version, not finished yet with many spelling mistakes.

Please fork and perform some pull request. If you are willing to contribute.



