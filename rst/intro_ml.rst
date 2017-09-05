
Introduction to Machine Learning
================================


Machine learning within data science
------------------------------------

.. image:: images/Data_Science_VD.png
   :scale: 50
   :align: center

Machine learning covers two main types of data analysis:

1. Exploratory analysis: **Unsupervised learning**. Discover the structure within the data. E.g.: Experience (in years in a company) and salary are correlated.
2. Predictive analysis: **Supervised learning**. This is sometimes described as **"learn from the past to predict the future"**. Scenario: a company wants to detect potential future clients among a base of prospects. Retrospective data analysis: we go through the data constituted of previous prospected companies, with their characteristics (size, domain, localization, etc...). Some of these companies became clients, others did not. The question is, can we possibly predict which of the new companies are more likely to become clients, based on their characteristics based on previous observations? In this example, the training data consists of a set of *n* training samples. Each sample, :math:`x_i`, is a vector of *p* input features (company characteristics) and a target feature (:math:`y_i \in \{Yes, No\}` (whether they became a client or not).


IT/computing science tools
--------------------------

    - Python: the programming language
    - Numpy: python library particularly useful for handling of raw numerical data (matrices, mathematical operations)
    - Pandas: python library adept at handling sets of structured data: list, tables

Statistics and applied mathematics
----------------------------------

    - Linear model
    - Non parametric statistics
    - Linear algebra: matrix operations, inversion, eigenvalues.


Data analysis methodology
-------------------------

DIKW Pyramid: Data, Information, Knowledge, and Wisdom
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. image:: images/Wisdom-Knowledge-Information-Data-Pyramid15.png
   :scale: 50
   :align: center

Methodology
~~~~~~~~~~~

1. Discuss with your customer:

    * Understand his needs.
    * Formalize his needs into a learning problem.
    * Define with your customer the learning dataset required for the project .
    * Goto 1. until convergence of both sides (the customer and you).

2. In a document formalize (i) the project objectives; (ii) the required learning dataset (more specifically the input data and the target variables); (iii) The conditions that define the acquisition of the dataset. In this document, warn the customer that the learned algorithms may not work on new data acquired under different condition.

3. Read your learning dataset (level D of the pyramid) provided by the customer.

4. Clean your data (QC: Quality Control) (reach level I of the pyramid).

5. Explore data (visualization, PCA) and perform basic univariate statistics (reach level K of the pyramid).

7. Perform more complex multivariate-machine learning.

8. Model validation. Primary test: the predictive model with perform on training dataset.

9. Apply on new data (level W of the pyramid).

