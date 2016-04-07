
Introduction to Machine Learning
================================


Machine Learning within data science
------------------------------------

.. image:: images/Data_Science_VD.png
   :scale: 50
   :align: center

ML cover two main types of data analysis:

1. Exploratory analysis: **Unsupervised learning**. 
Discover the structure within the data. Eg.: Experience (in years in a company) and salary are correlated.

2. Predictive analysis **Supervised learning**. **"learn from the past to predict the future"**.
Scenario: a company wants to detect potential future client among a base of prospect. Retrospective data analysis: given the base of prospected company (with their characteristics: size, domain, localization) some became client some not. Is it possible to learn to predict those which are more likely to become client from the company characteristics?  The training data consist of a set of *n* training samples. Each sample :math:`x_i` is a vector of *p* input features (company characteristics) and a target feature (:math:`y_i \in \{Yes, No\}` (became client). 


IT/computing science tools
--------------------------

    - Python: the language
    - Numpy: raw numerical data
    - Pandas: structured data

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
    * Goto 1. until convergence of both sides (you and the customer).

2. In a document formalize (i) the project objectives; (ii) the required learning dataset; More specifically the input data and the target variables. (iii) The conditions that define the acquisition of the dataset. In this document warm the customer that the learned algorithms may not work on new data acquired under different condition.

3. Read your learning dataset (level D of the pyramid) provided by the customer.

4. Clean your data (QC: Quality Control) (reach level I of the pyramid).

5. Explore data (visualization, PCA) and perform basics univariate statistics (reach level K of the pyramid).

7. Perform more complex multivariate-machine learning.

8. Model validation. First deliverable: the predictive model with performance on training dataset.

9. Apply on new data (level W of the pyramid).

