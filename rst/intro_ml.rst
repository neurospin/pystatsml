
Introduction to Machine Learning
================================


Machine learning within data science
------------------------------------

.. image:: images/Data_Science_VD.png
   :scale: 50
   :align: center

Machine learning covers two main types of data analysis:

1. Exploratory analysis: **Unsupervised learning**. Discover the structure within the data. E.g.: Experience (in years in a company) and salary are correlated.
2. Predictive analysis: **Supervised learning**. This is sometimes described as to **"learn from the past to predict the future"**. Scenario: a company wants to detect potential future clients among a base of prospect. Retrospective data analysis: given the base of prospected company (with their characteristics: size, domain, localization, etc.) some became clients, some do not. Is it possible to learn to predict those that are more likely to become clients from their company characteristics? The training data consist of a set of *n* training samples. Each sample, :math:`x_i`, is a vector of *p* input features (company characteristics) and a target feature (:math:`y_i \in \{Yes, No\}` (whether they became a client or not).


IT/computing science tools
--------------------------

    - High Performance Computing (HPC)
    - Data flow, data base, file I/O, etc.
    - Python: the language.
    - Numpy: raw numerical data for computation.
    - Pandas: structured data, I/O etc.


Statistics and applied mathematics
----------------------------------

    - Linear model.
    - Non parametric statistics.
    - Linear algebra: matrix operations, inversion, eigenvalues.


Data analysis methodology
-------------------------

1. Formalize customer's needs into a learning problem:
    * A target variable: supervised problem.
        - Target is qualitative: classification.
        - Target is qualitative: regression.
    * No target variable: unsupervised problem
        - Vizualisation of high-dimensional samples: PCA, manifolds learning, etc.
        - Finding groups of samples (hidden structure): clustering.

2. Ask question about the datasets
    * Number of samples
    * Number of variables, types of each variable.

3. Define the sample
    * For prospective study formalize the experimental design: inclusion/exlusion criteria. The conditions that define the acquisition of the dataset.
    * For retrospective study formalize the experimental design: inclusion/exlusion criteria. The conditions that define the selection of the dataset.

4. In a document formalize (i) the project objectives; (ii) the required learning dataset; More specifically the input data and the target variables. (iii)  In this document warm the customer that the learned algorithms may not work on new data acquired under different condition.

5. Read the learning dataset.

6. (i) Sanity check (basic descriptive statistics); (ii) data cleaning (impute missing data, recoding); Final Quality Control (QC) perform descriptive statistics and think ! (remove possible confounding variable, etc.).

7. Explore data (visualization, PCA) and perform basics univariate statistics for association between the target an input variables.

8. Perform more complex multivariate-machine learning.

9. Model validation. First deliverable: the predictive model with performance on training dataset.

10. Apply on new data.

