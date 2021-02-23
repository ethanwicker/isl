---
layout: post
title: "Nested Cross-Validation"
subtitle: An Overview and scikit-learn Example
comments: false
---

### Big Header

### Small Header

The structure of this post was influenced by the third chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

| ![2021-01-08-multiple-linear-regression-001-fig-1.png](/assets/img/2021-01-08-multiple-linear-regression-001-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

### Start Here

Include this: Great video: https://www.youtube.com/watch?v=az60jS7MQhU by Dr. Cynthia Rudin at Duke University (https://en.wikipedia.org/wiki/Cynthia_Rudin)

Nested cross-validation can be viewed as an extension of simpler cross-validation techniques.  When performing model selection or model evaluation, $k$-fold cross-validation is a crucial method for estimating a particular model's test error on unseen observations.  However, as [Cawley and Talbot discussed in a 2010 paper](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), when performing model selection *and* model evaluation, we should not use the same test sets to *both* select the hyperparameters of a model *and* evaluate a model.  By doing so, we may optimistically bias our model evaluations and underestimate our estimated test errors.

For clarity, hyperparameters are parameters that not determined directly within the model's learning procedures, and thus must be defined prior to model fitting by the user.  In scikit-learn, hyperparameters are passed as arguments to the constructor of the estimator class.  For example, when performing regularized logistic regression, we can define the regularization hyperparameter `C` via `LogisticRegression(C=0.01)`.

Using nested cross-validation, we are able to both select the hyperparameters of a model and evaluate the model on the same initial dataset without optimistically biasing our model evaluations.  We accomplish nested cross-validation by performing two sequential, or nested, $k$-fold cross-validation loops.

* we first get the training/test split on entire dataset
* then for training set do grid search, which is CV as well, to select hyperparameters
* then select best model from grid search
* and evaluate this model on a CV of test sets

To begin the nested cross-validation procedure, we first split our entire dataset into $k$ outer folds, where one fold is a test set and the remaining $k-1$ folds are our outer training set.  We then take the outer training set and further split it into $l$ inner folds, where one fold is an inner validation set and the remaining $l-1$ folds are our inner training set.  Next, we train each individual particular model with different hyperparameters on the inner training set and evaluate it on the inner validation set.  We then take the best performing model on the inner validation set, and evaluate it on the outer test set.

We then repeat the above process, looping through the outer $k$ folds until each $k_i$th fold is used as a test set once.  Thus, we have performed an outer $k$-fold cross-validation loop on our entire dataset, and an inner $l$-fold cross-validation loop for each outer training set made up of $k-1$ folds.

After performing this nested cross-validation procedure, we typically report the mean and standard deviation of the test error estimates from the outer $k$-fold cross-validation procedure.

Next, let's explore a working example to clarify the nested cross-validation technique.

### Breast Cancer Data

For this working example, I'll use the common breast cancer dataset available within scikit-learn's `datasets` module.  The breast cancer dataset was assembled by the University of Wisconsin and contains 569 observations and 30 predictor features.  The target variable is encoded as `1` or `0`, indicating whether the observation was found to have malignant or benign breast cancer.

```python
from sklearn import datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
```

On the breast cancer dataset, let's train a regularized logistic regression model with multiple values of `C`.  In a regularized logistic regression setting, `C` controls the penalty, or cost function, the penalizes the model for variable complexity.  `C` commonly takes values ranging from 0.0001 to 10000.

Because we're performing regularization, we should also scale our input features.
```python
from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

X = standard_scaler.fit_transform(X)
```

### Inner $l$-Fold Cross-Validation Loop (Hyperparamter Tuning)

To perform our inner $l$-Fold Cross-Validation procedure, we'll first construct a `LogisticRegression` estimator and a `GridSearchCV` object, both from scikit-learn.

We'll construct our grid of `C`s as 10 values between 0.0001 and 10000, evenly spaced on a logarithmic scale.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Creating grid of potential C values
# Could have also used dict(C=(0.0001, 0.001, 0.01, ... , 1000)) to specify by hand
C_grid = dict(C=np.logspace(-4, 4, 10))

# Initializing estimator
log_reg = LogisticRegression()

# Creating GridSearch object with logistic regression classifier and candidate C values
# cv = 5, defaults to 5-fold cross-validation
log_reg_grid = GridSearchCV(estimator=log_reg, param_grid=C_grid)
```

Note, we could have also used `RandomSearchCV` instead of `GridSearchCV` above.  While both objects perform cross-validation, `GridSearchCV` is an exhaustive search of all parameter combinations in `param_grid`, while `RandomSearchCV` implements a randomized search of the hyperparamter space.  Within `RandomSearchCV`, the `n_iter` argument controls how many random combinations of parameters to search over.  A random search, as opposed to an exhaustive grid search, can be understood as trading runtime for solution quality.

At this point, nothing else needs to be done to the `log_reg_grid` object that represents the inner loop of our nested cross-validation procedure.  However, to demonstrate this object can be used to find the optimal value of `C` (of those defined in `C_grid`), we can fit the object on `X` and `y`.

```python
# Fitting grid search object
log_reg_grid.fit(X=X, y=y)

# Getting optimal value of C
log_reg_grid.best_estimator_.C  # 0.3593813663804626
```

### Outer $k$-Fold Cross-Validation Loop (Model Evaluation)

Now that we have constructed our inner $l$-fold cross-validation loop using `GridSearchCV` above, we can implement it into an outer $k$-fold cross-validation loop using scikit-learn's `cross_val_score`.  Here, `cross_val_score` will iteratively perform cross-validation on the outer $k$ folds of our data to determine how well the optimal model with a particular `C` value found on the inner loop is performing.

Below, let's perform an outer loop with `cv=3` folds.  `cross_val_score` will iteratively split the data into three folds, where one fold is a test set and the other two folds are a training set.  Then, our `log_reg_grid` search object will further perform 5-fold cross-validation on this training set to determine the optimal value of `C`, and this model with the optimal value of `C` is evaluating on the test set.  This process in then repeated until each outer fold is used as the test set once.

```python
>>> from sklearn.model_selection import cross_val_score

>>> scores = cross_val_score(estimator=log_reg_grid, X=X, y=y, cv=3)
>>> scores
array([0.97894737, 0.97368421, 0.97883598])
```

Each of the above three values represent our unbiased test error estimate for our regularized logistic regression model.  We can calculate their mean to evaluate how well our model performs **with hyperparameter tuning as a part of the fitting procedure**.

```python
>>> scores.mean()
0.977155852594449
```

### Drawbacks of Nested Cross-Validation

Nested CV is computationally expensive --> useful when you have a few thousand observations or less
For a large dataset, nested CV isn't as useful

### So What is the Final Model?

DOING normal k-FOLD AGAIN for hyperparamter tuning.  nested cv used to evaluate models with hyperparameter tuning as a part of the evaluation process








