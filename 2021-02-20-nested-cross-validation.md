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

Nested cross-validation can be viewed as an extension of simpler cross-validation techniques.  When performing model selection or model evaluation, $k$-fold cross-validation is a crucial method for estimating a particular model's test error on unseen observations.  However, as [Cawley and Talbot discussed in a 2010 paper](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf), when performing model selection *and* model evaluation, we should not use the same test sets to *both* select the hyperparameters of a model *and* evaluate a model.  By doing so, we may optimistically bias our model evaluations and underestimate our estimated test errors.

For clarity, hyperparameters are parameters that not determined directly within the model's learning procedures, and thus must be defined prior to model fitting by the user.  In scikit-learn, hyperparameters are passed as arguments to the constructor of the estimator class.  For example, when performing regularized logistic regression, we can define the regularization hyperparameter `C` via `LogisticRegression(C=0.01)`.

Using nested cross-validation, we are able to both select the hyperparameters of a model and evaluate the model on the same initial dataset.  We accomplish by performing two sequential $k$-fold cross-validation procedures.  

* we first get the training/test split on entire dataset
* then for training set do grid search, which is CV as well, to select hyperparameters
* then select best model from grid search
* and evaluate this model on a CV of test sets

We first perform an *inner* $k$-fold cross-validation procedure to select the optimal hyperparamters of our model.  For a subset of trai

### Breast Cancer Data

For this working example, I'll use the common breast cancer dataset available within scikit-learn's `datasets` module.  The breast cancer dataset was assembled by the University of Wisconsin and contains 569 observations and 30 predictor features.  The target variable is encoded as `1` or `0`, indicating whether the observation was found to have malignant or benign breast cancer.

```python
from sklearn import datasets
X, y = datasets.load_breast_cancer(return_X_y=True)
```

For this working example, let's find the optimal value of `C` for a regularized logistic regression model.  









