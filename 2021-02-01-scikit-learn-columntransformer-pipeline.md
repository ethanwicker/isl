---
layout: post
title: "scikit-learn: `ColumnTransformer` and `Pipeline`"
subtitle: Summarizing an Effective Workflow with pandas
comments: false
---

I recently read through this [excellent Medium article](https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62) about the `ColumnTransformer` class in scikit-learn and how it can be used in tandem with `Pipeline`s and the `OneHotEncoder` class.  To strengthen my own understanding of the concept, I decided to follow the post with my own working example, and summarize the concepts along the way.

### Introduction

In scikit-learn's 0.20 release, the `ColumnTransformer` estimator was released.  This estimator allows different transformers to be applied to different fields of the data in parallel, before concatenating them together.  In particular, this estimator is attractive when processing a pandas DataFrame with categorical fields.

For this working example, I'll be using the same slimmed down Titanic dataset from my previous [logistic regression post](https://ethanwicker.com/2021-01-27-logistic-regression-002/).

Below, I'll make use of the `ColumnTransformer` estimator to encode two categorical fields via scikit-learn's `OneHotEncoder`.  Because scikit-learn machine learning models require their input to be two-dimensional numerical arrays, an encoding preprocessing step is required.

I'll also use `SimpleImputer` estimator to preform some preprocessing of numerical fields.  Lastly, I'll perform a ridge regression and wrap all of these steps into a reusable and convenient `Pipeline`.

### OneHotEncoder

Let's first encode `sex` to demonstrate some functionality of `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder

# Initializing one-hot encoder
# Forcing dense matrix to be returned
encoder = OneHotEncoder(sparse=False)

# Encoding categorical field
X_categorical = titanic[["sex"]]
X_categorical = encoder.fit_transform(X_categorical)

>>> X_categorical
array([[0., 1.],
       [1., 0.],
       [1., 0.],
       ...,
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

From the output, we can see that the `male` and `female` values of `sex` have been encoded into two binary columns.

Notice that a NumPy array was returned.  We can access column names indicating which feature of `sex` is represented by each column using the `get_feature_names()` method.

```python
>>> feature_names = encoder.get_feature_names()
>>> feature_names
array(['x0_female', 'x0_male'], dtype=object)
```

We can also use the `inverse_transform()` method to return the original categorical label from the `sex` column.  Notice the brackets around `X_categorical[0]` that force a list to returned, instead of a NumPy array.

```python
# Inverse transforming the first row of X_categorical
encoder.inverse_transform([X_categorical[0]])
```



