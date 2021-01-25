---
layout: post
title: "Multiple Linear Regression #5"
subtitle: Update this (Using scikit-learn, statsmodels)
comments: false
---

# Notes for post

### Big header

#### Small header

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

* Include qualitative predictors & dummy encoding of some sort
    * Charles river variable covers this
    * But also use scikit-learn to do the encoding just as an example (maybe with some low/medium/high variable)
    
* Removing the additive assumption: interaction terms (got this one)
* Removing the linear assumption: non-linear relationships (polynomial stuff)
  
* A comparison of scikit-learn, sm and smf

Discuss these 6 problems (maybe not all, but some):
1. Non-linearity of the response-predictor relationships.  <<-- residual plot
2. Correlation of error terms.
3. Non-constant variance of error terms.
4. Outliers.                <<-- maybe studentized residuals
5. High-leverage points.    <<-- leverage statistics
6. Collinearity.            <<-- VIF

# Starting post below

This is the fifth post in a series on the multiple linear regression model.  In previous posts, I've introduced the theory behind the model, exploring using Python's scikit-learn and statsmodels libraries, and discussed potential problems with the model, such as collinearity and correlation of the error terms.

In this post, I'll once again compare scikit-learn and statsmodels, and will explore how to include interaction terms and non-linear relationships in the model.  I'll also discuss nuances and potential problems of the resulting models, and possible areas for improvement using more sophisticated techniques.

### Boston Housing Dataset

I'll make use of the classic Boston Housing Dataset for this working example.  This dataset, originally published by Harrison, D. and Rubinfeld, D.L in 1978 has become one of the more common toy datasets for regression analysis.  The dataset contains information on 506 census tracts around the Boston, MA area, and is available via scikit-learn.

```python
from sklearn import datasets

# Load boston
boston = datasets.load_boston()
```

`datasets.load_boston()` returns a scikit-learn `bunch` object.  We can view information about the dataset via the `DESCR` attribute.

```python
# Printing description
>>> print(boston.DESCR)
.. _boston_dataset:
Boston house prices dataset
---------------------------
**Data Set Characteristics:**  
    :Number of Instances: 506 
    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.
    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's
```

Instead of using the `return_X_y` parameter which returns two numpy arrays, I'll create a pandas DataFrame for ease of use below.

```python
import pandas as pd

X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
y_df = pd.DataFrame(boston.target, columns=["MEDV"])
boston_df = pd.concat([X_df, y_df], axis=1)
```

### Qualitative Predictors & Dummy Encoding

Most of the predictors in the Boston housing dataset are quantitative.  The `CHAS` variable however, indicating if the census tract is bounded by the Charles River or not is qualitative and has been pre-encoded as 0 or 1.

use the first feature to drop the first feature

* For demonstration, going to do the encoding myself using scikit-learn I think preprossing
* won't actually use this variable, but just wanted to explore this functionality of scikit-learn.
* START HERE

### Removing the Additive Assumption: Interaction Terms

### Removing the Linear Assumption: Polynomial Regression
