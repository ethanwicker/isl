---
layout: post
title: "Multiple Linear Regression #5"
subtitle: Update this (Using scikit-learn, statsmodels, plotly)
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

I'll make use of the encoded `CHAS` variable in the models below.  However, for demonstration and to familiarize myself with procedure, below I create a new categorical field `crime_label` and encode it via scikit-learn's `OneHotEncoder()` class. 

```python
from sklearn.preprocessing import OneHotEncoder

# Creating crime_label field
boston_df = \
    (boston_df
     .assign(
        crime_label=pd.cut(boston_df["CRIM"],
                           bins=3,
                           labels=["low_crime", "medium_crime", "high_crime"]))
    )

# Converting crime_label field to NumPy array
crime_labels_ndarray = boston_df["crime_label"].to_numpy().reshape(-1, 1)

# Defining encoder
encoder = OneHotEncoder()

# Fitting encoder on array, and transforming
crime_labels_encoded = encoder.fit_transform(crime_labels_ndarray)

# Converting encoded array to DataFrame
crime_labels_df = pd.DataFrame(data=crime_labels_encoded.toarray(),
                               columns=encoder.get_feature_names())

# Concatenating with boston_df
boston_df = pd.concat(objs=[boston_df, crime_labels_df], axis=1)
```

`boston_df` now contains three addition fields, indicating the encoded values of `crime_label`.

```python
>>> boston_df.head()
      CRIM    ZN  INDUS  ...  x0_high_crime  x0_low_crime  x0_medium_crime
0  0.00632  18.0   2.31  ...            0.0           1.0              0.0
1  0.02731   0.0   7.07  ...            0.0           1.0              0.0
2  0.02729   0.0   7.07  ...            0.0           1.0              0.0
3  0.03237   0.0   2.18  ...            0.0           1.0              0.0
4  0.06905   0.0   2.18  ...            0.0           1.0              0.0
```

Note, it is common in some use cases to drop an encoded column, as this column can be inferred explicitly from the other columns.  This can be accomplished by passing `drop="first` to `OneHotEncoder()`.  

```python
# Could also use this to drop the first column
# Necessary when fitting unregularized linear models, so as to not create linear dependencies
encoder = OneHotEncoder(drop="first")
```

It is also possible to achieve a similar result using pandas `get_dummies()`.

```python
pd.get_dummies(boston_df["crime_label"], drop_first=True)
```

### Removing the Additive Assumption: Interaction Terms

Next, I'll explore relaxing the additive assumption of the multiple linear regression model.  In particular, I'll train a model on `MEDV` versus `ZN`, `CHAS` and `RM`.  For ease of use, user readability, and statistical inference results, I'll use the formula interface provided by statsmodels below.

Let's first train a model with no interaction terms, for comparison purposes.
```python
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula="MEDV ~ ZN + CHAS + RM", data=boston_df)
>>> result = model.fit()
>>> result.summary()

<class 'statsmodels.iolib.summary.Summary'>
"""
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                   MEDV   R-squared:                       0.522
Model:                            OLS   Adj. R-squared:                  0.519
Method:                 Least Squares   F-statistic:                     182.5
Date:                Mon, 25 Jan 2021   Prob (F-statistic):           5.18e-80
Time:                        08:35:33   Log-Likelihood:                -1653.6
No. Observations:                 506   AIC:                             3315.
Df Residuals:                     502   BIC:                             3332.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -30.4697      2.654    -11.480      0.000     -35.684     -25.255
ZN             0.0666      0.013      5.182      0.000       0.041       0.092
CHAS           4.5212      1.126      4.017      0.000       2.310       6.733
RM             8.2635      0.428     19.313      0.000       7.423       9.104
==============================================================================
Omnibus:                      116.852   Durbin-Watson:                   0.757
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              675.350
Skew:                           0.866   Prob(JB):                    2.24e-147
Kurtosis:                       8.388   Cond. No.                         248.
==============================================================================
Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""
```

From the model summary, we see that all the variables are significant.  Let's take a look at the residual plot to see if any patterns standout.

Using formula interface for ease here and statsmodels for stastical inference

Should do it with no interaction terms
Then look at residual plot
Then add interaction term

```python
import statsmodels.formula.api as smf

# Adding interaction term
# R^2 is 0.537
# Discuss : vs. * here
# All terms with CHAS no longer significant, so dropping
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + ZN:CHAS + CHAS:RM + ZN:RM", data=boston_df)
result = model.fit()
result.summary()
```


### Removing the Linear Assumption: Polynomial Regression
