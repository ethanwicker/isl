---
layout: post
title: "Multiple Logistic Regression #2"
subtitle: scikit-learn, statsmodels, Plotly, One-Hot Encoding & Multiclass Logistic Regression 
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


![2021-01-08-multiple-linear-regression-001-fig-1.png](/assets/img/2021-01-08-multiple-linear-regression-001-fig-1.png){: .mx-auto.d-block :}

### Start Below

This post is the second in a series on the multiple logistic regression model.  In this post, I'll work through an example using the well known Titanic dataset, scikit-learn and statsmodels.  I'll discuss one-hot encoding, create a 3D logistic regression plot using Plotly, and demonstrate multiclass logistic regression with scikit-learn.

### Titanic Dataset

The Titanic dataset is available from many sources.  In this example, I'll be using the training dataset only available from Kaggle.  You can download the data manually, or use Kaggle's command line interface.  After reading in the data as `titanic`, let's take a quick peek at it.

```python
>>> titanic.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
 #   Column       Non-Null Count  Dtype  
---  ------       --------------  -----  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 2   Pclass       891 non-null    int64  
 3   Name         891 non-null    object 
 4   Sex          891 non-null    object 
 5   Age          714 non-null    float64
 6   SibSp        891 non-null    int64  
 7   Parch        891 non-null    int64  
 8   Ticket       891 non-null    object 
 9   Fare         891 non-null    float64
 10  Cabin        204 non-null    object 
 11  Embarked     889 non-null    object 
dtypes: float64(2), int64(5), object(5)
memory usage: 83.7+ KB

>>> titanic.head()
   PassengerId  Survived  Pclass  ...     Fare Cabin  Embarked
0            1         0       3  ...   7.2500   NaN         S
1            2         1       1  ...  71.2833   C85         C
2            3         1       3  ...   7.9250   NaN         S
3            4         1       1  ...  53.1000  C123         S
4            5         0       3  ...   8.0500   NaN         S
[5 rows x 12 columns]
```

Personally, I'm a big fan of `snake_case` column names when working with data, so let's do some quick clean up.  I decided to explore the janitor library for this, and made use of the `clean_names()` function.  And since we're only interested in a subset of fields here, let's go ahead and select just those fields to keep via pandas `loc` method.

```python
import janitor
import pandas as pd

titanic = (janitor
          .clean_names(titanic)
          .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
          .rename(columns={"pclass": "ticket_class"}))
```

Now that our data is cleaned up, let's take a quick look around to see if we have any missing values.

```python
>>> titanic.isna().sum()
sex               0
age             177
fare              0
ticket_class      0
survived          0
dtype: int64
```

We can see that there are 177 missing `age` values.  In a future post, I'll explore methods of imputing missing values, and when and why imputing is appropriate.  But for this working example, let's just remove these observations from our dataset.

```python
titanic = titanic.query("age.notna()").reset_index()
```

Lastly, since at first we'll be attempted to classify whether a passenger survived or not, let's look at the frequency of `survived`.

```python
titanic["survived"].value_counts()
0    424
1    290
```

Out of the 714 passengers in our current dataset, only 290 survived, or about 41%.  In many classification problems, we might be interested in equaling out these binary classes to produce a better predictive model.  A variety of upsampling and downsampling techniques exist for this, which I'll explore in future posts.  For this example however, we'll just take the class frequencies as it, but keep in mind that better results may be possible with more robust methods.

### Simple Logistic Regression

To keep it simple at first, let's start out with a logistic regression model with only a single predictor, `age`.  

```python
X = titanic[["age"]]
y = titanic["survived"]
```

#### scikit-learn

Let's use scikit-learn's `LogisticRegression()` first to train our model.  Note, by default, `LogisticRegresson()` preforms variable regularization by default.  We can disable this by passing `penalty="none"` when we initialize the classifier. 

```python
from sklearn.linear_model import LogisticRegression

# Discuss why penalty here is none
# Scoring method here is misclassification error rate
log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)
```
The `score` method here provides the misclassification error rate.  For this particular model, the error rate is 0.594.

We can verify the misclassification error rate is being provided by manually calculating it.
```python
y_hat = log_reg.predict(y.to_numpy().reshape(-1, 1))
misclass_error_rate = np.sum(y == y_hat) / len(y)

sklearn_score = log_reg.score(X, y)

>>> np.equal(sklearn_score, misclass_error_rate)
True
```

With a little more investigation into the results our model, we can see that this model doesn't have very much predictive potential.  Let's look at the unique predicted values of our model

```python
import numpy as np

np.unique(y_hat)
```

This yields a single unique predicted value, 0.  This indicates our model is predicting that every passenger will not survive.  By plotting our model, we can learn some more information.

```python
import seaborn as sns

sns.regplot(x="age", y="survived", data=titanic, logistic=True)
```

![2021-01-27-logistic-regression-002-fig-1.png](/assets/img/2021-01-27-logistic-regression-002-fig-1.png){: .mx-auto.d-block :}

From the plot, we don't see the characteristic $S$-shaped sigmoid function of the logistic regression model.  Instead, we do see a sigmoid function, but it more-or-less appears equivalent to a linear function for the particular domain of our data.  Clearly, `age` alone is not explained our `survived` response variable well.

Before adding new predictors to our model, let's perform the same regression as above using the statsmodels library.

#### statsmodels' `Logit()`

statsmodels provides two functions for performing logistic regression.  The first is the `Logit()` function provided in the `discrete_model` module.  The second is the `GLM()` function provided.  We'll explore the `Logit()` function first.


```python
import statsmodels.api as sm

from statsmodels.discrete.discrete_model import Logit

model = Logit(endog=y, exog=sm.add_constant(X))
result = model.fit()
```

Remember, in statsmodels, an intercept term is not included by default.  So the make our results match those of scikit-learn's, we'll have to add one via `sm.add_constant()`.

As usual with statsmodels, a wealth of summary information is provided by the `summary() method`.

```python
result.summary()
<class 'statsmodels.iolib.summary.Summary'>
"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:               survived   No. Observations:                  714
Model:                          Logit   Df Residuals:                      712
Method:                           MLE   Df Model:                            1
Date:                Sat, 30 Jan 2021   Pseudo R-squ.:                0.004445
Time:                        10:12:45   Log-Likelihood:                -480.11
converged:                       True   LL-Null:                       -482.26
Covariance Type:            nonrobust   LLR p-value:                   0.03839
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0567      0.174     -0.327      0.744      -0.397       0.283
age           -0.0110      0.005     -2.057      0.040      -0.021      -0.001
==============================================================================
"""
```

Worth mentioning here is the pseudo $R^2$ summary statistic, labeled `Pseudo R-squ`.  The pseudo $R^2$ statistic is a measure of how well a logistic regression model explains the response variable, akin to a linear regression model's $R^2$ statistic.  A variety of pseudo $R^2$ statistics have been proposed, and statsmodels reports McFadden's pseudo $R^2$, published in 1974.  While McFadden's pseudo $R^2$ statistic can take values between 0 and 1, a value in the range of 0.2 to 0.4 is concerned an excellent fit.  Not surprisingly, our value of 0.004445 is quite low, confirming our model does not fit the data well.

One useful feature about the statsmodels implementation is that probabilities, not just classes, are reported.  We can get the probability that an observations `survived` but using the predict method.

```python
>>> probs = model.predict(params=result.params)

# First 10 probabilities
>>> probs[:10]
array([0.42606613, 0.38382723, 0.41537876, 0.39163506, 0.39163506, 0.34327126,
       0.48034749, 0.4127189 , 0.44763968, 0.47487681])
```

#### statsmodels' `GLM()`

statsmodels also provides the `GLM()` function, which when the `family` parameter is set to `Binomial()`, produces the same results as the above two methods.

```python
from statsmodels.genmod.families.family import Binomial

model = sm.GLM(endog=y, exog=sm.add_constant(X), family=Binomial())
result = model.fit()
result.summary()
```

### Multiple Logistic Regression


