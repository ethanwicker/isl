---
layout: post
title: "Multiple Linear Regression #1"
subtitle: A Brief Introduction
comments: false
---

### Big Header

#### Small Header

The structure of this post was influenced by the third chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

| ![2021-01-08-multiple-linear-regression-001-fig-1.png](/assets/img/2021-01-08-multiple-linear-regression-001-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

### Misc

Importantly, any data preparation prior to fitting the model or tuning of the hyperparameter of the model must occur within the for-loop on the data sample. This is to avoid data leakage where knowledge of the test dataset is used to improve the model. This, in turn, can result in an optimistic estimate of the model skill.

A useful feature of the bootstrap method is that the resulting sample of estimations often forms a Gaussian distribution. In additional to summarizing this distribution with a central tendency, measures of variance can be given, such as standard deviation and standard error. Further, a confidence interval can be calculated and used to bound the presented estimate. This is useful when presenting the estimated skill of a machine learning model.

Sample Size
In machine learning, it is common to use a sample size that is the same as the original dataset.
If the dataset is enormous and computational efficiency is an issue, smaller samples can be used, such as 50% or 80% of the size of the dataset.

Notes on standard deviation vs. standard error
KEY TAKEAWAYS
The T distribution is a continuous probability distribution of the z-score when the estimated standard deviation is used in the denominator rather than the true standard deviation.
The T distribution, like the normal distribution, is bell-shaped and symmetric, but it has heavier tails, which means it tends to produce values that fall far from its mean.
T-tests are used in statistics to estimate significance.
I really want the standard deviation here because I already have my distribution
The standard deviation (often SD) is a measure of variability. When we calculate the standard deviation of a sample, we are using it as an estimate of the variability of the population from which the sample was drawn. For data with a normal distribution,2 about 95% of individuals will have values within 2 standard deviations of the mean, the other 5% being equally scattered above and below these limits.
When we calculate the sample mean we are usually interested not in the mean of this particular sample, but in the mean for individuals of this type—in statistical terms, of the population from which the sample comes. We usually collect data in order to generalise from them and so use the sample mean as an estimate of the mean for the whole population. Now the sample mean will vary from sample to sample; the way this variation occurs is described by the “sampling distribution” of the mean.
So, if we want to say how widely scattered some measurements are, we use the standard deviation. If we want to indicate the uncertainty around the estimate of the mean measurement, we quote the standard error of the mean.
Because I already have my distribution, I'm interested in the standard deviation here
If instead, I was interested in how the mean of my estimates would vary across many bootstrap resampling procedures, I would want the standard error


To do the above to look at the distribution of a coef, we would just need to capture that coef value each time in a list and plot it
and then get CIs and such

This might also be useful
```python
# I think the below does want I want it to?
# Not sure, need to verify
# This is bagging aggregation
# Include just to show how the different models all look
# With bootstrapping we are estimating the CIs of (in this case) beta_0 and beta_1
# Actual bagging will come in a later post, but show the last line of this just to demonstrating boostrap aggregation for regression prediction

# If you want to use scikit's API for the bootstrap part of the code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

# Create toy data 
x = np.linspace(0, 10, 20)
y = x + (np.random.rand(len(x)) * 10)

# Extend x data to contain another row vector of 1s
X = np.vstack([x, np.ones(len(x))]).T

n_estimators = 50
model = BaggingRegressor(LinearRegression(), 
                         n_estimators=n_estimators,
                         bootstrap=True)

model.fit(X, y)

plt.figure(figsize=(12,8))

# Accessing each base_estimator (already fitted)
for m in model.estimators_:
    plt.plot(x, m.predict(X), color='grey', alpha=0.2, zorder=1)

plt.scatter(x,y, marker='o', color='orange', zorder=4)

# "Bagging model" prediction
plt.plot(x, model.predict(X), color='red', zorder=5)
```

### Start Here