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

### Start Here

The *bootstrap* is a widely used resampling technique first introduced by Bradley Efron in 1979 commonly used to quantify the uncertainty associated with a given estimator or statistical learning method.  The bootstrap can be applied to many problems and methods, and is commonly used to estimate the standard errors of the coefficients estimated from regression model fits, or the distribution of $R^2$ values from those fits.

Via bootstrap aggregation, we can estimate the uncertainty - or variability - associated with a given method by taking repeated samples from a dataset with replacement and applying the method.  For example, to estimate the uncertainly of a coefficient estimate $\hat{\beta_1}$ from a linear regression fit, we take $n$ repeated samples with replacement from our dataset and train our linear regression model $n$ times and record each value $\hat{\beta}_1^{*i}$.  With enough sampling, the distribution of all estimates $\hat{\beta}_1^{*i}$ will approach the Gaussian distribution, and thus we can quantify the variability of this estimate by calculating standard errors and confidence intervals.

The power of the bootstrap lies in the ability to take repeated samples of the dataset, instead of collecting a new dataset each time.  Also, in contrast to standard error estimates typically reported with statistical software that rely on algebraic methods and underlying assumptions, bootstrapped standard error estimates are more accurate as they are calculated computationally.  For example, the common standard error estimate for a linear regression fit is dependent upon an unknown parameter $\sigma_2$ that is estimated using the residual sum of squares values.  Bootstrapped standard error estimates do not rely on these assumptions and unknown parameters, so it likely to produce more accurate results.

### Calculating Bootstrapped Estimates using scikit-learn's `resample`

START HERE TOMORROW