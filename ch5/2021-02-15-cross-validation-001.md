---
layout: post
title: "Cross-Validation #1"
subtitle: Validation Sets, Leave-One-Out Cross-Validation, and *k*-Fold Cross-Validation
comments: false
---

### Big Header

### Small Header

The structure of this post was influenced by the fifth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

| ![2021-01-08-multiple-linear-regression-001-fig-1.png](/assets/img/2021-01-08-multiple-linear-regression-001-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

### Start Below

*Resampling methods* are a crucial tool used commonly in modern statistics and data science.  These methods involve taking repeated samples from a training dataset and refitting a model of interest on each individual sample to obtain additional information about the fitted model.  These methods allow us to learn new information about the model that might not be available from a single fit of the model.

Two of the most common resampling methods are *cross-validation* and the *bootstrap*.  In this post, I'll provide an introduction to cross-validation, and will touch on the boostrap in a future post.

Cross-validation is often used to estimate the *test error* associated with a certain fitted model in order to evaluate its performance, or to decide on an optimal level of flexibility for the model.  Evaluating a model's performance is often known as *model assessment*, while selecting the optimal level of flexibility is often known as *model selection*.

The structure of this post was influenced by the fifth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### The Validation Set Approach

To evaluate a model's performance or select optimal flexibility, we are often interested in how the model performs on a test dataset.  That is, a dataset that was not used to train the model, and is thus *new* data the model has not seen before.  In particular, we are often interested in selecting a model with low test error.  

Of course, we are interested in test error and not *training error* because training error can be deceiving and often underestimates test error.  Because the observations used to train the model are also used to calculate the training error, models with more flexibility that can better fit the nuances of the training data will tend to have lower training error.  We refer to this as *overfitting*, and overfit models will tend to perform poorly on new test data. 

A simple method to estimate test error is the *validation set approach*.  This approach involves randomly dividing the set of available observations into a training set and a *validation set* or *hold-out set*.  The model of interest is trained on the training set, and then is used to make predictions on the observations in the validation set. 

The resulting validation set error rate is typically assessed using mean squared error (MSE) in the case of a quantitative response or misclassification rate in the case of a qualitative response.  This validation set error rate provides an estimation for the test error rate.

A common training set/validation set split of 70%/30% or 80%/20% is commonly used, depending on the overall number of available observations.  

The validation set approach is conceptually quite simple and easy to implement, and is particularly useful for demonstration purposes, but has two potential drawbacks:

1. Due to the randomness of creating the training and validation sets, the validation set estimate of the test error is often highly variable.  A group of outliers, for example, may end up in the training set, the validation set, or split among both sets due to chance.

2. Since a subset of observations is intentionally left out of model training, the model is trained on less data.  Because models tend to perform better when more data is available, the resulting validation set error rate will tend to overestimate the test error rate.  If the same model were fit on all available observations, it would likely provide a lower test error rate in practice.

### Leave-One-Out Cross-Validation