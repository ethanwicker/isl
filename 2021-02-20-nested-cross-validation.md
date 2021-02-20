---
layout: post
title: "Multiple Linear Regression #1"
subtitle: A Brief Introduction
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

Miscellaneous notes below:

cross validation useful to evaluate the performance of a model on a dataset --> how well is my model doing
nested cross validation used to turn parameters of an algorithm and evaludate the perform of the model - let's pick the value of C and evaluate how well the model is doing

Outer loop: CV for model evaluation
Inner Loop: CV for parameter turning
Nested CV as a whole evaluates how well an algorithm performs including parameter tuning
the final model we pick the best parameters & train the model on all data
"Set aside a test fold"
"Reserve a validation set from the training set"
Great video: https://www.youtube.com/watch?v=az60jS7MQhU by Dr. Cynthia Rudin at Duke University (https://en.wikipedia.org/wiki/Cynthia_Rudin)

For nested cross validation, we would perform a GridSearchCV or RandomSearchCV within k-fold cross validation (pretty sure, similar to this post: https://towardsdatascience.com/nested-cross-validation-hyperparameter-optimization-and-model-selection-5885d84acda)
