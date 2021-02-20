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

* Explain what cross validation is
* In a previous post, I introduced much of the theory behind CV
* In this post, I'll explore some working example using CV from scikit-learn
* In particular, I'll explore simple validation sets using train_test_split, `KFold` CV, `LeaveOneOut`, `StratifiedKFold` CV,
* GroupKFold         # will just discuss this, won't actually do it
* TimeSeriesSplit    # will just discuss this, won't actually do it
