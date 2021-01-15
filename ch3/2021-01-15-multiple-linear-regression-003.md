---
layout: post
title: "Multiple Linear Regression #3"
subtitle: Qualitative Predictors, Interaction Terms, and Non-linear Relationships
comments: false
---

**Notes below**

### Big header

#### Small header

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

**Notes above**

This post is the third in a series on the multiple linear regression model.  In previous posts, I introduced the multiple linear regression model and explored a comparison of Python's scikit-learn and statsmodels libraries.  However, both of these previous posts exclusively explored quantitative predictors.  

In this post, I'll explore qualitative predictors.  I'll also explore classical methods of relaxing some restrictive assumptions of the linear model - namely the *additive* and *linear* assumptions - via the use of *interaction terms* and *polynomial regression*.

This structure of this post was influenced by the third chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

#### Qualitative Predictors

Depending on the context, qualitative predictors are sometimes referred to as categorical or factors variables.  The most common methods of implementing these predictors in a linear model is via *indicator* or *dummy variables*.  Below, I'll introduce this topic in the context of qualitative predictors with only two levels, and then extend the concept to qualitative predictors with multiple levels.

##### Qualitative Predictors Predicting with Only Two Levels

##### Qualitative Predictors with More than Two Levels
