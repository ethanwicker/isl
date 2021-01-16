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

Depending on the context, qualitative predictors are sometimes referred to as categorical or factors variables.  The most common methods of implementing these predictors in a linear model is via *indicator* or *dummy variables*.  Below, I'll introduce this topic in the context of qualitative predictors with only two levels, where a *level* refers to a unique value the qualitative variable can take.  I'll then extend the concept to qualitative predictors with multiple levels.

##### Qualitative Predictors with Two Levels

In the case where we have a predictor variable with two possible levels, it is straightforward to create a dummy variable that takes on two possible numerical values, 0 and 1:

$$
\begin{aligned} 
x_i = 
    \begin{cases}
        1 & \text{if $ith$ observation is the first level}\\
        0 & \text{if $ith$ observation is the second level}
    \end{cases} 
\end{aligned}
$$.

We then use this dummy variable in the standard multiple linear regression model:

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_i + \epsilon_i = 
    \begin{cases}
        \beta_0 + \beta_1X_1 + \beta_2 + \epsilon_i & \text{if $ith$ observation is the first level}\\
        \beta_0 + \beta_1X_1 + \epsilon_i & \text{if $ith$ observation is the second level}
    \end{cases}
\end{aligned}
$$

(Start here, with 3rd bullet in my notes)

##### Qualitative Predictors with More than Two Levels
