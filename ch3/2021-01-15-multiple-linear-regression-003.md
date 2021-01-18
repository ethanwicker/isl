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

In the case where we have a predictor variable with two possible levels, it is straightforward to create a dummy variable that takes on two possible numerical values, $0$ and $1$:

$$
\begin{aligned} 
x_i = 
    \begin{cases}
        1 & \text{if $ith$ observation is the first level}\\
        0 & \text{if $ith$ observation is the second level}
    \end{cases} 
\end{aligned}
$$.

We then use this dummy variable in the linear regression model:
(Note: Make sure the below is correct)

$$
\begin{aligned} 
Y_i = \beta_0 + \beta_1X_i + \epsilon_i = 
    \begin{cases}
        \beta_0 + \beta_1 + \epsilon_i & \text{if $ith$ observation is the first level}\\
        \beta_0 + \epsilon_i & \text{if $ith$ observation is the second level}
    \end{cases}
\end{aligned}
$$.

Alternatively, instead of encoding our dummy variable as $0$ or $1$, we could instead have encoded it as $-1$ or $1$.  Doing so would have changed the coefficient estimates of the model, and the interpretation, but not the predictors.  In addition, when performing statistical inference, we interpret the corresponding p-values of qualitative predictors exactly as we do quantitative predictors.

##### Qualitative Predictors with More than Two Levels

In the case where a qualitative predictor has more than two levels, we cannot use a single dummy variable to represent all possible levels.  In this situation, we simply create additional dummy variables.  For example, if we have a predictor variables with three possible levels, we can encode these levels into two dummy variables

$$
\begin{aligned} 
x_{i1} = 
    \begin{cases}
        1 & \text{if $ith$ observation is the first level}\\
        0 & \text{if $ith$ observation is not the first level}
    \end{cases} 
\end{aligned}
$$

and 

$$
\begin{aligned} 
x_{i2} = 
    \begin{cases}
        1 & \text{if $ith$ observation is the second level}\\
        0 & \text{if $ith$ observation is not the second level}
    \end{cases} 
\end{aligned}
$$.

We then use these dummy variables in the regression equation to obtain the model

$$
\begin{aligned} 
Y_i = \beta_0 + \beta_1X_{i1} + \beta_2X_{i2} + \epsilon_i = 
    \begin{cases}
        \beta_0 + \beta_1 + \epsilon_i & \text{if $ith$ observation is the first level}\\
        \beta_0 + \beta_2 + \epsilon_i & \text{if $ith$ observation is the second level}
        \beta_0 + \epsilon_i & \text{if $ith$ observation is the third level}
    \end{cases}
\end{aligned}
$$.

There will always be one fewer dummy variables than the number of levels of the predictors.  The level with no dummy variable - the third level in the above example - is referred to as the *baseline*.  Of note, depending on the context, dummy variable encoding may also be referred to as *one-hot encoding*.  These techniques are equivalent, but one-hot encoding tends to keep the *baseline* level as an encoded variable, as opposed to dropping it.  This is sometimes preferred in machine learning models using *regularization*, which will be discussed in a future post.

With the dummy variable approach, we can incorporate both quantitative and qualitative predictors into the multiple regression model.  Graphically, doing so results in parallel hyperplanes in the predictor space.  As mentioned above, the interpretation of the p-values does not change, but the p-values themselves do depend on the choice of dummy variable encoding.  As mentioned in earlier posts, we can still use the F-test to test $H_0: \beta_1 = \beta_2 = \ldots = \beta_p = 0\$ to determine if any relationship exists between the predictors and the response.

(Maybe above: \{0,1,2,\,\ldots\})

#### Extensions of the Linear Model
