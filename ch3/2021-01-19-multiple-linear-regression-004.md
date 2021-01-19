---
layout: post
title: "Multiple Linear Regression #4"
subtitle: Update this
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

This post is the fourth in a series on the multiple linear regression model.  In previous posts, I explored the topic, including methods of relaxing various assumptions made by the model.  I also performed a comparison of Python's scikit-learn and statsmodels libraries for multiple linear regression.

In this post, I'll briefly discuss potential problems that can occur when fitting a multiple regression model to a particular dataset, including:

1. *Non-linearity of the response-predictor relationships.* 
2. *Correlation of error terms.*
3. *Non-constant variance of error terms.*
4. *Outliers.*
5. *High-leverage points.*
6. *Collinearity.*

The structure of this post was influenced by the third chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

##### Non-linearity of the Response-Predictor Relationships

A key assumption made by the linear regression model is linearity between the predictors and response.  If this assumption is far from true for a particular dataset, then we should be skeptical of nearly all conclusions drawn from the model fit.

One method of identifying non-linearity is the *residual plot*.  In the multiple regression context, we plot the residuals $e_i = y_i - \hat{y_i}$ versus the fitted values $\hat{y_i}$.  If the true relationship is approximately linear, we should see little to no discernible pattern in the residual plot.  If, however, a pattern in the residual plot is present, this is a strong indication that non-linearity does exist.

| ![2021-01-19-multiple-linear-regression-003-fig-1.png](/assets/img/2021-01-19-multiple-linear-regression-003-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

Above are two residual plots.  The left panel shows the resulting residual plot after a linear model fit, while the right panel shows the resulting residual plot are a quadratic model fit.  In the left panel, there is a clear U-shape, indicating a strong non-linearity in the data.  In contrast, in the right panel a much smaller pattern is present, indicating a quadratic term in the model improves the fit.

If the residual plot indicates a non-linearity relationship in the data, a simple solution is to include a non-linear transformation of the predictors in the regression model.  In future posts, I'll explore more advanced non-linear approaches for addressing this issue. 

##### Correlation of Error Terms



##### Non-constant Variance of Error Terms

##### Outliers

##### High Leverage Points

##### Collinearity