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

### Start Here

Partial least squares (PLS) is an alternative method to principal components regression (PCR).  Similar to PCR, PLS is a dimension reduction method that identifies a new set of features $Z_1, 
ldots, Z_M$ that are linear combinations of the original features $X_1, \ldots, \X_p$, and then fits a linear model via least squares using these $M$ new features.  However, unlike PCR, PLS identifies these new features in a supervised way by making use of the response $Y$ in order to identify these new features.  These $M$ new features identified are identified as not only approximating the original features well, but also are related to the response.  PLS attempts to find *directions* that help explain both the response and the predictors.

In contrast, PCR attempts to approximate the original features well by finding principal components that explain the most variation in the data.   

The structure of this post was influenced by the sixth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Partial Least Squares


* standardarize the predictors
* fit Z1 = phi1X1 + phi2X2 where phi1, phi1 are equal to beta1, beta2 from simple linear regression
* then regress X1 ~ Z1 and get the residuals. regress X2 ~ Z2 and take residuals.  then calculate Z2 = phi1 * residuals_from_x1_on_z1 + phi2 * residuals_from_x2_on_z2
* the repeat
* regress X1 ~ Z1 + Z2 and get residuals, X2 ~ Z1 + Z2 and ge residuals, then Z3 = phi1 * residuals_from_here + phi2* residuals_from_here




Let Z1 , Z2 , . . . , ZM represent M < p linear combinations of our original p predictors. That is,
θmzim +εi, i=1,...,n, (6.17)
using least squares. Note that in (6.17), the regression coefficients are given byθ0,θ1,...,θM.Iftheconstantsφ1m,φ2m,...,φpm arechosenwisely,then such dimension reduction approaches can often outperform least squares regression. In other words, fitting (6.17) using least squares can lead to better results than fitting (6.1) using least squares.
The term dimension reduction comes from the fact that this approach reduces the problem of estimating the p+1 coefficients β0,β1,...,βp to the simpler problem of estimating the M + 1 coefficients θ0, θ1, . . . , θM , where M < p. In other words, the dimension of the problem has been reduced from p+1 to M +1.
Notice that from (6.16),
􏰉M 􏰉M􏰉p 􏰉p􏰉M 􏰉p
βˆL 1/βˆ1
0.4 0.6 0.8 1.0
                linear regression model
yi =θ0 +
􏰉M m=1
m=1
m=1
j=1
j=1 m=1
θmzim =
θm
φjmxij =
θmφjmxij = βjxij, j=1
Zm =
􏰉p j=1
dimension reduction linear combination
φjmXj (6.16) for some constants φ1m,φ2m ...,φpm, m = 1,...,M



