---
layout: post
title: "Principal Components Regression"
subtitle: Update
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

*Principal components analysis* (PCA) is a common and popular technique for deriving a low-dimensional set of features from a large set of variables.  For more information on PCA, please refer to my earlier post on the [technique](https://ethanwicker.com/2021-03-11-principal-components-analysis-001/).  In this post, I'll explore using PCA as a dimension reduction technique for regression, known as *principal components regression*.

The structure of this post was influenced by the sixth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Principal Components Regression

Principal components regression (PCR) involves first performing principal components analysis on our data set to obtain the first $M$ principal components $Z_1, \ldots, Z_M$.  These components are then used as the predictors in a linear regression model fit using least squares.

The underlying idea behind PCR is that often a small number of principal components can sufficiently explain most of the variability in the data, as well as the predictor's relationship with the response.  Thus, we assume that *the directions in which $X_1, \ldots, X_p$ show the most variation are the directions that are associated with $Y$*.  This assumption is often not guaranteed, but does turn out to be a reasonable enough approximation and provide good results.  If the assumption does hold, then fitting a least squares model to $Z_1, \ldots, Z_M$ will lead to better results that fitting a least squares model to $X_1, \ldots, X_p$, and we will also be able to mitigate overfitting.  As with PCA, it is recommended to standardize each predictor before performing PCR.

COMMENT/Question: PCA is performed on the components.  Particularly the scores, just like we do Y ~ X1 + X2 where X1 contains x1,x2,... we do PCR on the Y ~ Z1 + Z2 where Z1 contains z1, z2, ... 

---->>>>RECONSTRUCT FIGURE 6.18 maybe?  Test MSE vs. Number of Components for PCR on a dataset with lots of predictor variables.  Calculate Test MSE via 10-fold cross validation.     <<<<----

In the above plot, we see the results of PCR applied to the **INSERT DATA SET**.  The resulting test mean squared error (MSE) from each fit is plotted against the number of principal components used in that fit.  From the plot, we see a typical U-shaped curve for the mean squared error.  As more principal components are used in the regression model, the bias decreases, but the variance increases, causing this U-shaped curve.  Of course, when the number of components $M$ used is equivalent to the number of predictor variables $p$, then PCR is simply the least squares fit using all of the original predictors.  The above plot indicates that PCR performed with an appropriate choice of $M$ can lead to a significant improvement over least squares.  This is especially true when much of the variability and association with the response are contained in a small number of the predictors.  In contrast, PCR will tend to not perform as well when many principal components are needed to adequately model the response.  In some situations, PCR may outperform shrinkage methods, such as ridge regression and the lasso, and in our situations it may not.  For any given use case, model performance evaluation is needed to determine the best performing model.

Note, PCR is *not* a feature selection method.  This is because each of the $M$ principal components used in the regression model is a linear combination of all of the original predictors $p$.  Thus, PCR is more closely related to ridge regression than the lasso.

### Choosing the Number of Principal Components

In contrast to principal components analysis, which is inherently an unsupervised approach, PCR is a supervised approached.  In PCR, the number of principal components $M$ is typically chosen via cross-validation.  A method such as $k$-fold cross-validation or leave-one-out cross-validation should be used.
