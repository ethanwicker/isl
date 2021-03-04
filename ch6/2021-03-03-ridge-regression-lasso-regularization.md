---
layout: post
title: "Ridge Regression and Lasso Regularization Methods"
subtitle: Change: Overview, Comparison, scikit-learn example, statsmodels comparison, grid search, maybe nested CV
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
| <sub><sup>**Sour


### Notes
- also known as Tikhonov regularization, after Andrey Nikolayevich Tikhonov, the Russian and Soviet mathematician and geophysicist
- show the comparison to statsmodels https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html
- lasso = least absolute shrinkage and selection operator

Structure
- Regularization is a method of fitting a model containing all predictors that *regularizes* the coefficient estimates towards zero.  Also known as *constraining* or *shrinking* the coeficient estimaties.  This is useful because doing so can significantly reduce the models variance, and thus improve test error results.  Two common methods are ridge regression and the lasso

### Start Here

Regularization is a method of fitting a model containing all predictors $p$ that *regularizes* the coefficient estimates towards zero.  Also known as *constraining* or *shrinking* the model's coefficient estimates, regularization can be helpful because the technique can significantly reduce the model's variance and thus improve test error estimates and model performance.  The two most commonly used regularization methods are *ridge regression* and the *lasso*.

In the below post, I'll provide an overview of both of these methods.  I'll also explore using cross-validation to tune model hyperparameters, and will provide working examples of these methods using both Python's scikit-learn and statsmodels packages.

The structure of this post was influenced by the sixth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Ridge Regression

Ridge regression can be viewed as an extension of the least squares fitting procedure.  The least squares fitting procedure estimates the coefficients $\beta_0, \beta_1, ... \beta_p$ as the values that minimize 

$$
\begin{aligned} 
RSS = \sum_{i=1}^{n} (y_i - \hat{\beta_0} - \sum_{j=1}^{p} \beta_j x_{ij})^2.
\end{aligned}
$$

In contrast, ridge regression estimates the coefficients by minimizing a similar, but different quantity.  In particular, the ridge regression coefficient estimates $\hat{\beta}^R are the values that minimize 

$$
\begin{aligned} 
RSS = \sum_{i=1}^{n} (y_i - \hat{\beta_0} - \sum_{j=1}^{p} \beta_j x_{ij})^2 + \lambda \sum_{j=1}^{p} \beta_j^2 = RSS + \lambda \sum_{j=1}^{p} \beta_j^2
\end{aligned}
$$

where $\lambda \gte 0$ is a *tuning parameter* commonly selected using cross-validation.  The second term $\lambda \sum_{j=1}^{p} \beta_j^2$ is known as a *shrinkage penalty* and has the effect of shrinking the estimates $\hat{\beta_0}, \hat{\beta_1}, ... \hat{\beta_p}$ towards zero.  When $\lambda = 0$, penalty term has no effect and ridge regression will produce the same coefficient estimates as the least squares estimates.  However, as $\lambda \to \inf$, the ridge regression coefficient estimates will approach zero. 

Outside the statistical literature, ridge regression is known as Tikhonov regularization (and less commonly Tikhonov–Phillips regularization), after the Russian and Soviet mathematician and geophysicist Andrey Nikolayevich Tikhonov.

| ![2021-03-03-ridge-regression-lasso-regularization-001-fig-1.png](/assets/img/2021-03-03-ridge-regression-lasso-regularization-001-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Sour

The figure above shows ridge regression performed on the Credit data set (available via the ISLR R package).  The left plot shows the standardized coefficients as a function of $\lambda$, while the right plot shows the same coefficients as a function of $\| \hat{\beta_{\lambda}^R \|_2 / \| \hat{}\beta \|_2}$.  Here, \| \beta \|_2 = \sqrt{\sum_{j=1}^{p} \beta_j^2} is the $l_2 norm$ and measures the distance of $\beta$ from zero.  From the image, we can see that the `Income`, `Limit`, `Rating` and `Student` variable approach zero as  $\lambda$ increases, while the other variables (dulled out in gray) maintain values near zero for any value of $\lambda$.

Note that ridge regression is influenced by the scale of the predictor variables.  As such, it is best to apply ridge regression after all predictor variables have been scaled such that they each have a mean of zero and a standard deviation of 1.  We do not have to also standardize the response variable when standardizing the predictor variables, but it does no harm.



\| x\| 


Next:
- Comment: Must standardize variables
- Discuss credit picture & L2 norm
- Continue below equation 6.5 on p.225