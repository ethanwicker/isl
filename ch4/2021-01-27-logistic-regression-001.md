---
layout: post
title: "Logistic Regression #1"
subtitle: A Brief Introduction
comments: false
---

### Big Header

### Small Header

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

| ![2021-01-08-multiple-linear-regression-001-fig-1.png](/assets/img/2021-01-08-multiple-linear-regression-001-fig-1.png){: .mx-auto.d-block :} |
| :--: |
| <sub><sup>**Source:** *Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani. An Introduction to Statistical Learning: with Applications in R. New York: Springer, 2013.* |

## Start Below

This is the first post in a series on the logistic regression model.  The structure of this post was influenced by the fourth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Logistic Regression

In contrast to linear regression, which attempts to model the response $Y$ directly, logistic regression attempts to models the probability that $Y$ belongs to a particular category, or class.

Although we can use a linear regression model to represent probabilities $p(x)$, as in 

$$
\begin{aligned} 
p(x) = \beta_0 + \beta_1X_1,
\end{aligned}
$$

this approach is problematic.  In particular, for some values of $X_1$, this method will predict probability values below 0 and above 1.

To avoid this problem, we should instead model the probability that $Y$ belongs to a given class using a function that provides bounded output between 0 and 1.  Many function could be used for this, including the *logistic function* used in logistic regression:

$$
\begin{aligned} 
p(x) = \frac{e^{\beta_0 + \beta_1X_1}{1 + e^{\beta_0 + \beta_1X_1}}.
\end{aligned}
$$

This model is fit via the method of *maximum likelihood*, to be discussed below.  It will always produce an $S$ shaped function, bounded between 0 and 1.

We can further manipulate the function such that 

$$
\begin{aligned} 
\frac{p(x)}{1-p(x)} = e^{\beta_0 + \beta_1X_1}.
\end{aligned}
$$

The left-hand side of the above equation, $p(x) - (1 - p(x))$, is known as the *odds* and can take any value between 0 and $\inf$.  Odds values close to 0 indicate very low probabilities, and odds values close to $\inf$ indicate very high probabilities.

Finally, by taking the logarithm of both sides of the above equation we have

$$
\begin{aligned} 
log(\frac{p(x)}{1-p(x)}) = \beta_0 + \beta_1X_1.
\end{aligned}
$$

Here, the left-hand side is referred to as the *log-odds* or *logit*.  The logistic regression model has a logit that is linear with respect to $X$.  Thus, increasing $X$ by one unit changes the log odds by $B_1$, or equivalently, multiples the odds by $e^{B_1}$. 

#### Estimating the Regression Coefficients