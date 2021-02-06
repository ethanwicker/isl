---
layout: post
title: "Linear Discriminant Analysis #1"
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

This post is the first in a short series on disciminant analysis.  In this series, I'll discuss linear discriminant analysis, quadratic discriminant analysis, as well as applications in Python.

The structure of this post was influenced by the fourth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Linear Discriminant Analysis

In logistic regression, we model $Pr(Y=k|X=x)$ using the logistic function.  Specifically, we model the conditional distribution of the response $Y$, given the predictors $X$.

In contrast, in disciminant analysis, we use an alternative and less direct approach to estimating these probabilities.  Instead of directly modeling $Pr(Y=k|X=x)$, we model the distribution of the predictors $X$ separately for each of the response classes of $Y$.  We then use Bayes' theorem to flip these around into estimates for $Pr(Y=k|X=x)$.  When the distributions of the predictors $X$ are assumed to be normal, linear discriminant analysis takes a form very similar to logistic regression.

If logistic regression and linear discriminant analysis end up taking such similar forms, then why do we need both?  There are several key reasons:

* In the case where the response classes of $Y$ are well-separated, the parameter estimates for the logistic regression model are surprisingly unstable.  This causes them to swing and vary, and does not produce an accurate prediction for all cases.  Linear discriminant analysis does not suffer from this problem.

* In the case where *n*, the sample size, is small and the distribution of the predictors $X$ is approximately normal in each of the response classes, the linear discriminant model is again more stable than the logistic regression model.

* In the case where we have more than two response classes, linear discriminant analysis is an attractive approach over multi-class logistic regression.

### Bayes' Theorem for Classification

Next, we'll explore using *Bayes' theorem* for classification.  Bayes' theorem will allow us to perform the "flip" discussed above to determine estimates for $Pr(Y=k|X=x)$.  I'll simultaneously introduce the theory and discuss a working example to clarify understanding.

Suppose we wish to classify an observation into one of $K$ classes, where $K \gte 2$.  Let $\pi_k$ represent the overall or *prior* probability that a randomly chosen observation belongs to the $k$th class.  Let $f_k(x) \equiv Pr(X=x|Y=k)$, where $f_k(x)$ denotes the *density function* of X for an observation that comes from the $k$th class.  Remember, the total area under a *density curve* is always equal to one, indicating that across multiple values of $x$, the area under f_k(x)$ is equal to one, for a specific class $k$.

Note, $f_k(x) \equiv Pr(X=x|Y=k)$ is technically only true when $X$ is a discrete random variable.  For the event where $X$ is continuous, $f_k(x)dx$ corresponds to the probability of $X$ falling in a small region $dx$ around $x$.

Let's explore an example.  Imagine the United States, the United Kingdom and Canada are comparing the times it takes their citizens to run an 800-meter run.  The United States provides the times for 200 of their citizens, the United Kingdom provides the times for 300 of their citizens, and Canada provides the times for 100 of their citizens.

Let's find the prior probability that a random chosen person is from the USA.  In notation, this is equivalent to $\pi_{USA}$.  Of course, this is just equal to 

$$
\begin{aligned} 
$\pi_{USA}$ = \frac{200 USA citizens}{600 total citizens}.
\end{aligned}
$$

The density function for the USA class, or $f_{USA}(x), is a function indicating the probability that a given observation $x$ ran a given 800-meter run in minutes $f(x)$, where $x$ belongs to the USA class.  We can imagine that some runners might be very fast and run times below 2 minutes, while some will be slow and run times greater than 4 minutes.  However, most will likely run times between 2 and a half minutes and 3 and a half minutes.  The probability an America runs under 2 minutes is low, but the probability an American runs between 2 and 4 minutes is quite high.

Then Bayes' theorem states that

$$
\begin{aligned} 
Pr(Y = k|X = x) = \frac{\pi_k f_k(x){\sum_{l=1}{K}\pi_l f_l(x)}.
\end{aligned}
$$


Start at with accordance...