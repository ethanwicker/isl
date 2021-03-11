---
layout: post
title: "Principal Components Analysis #1"
subtitle: (Update this)
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

*Principal components analysis* (PCA) is a technique that computes the *principal components* of a dataset and then subsequently uses these components in understanding the data.  PCA is an unsupervised approach.  In a future post, I'll explore *principal components regression*, a related supervised technique that makes uses of the principal components when fitting a linear regression model.  PCA is commonly used during exploratory data analysis to understand and project multidimensional data into a lower dimensional space, and as a preprocessing step for classification and clustering techniques.

In the below post, I'll provide an overview of PCA that includes an explanation of principal components, as well as a geometric interpretation of the principal components.

The structure of this post was influenced by the tenth chapter of *An Introduction to Statistical Learning: with Applications in R* by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani.

### Principal Components

When attempting to understand $n$ observations measured over $p$ features $X_1, X_2, \ldots, X_p$, $p$ is often quite large.  When this is the case, it is not feasible to plot our data entirely via two or three-dimensional scatter plots and accurately understand the larger dataset.  By only viewing two or three features at a time, we simply lose to much information, especially when $p$ is greater than 10 or 20.  

To mitigate this, we can find a low-dimensional representation of our data that captures as much of the information as possible.  For example, if we are able to find a two or three-dimensional representation of a dataset that contains 20 predictor features - and this low-dimensional representation accurately represents these features - we can then project our data into this low-dimensional space to better visualize and understand it. 

PCA allows us to do just this by finding a low-dimensional representation of a dataset that contains as much of the variation as possible.  The idea behind PCA is that each of the $n$ observations lives in $p$-dimensional space, but not all of these dimensions are equally interesting.  PCA attempts to find a small number of dimensions that are as interesting as possible, where *interesting* is measured by the amount that the observations vary along each dimension.  These small number of dimensions that contain as much variation as possible are referred to as *principal components*.  Each of the principal components found by PCA is a linear combination of the $p$ features.

The first principal component of a set of features $X_1, X_2, \ldots, X_p$ is the normalized linear combination of the features

$$
\begin{aligned} 
Z_1 = \phi_{11}X_1 + \phi_{21}X_2 + \ldots + \phi_{p1}X_0
\end{aligned}
$$

that has the largest variance.  Here, *normalized* means that $\sum_{j=1}^{p} \phi_{j1}^2 = 1$.  The elements $\phi_{11}, \phi_{21}, \ldots, \phi{p1}$ are referred to as the *loadings* of the first principal component.  Together, the loadings make up the principal component loading vector $\phi_1 = (\phi_{11} \phi_{21} \ldots \phi_{p1})^T$.  Note, if we did not constrain the loadings such that their sum of squares must equal one, these elements could be arbitrarily large in absolute value and would result in arbitrarily large variance.

Specifically, the first principal component loading vector solves the optimization problem

$$
\begin{aligned} 
\stackrel{\text{maximize}}{{\phi{11}, \ldots, \phi{p1}}} \bigg\{ \big\ \frac{1}{n}\sum_{i=1}^{n}{\bigg( \sum_{j=1}^{p}{\phi_{j1}x_ij} \bigg)^2}\bigg\}}} \text{ subject to } \sum_{j=1}^{p}{\phi_{j1}^2 = 1}
\end{aligned}
$$

which is solvable via eigen decomposition.  The above objective function is equivalent to maximizing the sample variance of the $n$ values $z_{11}, z_{21}, \ldots, z_{n1}.  We were to the values $z_{11}, z_{21}, \ldots, z_{n1 as the *scores* of the first principal component.

There is a nice geometric interpretation of the first principal component as well.  The loading vector $\pi_1$ with elements $\phi_{11}, \phi_{21}, \ldots, \phi{p1}$ defines a direction in feature space along which the data *vary the most*.  Projecting the $n$ data points $x_1, \ldots, \x_n$ onto this direction gives us the principal component scores $z_{11}, \ldots, z_{n1}$ themselves.

INSERT IMAGE HERE.  Maybe Figure 6.14 or maybe make a plot myself.

After calculating the first principal component $Z_1$ of the features, we can find the second principal component $Z_2$.  The second principal component is the linear combination of $X_1, \ldots, \X_p$ that has maximal variance out of all linear combinations that are *uncorrelated* with $Z_1$.  Constraining $Z_2$ to be uncorrelated with $Z_1$ is equivalent to constraining the second principal component loading vector $\phi_2$ with elements $\phi_{12}, \phi_{22}, \ldots, \phi{p2}$ to be orthogonal with the direction $\phi_1$.  We calculate additional principal components in the same way as above, with the constraint that they are uncorrelated with earlier principal components.

After computing the principal components, we can plot them against each other to produce low-dimensional views of the data.  Geometrically, this is equivalent to projecting the original data down onto the subspace spanned by $\phi_1$, $\phi_2$, and $\phi_3$ (in the event where we calculated the first three principal components), and plotting the projected points. 

INSERT BIPLOT IMAGE HERE

### Another Interpretation of Principal Components

Earlier, I described the principal component loading vectors as the direction in feature space which the data vary the most, and the principal component scores as projections along these directions.  However, it is also accurate to interpret the principal components as providing low-dimensional linear surfaces that are *closest* to the observations.

To expand on this, the first principal component loading vector is the line in $p$-dimensional space that is *closest* to the $n$ observations, where closeness is measured via average squared Euclidean distance.  Thus, the first principal component satisfies the condition that it represents a single dimension of that data that lines as close as possible to all of the data points.  Therefore, this line will likly provide a good summary of the data.

This concept extents beyond just the first principal component.  For example, the first two principal components of a dataset span the plan that is closest to the $n$ observations.  The first three principal components of a data set span a three-dimensional hyperplane that is closests to the $n$ observations, and so forth.

### Scaling the Variables











