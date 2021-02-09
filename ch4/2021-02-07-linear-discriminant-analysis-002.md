---
layout: post
title: "Linear Discriminant Analysis #2"
subtitle: scikit-learn, Precision, Recall, F1-scores, ROC Curves, and a comparison to Logistic Regression
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

### Start below

This post is the second in a series on linear discriminant analysis (LDA).  In the first post, I introduced much of the theory behind linear discriminant analysis.  In this post, I'll explore the method using scikit-learn.  I'll also discuss classification metrics such as precision and recall, and compare LDA to logistic regression.

### Titanic Dataset

For this working example, I’ll be using the same slimmed down Titanic dataset from my previous [logistic regression post](https://ethanwicker.com/2021-01-27-logistic-regression-002/).

```python
>>> titanic
        sex   age     fare  ticket_class  survived
0      male  22.0   7.2500             3         0
1    female  38.0  71.2833             1         1
2    female  26.0   7.9250             3         1
3    female  35.0  53.1000             1         1
4      male  35.0   8.0500             3         0
..      ...   ...      ...           ...       ...
709  female  39.0  29.1250             3         0
710    male  27.0  13.0000             2         0
711  female  19.0  30.0000             1         1
712    male  26.0  30.0000             1         1
713    male  32.0   7.7500             3         0
[714 rows x 5 columns]
```

### Linear Discriminant Analysis with One Predictor

To begin, let's explore linear discriminant analysis with just one predictor.  We'll try and classify whether a passenger `survived` or not based on the `fare` that passenger paid.  We might expect passengers who paid a higher fare would also be more likely to survive.

```python
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Assigning X and y
X = titanic[["fare"]]
y = titanic["survived"]

# Initializing the LDA estimator
lda = LinearDiscriminantAnalysis()

# Performing LDA
lda.fit(X, y)
```

We can view the predictors values and the correct classification rate via the `predict()` and `score()` methods.

```python
>>> lda.predict(X)
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
       ...
       ])

>>> lda.score(X, y)
0.6526610644257703
```

Our current model correctly predicts whether a passenger survives or not about 65% of the time.  Below we'll investigate if this model is performing well or not.

### Linear Discriminant Analysis with Multiple Predictors

Because linear discriminant analysis assumes that the random variable $X = (X_1, X_2, \ldots, X_p)$ is drawn from a multivariate Gaussian distribution, it does not tend to perform well with encoded categorical predictors.  Similarly, because of this assumption, LDA is also not guaranteed to find an optimal solution for non-Gaussian distributed predictor variables.  It should be noted that LDA is somewhat robust to such predictor variables - and may even perform fairly well on classification tasks - but it will just likely not find LDA decision boundaries as accurate as Bayes decision boundaries.

In the below example, I'll use both the quantitative `age` and `fare` variables to build an LDA classifier for whether a passenger `survived` or not.  Before creating the classifier, let's visualize the distribution of the two predictors to access their normality.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Creating matplotlib subplots
fig, axes = plt.subplots(1, 2)

# Plotting side-by-side histograms
# kde=True draws a kernel density estimation of probability density function
sns.histplot(titanic["age"], kde=True, color="skyblue", ax=axes[0])
sns.histplot(titanic["fare"], kde=True, color="olive", ax=axes[1])
```

![2021-02-07-linear-discriminant-analysis-002-fig-1.png](/assets/img/2021-02-07-linear-discriminant-analysis-002-fig-1.png){: .mx-auto.d-block :}

The two distributions are clearly skewed right, with `fare` having a stronger skew than `age`.  For the sake of this working example, we'll just keep this in mind as we continue.

Next, we'll perform our linear discriminant analysis.

```python
X = titanic[["age", "fare"]]

# Initializing new LDA estimator and fitting
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Getting correct classification rate
lda.score(X, y)
```

The correct classification rate for this model, obtained via the `score()` method, is approximately 0.647.  Interesting, including `age` seemed to cause our model to perform worse.

It should be noted, that the correct classification rate here is the rate on the training data.  We would expect our model to perform better on training data than test data.

### Null Error Rate

Worth discussing before we continue is the null error rate determined by the *null classifier*.  The null classifier is simply a classifier that always classifies an observation to the majority class.  For our use case, the null classifier would predict that every passenger died on the Titanic, and it would be correct for 59% of our training data.

```python
sum(titanic["survived"] == 0) / len(titanic)   # 0.594
```

As such, or best LDA classification rate of 65.3% above is not much better than the null error rate. 

### Classification Metrics

Next, we'll explore some common metrics to access the performance of our binary classifier.  In particular, we'll explore *accuracy*, *precision*, *recall*, and the $F_1-score$.

First however, let's create a *confusion matrix* of our observed and predicted values.  A confusion matrix is a special type of contingency table that shows the number of true positives, false positives, false negatives, and true negatives from a binary classification task.  We can use scikit-learns' `confusion_matrix()` function for this.

```python
>>> from sklearn.metrics import confusion_matrix

# Assigning predicted y values
>>> y_pred = lda.predict(X)

# Creating confusion matrix
>>> confusion_matrix(y_true=y, y_pred=y_pred)
array([[397,  27],
       [225,  65]])
```

We can also plot this confusion matrix for a visual representation.

```python
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(estimator=lda,
                      X=X,
                      y_true=y,
                      display_labels=["Did not survive", "Survived"])
```

![2021-02-07-linear-discriminant-analysis-002-fig-1.png](/assets/img/2021-02-07-linear-discriminant-analysis-002-fig-1.png){: .mx-auto.d-block :}

### Start here
# Discussing scikit-learns' classification report and the resulting metrics





