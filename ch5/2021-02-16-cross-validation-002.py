# Note: Do a quadratic logistic regression in this post

# Plan:
# Do a simple validation set
# Then a LOOCV example
# Then a 10-fold CV example and compare the below models
# Using a toy dataset, train a LDA, QDA, logistic regression and quartic logistic regression model

# train_test_split helper https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

# >>> import numpy as np
# >>> from sklearn.model_selection import train_test_split
# >>> from sklearn import datasets
# >>> from sklearn import svm
#
# >>> X, y = datasets.load_iris(return_X_y=True)
# >>> X.shape, y.shape
# ((150, 4), (150,))
#
# >>> X_train, X_test, y_train, y_test = train_test_split(
# ...     X, y, test_size=0.4, random_state=0)
#
# >>> X_train.shape, y_train.shape
# ((90, 4), (90,))
# >>> X_test.shape, y_test.shape
# ((60, 4), (60,))
#
# >>> clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# >>> clf.score(X_test, y_test)
# 0.96...

# Discuss cross_val_score
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default, (think StratifiedKFold here if doing classifier)
# talk about pipelines and using to prevent data leak
# the cross_validate function
# KFold
# LeaveOneOut
# StratifiedKFold to keep same percentage of class labels as in original dataset
# GroupKFold
# TimeSeriesSplit
# mention many more can be found here: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv

#####
# Start below
#####

# • Explain what cross validation is
# • In a previous post, I introduced much of the theory behind CV
# • In this post, I'll explore some working example using CV from scikit-learn
# • In particular, I'll explore simple validation sets using train_test_split, `KFold` CV, `LeaveOneOut`, `StratifiedKFold` CV,
# • GroupKFold   # will just discuss this, won't actually do it
# • TimeSeriesSplit    # will just discuss this, won't actually do it
# • mention many more can be found here: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv

## Note on scaling:
# For logistic regression (non-regularized), LDA, and QDA, I shouldn't need to scale my observations
# But going to anyway just because it typically doesn't hurt
# And compare results afterwards to see if equal or not
# Also all of Iris in the same units (cm)
##

####
# Getting Iris Dataset (classification task)
####

import pandas as pd

from sklearn.datasets import load_iris

# X, y = load_iris(return_X_y=True)

# Loading Iris as pandas DataFrame
iris = load_iris(as_frame=True)

# Creating target_names DataFrame to match up target with target_name
iris_target_names = pd.DataFrame(data=dict(target=[0, 1, 2], target_name=iris.target_names))

# Merging
data = (iris["frame"]
        .merge(right=iris_target_names,
               on="target",
               how="left"))

####
# Plot Iris as a scatterplot and show all the classes with different colors
####

import seaborn as sns

sns.scatterplot(x="sepal length (cm)",
                y="sepal width (cm)",
                hue="target_name",
                data=data)

####
# Try LDA by itself, don't scale, and use train_test_split
####

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Loading Iris again, returning X and y as NumPy arrays
X, y = load_iris(return_X_y=True)

# Simple validation set split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    shuffle=True)

lda = LinearDiscriminantAnalysis()
lda.fit_transform(X=X, y=y)  # I don't think y here is being considered a categorical variable

####
# Creating a pipeline to scale data and then train log reg, quadratic log reg, LDA, QDA with KFold/really Stratified KFold
# Logistic regression here is going to be a one-vs-rest multi-class logistic regression classifier
####

####
# Same as above with LeaveOneOut
####

####
# Discuss GroupKFold, TimeSeriesSplit, the above link
####




