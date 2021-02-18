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

# Initializing and fitting on training observations
lda = LinearDiscriminantAnalysis()
lda.fit(X=X_train, y=y_train)

lda.predict(X_train) # training predictions
lda.predict(X_test)  # training predictions
lda.score(X=X_train, y=y_train)  # training error rate
lda.score(X=X_test, y=y_test)    # test error rate




####
# KFold/really Stratified KFold
# Creating a pipeline to scale data and then train log reg, quadratic log reg, LDA, QDA with KFold/really Stratified KFold
# Logistic regression here is going to be a one-vs-rest multi-class logistic regression classifier
####

# Something like this what I want
# From: https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import load_iris
# iris_data = load_iris()
# X, y = iris_data.data, iris_data.target
#
#
# # Just initialize the pipeline with any estimator you like
# pipe = Pipeline(steps=[('estimator', SVC())])
#
# # Add a dict of estimator and estimator related parameters in this list
# params_grid = [{
#                 'estimator':[SVC()],
#                 'estimator__C': [1, 10, 100, 1000],
#                 'estimator__gamma': [0.001, 0.0001],
#                 },
#                 {
#                 'estimator': [DecisionTreeClassifier()],
#                 'estimator__max_depth': [1,2,3,4,5],
#                 'estimator__max_features': [None, "auto", "sqrt", "log2"],
#                 },
#                # {'estimator':[Any_other_estimator_you_want],
#                #  'estimator__valid_param_of_your_estimator':[valid_values]
#
#               ]
#
# grid = GridSearchCV(pipe, params_grid)

####
# Same as above with LeaveOneOut
####

####
# Discuss GroupKFold, TimeSeriesSplit, the above link
####




