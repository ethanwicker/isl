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
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default,
# talk about pipelines and using to prevent data leak
# the cross_validate function
# KFold
# LeaveOneOut
# StratifiedKFold to keep same percentage of class labels as in original dataset
# GroupKFold
# TimeSeriesSplit
# mention many more can be found here: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv