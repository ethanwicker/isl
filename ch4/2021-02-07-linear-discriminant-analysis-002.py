# Plan
# Perform LDA in the simple case where p=1
# Perform LDA when p>1
# Discuss false positive rates (type I error, 1 - specificity), true positive rate (sensitivity, recall), precision
# Discuss ROC curve
# Compare with logistic regression to indicate that ROC curves are useful for comparing classifiers to each other
# • Compare LDA vs logistic regression --> higher AUC is better

# LDA is available in
# scikit-learn
# mlpy (hasn't been updated since 2012)
# MDP (hasn't been updated since 2016)
# PyMVPA (recently updated, but scikit-learn far more popular - 66 downloads in past month vs. 7.3 million for scikit-learn)

# Consider just using the titanic dataset again, would be a simple comparison again logistic regression

import janitor
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

titanic = pd.read_csv("~/python/isl/data/titanic-train-kaggle.csv")

# Cleaning up field names
titanic = (janitor
          .clean_names(titanic)
          .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
          .rename(columns={"pclass": "ticket_class"}))

# Age is NA is 177 observations
# For sake of example, throwing these observations out
# In future post, will impute using median
titanic.isna().sum()
titanic = titanic.query("age.notna()").reset_index(drop=True)

##########

# LDA with p = 1

# scikit-learn
X = titanic[["fare"]]
y = titanic["survived"]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda.predict(X)
lda.score(X, y)

# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# y = np.array([1, 1, 1, 2, 2, 2])
# clf = LinearDiscriminantAnalysis()
# clf.fit(X, y)
# print(clf.predict([[-0.8, -1]]))
# clf.score(X, y)