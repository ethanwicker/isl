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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

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

##########

# LDA with p > 1

# Because of the assumption that X = (X_1, X_2, .., X_p) is drawn from a
# multivariate Gaussian distribution, categorical predictors don't play well with
# LDA.  So will use only quantitative predictors.

# Also LDA is somewhat robust to non-Gaussian distributions, but is not guarenteed
# to find an optimal solution.  I'll show the distribution of age and far, and comment on this,
# but won't do anything to correct any non-gaussian distributions.

# Showing distribution of age and fare
# kde = Kernel density estimation
# Both are skewed right, but will ignore this for now
fig, axes = plt.subplots(1, 2)
sns.histplot(titanic["age"], kde=True, color="skyblue", ax=axes[0])
sns.histplot(titanic["fare"], kde=True, color="olive", ax=axes[1])

X = titanic[["age", "fare"]]

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda.predict(X)
lda.score(X, y)   # actually worse than the one above

##########

# Comparing the same predictors using logistic regression

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X, y)
log_reg.predict(X)
log_reg.score(X, y)  # slightly better than LDA

##########

# Comment that both the scores above are on training data, and thus inflated

# the null classifier that predicts every dies always is right 0.594 of the time
sum(titanic["survived"] == 0) / len(titanic)   # 0.594


# Discuss false positive rates (type I error, 1 - specificity), true positive rate (sensitivity, recall), precision

# Show a ROC/AUC comparison to determine which is the better classifer (higher AUC)

