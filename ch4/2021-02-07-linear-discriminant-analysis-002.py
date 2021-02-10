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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

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
# Comment that this score is on training data, and thus inflated


##########

# the null classifier that predicts every passenger dies always is right 0.594 of the time
sum(titanic["survived"] == 0) / len(titanic)   # 0.594

# Confusion matrix
# metrics.confusion_matrix
# metrics.plot_confusion_matrix
y_hat = lda.predict(X)

# Confusion matrix
confusion_matrix(y_true=y, y_pred=y_hat)

# plot_confusion_matrix, need to get a screenshot
plot_confusion_matrix(estimator=lda,
                      X=X,
                      y_true=y,
                      display_labels=["Did not survive", "Survived"])

# Classification report
print(classification_report(y_true=y, y_pred=y_hat))

# precision = tp / (tp + fp)
# Intuitively: Out of all the predicted positives, how many are actually positive
# When comparing models, precision is a good measure when we want to avoid false positives.
# For example, when detecting spam emails, we do not want to label an email as spam incorrectly.


# recall = tp / (tp + fn)
# Intuitively: Out of all the actual positive samples, how many are labeled as true positive
# When comparing models, recall is a good measure when we want to avoid false negatives
# For example, if you don't want to mislabel someone with an infectious disease as negative

# f1 score = 2 * (precision * recall) / (precision + recall)
# harmonic mean of precision and recall
# Useful when we want to strike a balance between precision and recall

# Should mention accuracy --> overall accuracy of predictions (out of all observations, how many are correctly labeled?

# recall is also called sensitivity

# recall and precision widely used in information theory, as well as machine learning classification problems

# These terms widely used in medicine
# Sensitivity =  (True Positive rate)
# Specificity = (True Negative rate)

##########

# Comparing the same predictors using logistic regression

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X, y)
log_reg.predict(X)
log_reg.score(X, y)  # slightly better than LDA


# Show a ROC/AUC comparison to determine which is the better classifier (higher AUC)
# Discuss what ROC is
# Discuss what AUC is

# Incredibly similar
roc_lda = plot_roc_curve(estimator=lda, X=X, y=y)
roc_log_reg = plot_roc_curve(estimator=log_reg, X=X, y=y, ax=roc_lda.ax_)
plt.title("ROC Curve Comparison")
plt.show()






