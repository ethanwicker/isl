# Ethan Wicker

# Python for Logistic Regression #2 Post

# Things to Include:
# • 1 predictor logistic regression in scikit-learn and statsmodels
# • A plot showing the results
# • Comment on maximum likelihood estimation
# • Multiple Logistic Regression in scikit-learn and statsmodels, using qualitative variables
# • Multiclass Logistic Regression (hopefully using both scikit-learn and statsmodels)

# Plan
# • Use age + sex + fare to predict survived
# • Then use age + sex + survived to predict ticket_class (multiclass logistic  regression)

import numpy as np
import pandas as pd
import janitor   # just for fun
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.families.family import Binomial
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv("~/python/isl/data/titanic-train-kaggle.csv")

titanic.info()
titanic.head()

titanic = (janitor
          .clean_names(titanic)
          .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
          .rename(columns={"pclass": "ticket_class"}))

# Age is NA is 177 observations
# For sake of example, throwing these observations out
titanic.isna().sum()

# Removing NA age values
titanic = titanic.query("age.notna()").reset_index()

# Comment on class imbalances (might be affecting model, but won't dive into here)


#########################################

# scikit-learn
X = titanic["age"].to_numpy()
y = titanic["survived"].to_numpy()

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)   # score method is misclassification rate

np.unique(y_hat)  # It's only predicting 0, crappy model.  Maybe if I upsampled?

# Manually confirming that score is misclassification rate
y_hat = log_reg.predict(y)
np.sum(y.reshape(-1) == y_hat) / len(y)


#########################################

# Using statsmodels discrete_model.Logit()
# Maybe discuss pseudo R^2
model = Logit(endog=y, exog=sm.add_constant(X))
result = model.fit()
result.summary()
probs = model.predict(params=result.params)

# Calculating misclassification error rate: same as scikit-learn's score
# Model is also predicting only 0's
y_hat = (probs >= 0.5).astype(int)
np.sum(y.reshape(-1) == y_hat) / len(y)


#########################################

# Usig sm.GLM and statsmodels.genmod.families.family.Binomial
model = sm.GLM(endog=y, exog=sm.add_constant(X), family=Binomial())
result = model.fit()
result.summary()

probs = model.predict(params=result.params)

# Calculating misclassification error rate: same as scikit-learn's score
# Model is also predicting only 0's
y_hat = (probs >= 0.5).astype(int)
np.sum(y.reshape(-1) == y_hat) / len(y)





#########################################


# plotting my data

sns.boxplot(x="survived", y="age", data=titanic)


ax = sns.boxplot(x="day", y="total_bill", data=tips)

# This works for plotting
sns.regplot(x="age", y="survived", data=titanic, logistic=True)

# This also works for plotting
plt.scatter(x=X, y=y)
plt.scatter(x=X, y=probs)





## Trying to improve fit with other variables

# scikit-learn
X = titanic[["sex", "age", "fare"]]
y = titanic["survived"].to_numpy()

X = X.assign(sex_binary=(X["sex"] == "male")).drop("sex", axis=1)  # I should use the scikit-learn way of doing this, and then in a pipe

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)   # much improved score of 0.77731

log_reg.predict(X)

## Trying same as above with OneHotEncoder

X = titanic[["sex", "age", "fare"]].to_numpy()
y = titanic["survived"].to_numpy()

## Let's continue with this to finish it
# Goal:
# Convert age to a OneHotEncoded dense matrix manually, and then combine together
# Do the same for statsmodels both sm.OLS and smf.ols
# Then do predict survived or not
# Discussing upsampling may improve results, but don't do here
# Show a 3D plot with a 3D logistic plane
# Then predict the ticket_class just to show that multiclass/multinomial logistic regression is possible and familiarize myself with it


# This isn't working: Use what is on the link below to do the column stuff in scikit-learn
encoder = OneHotEncoder(sparse=False)
X_train = np.concatenate(encoder.fit_transform(X[:, 0].reshape(-1, 1)),
                         X[:, 1:])

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X_train, y=y)
log_reg.score(X, y)


# useful thread maybe: https://stackoverflow.com/questions/59481354/dummify-categorical-variables-for-logistic-regression-with-pandas-and-scikit-on

# Definitely write a blog post summarizing this information:
# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
















## Can delete below



ax = df_predictions.plot.scatter(x="x_train_points", y="x_train_probability_being_class_1", figsize=(10, 10), c='blueviolet')
plt.title("Predictions on Training Data", y=1.015, fontsize=20)
plt.xlabel("yearly income feature scaled", labelpad=14)
plt.ylabel("probability of a prediction being class 1 (accepted)", labelpad=14)
plt.axhline(y=0.5, linestyle="--", color='green')
bbox_props_decision_threshold = dict(boxstyle="round", fc="snow", ec="0.8", alpha=0.8)
ax.text(0.35, 0.53, "Decision Threshold", bbox=bbox_props_decision_threshold, size=25);






from yellowbrick.classifier import ClassificationReport
from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()
visualizer = ClassificationReport(log_reg)
visualizer.fit(X, y)
visualizer.score(X, y)
visualizer.show()

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show()
