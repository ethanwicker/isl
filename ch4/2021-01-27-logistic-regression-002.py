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

import janitor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from statsmodels.discrete.discrete_model import Logit
from statsmodels.genmod.families.family import Binomial


# Reading data
titanic = pd.read_csv("~/python/isl/data/titanic-train-kaggle.csv")

# Viewing data
titanic.info()
titanic.head()

# Cleaning up field names
titanic = (janitor
          .clean_names(titanic)
          .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
          .rename(columns={"pclass": "ticket_class"}))

# Age is NA is 177 observations
# For sake of example, throwing these observations out
# In future post, will impute using median
titanic.isna().sum()
titanic = titanic.query("age.notna()").reset_index()

# Comment on class imbalances (might be affecting model, but won't dive into here)
titanic["survived"].value_counts()

#########################################

# scikit-learn
X = titanic[["age"]]
y = titanic["survived"]

# Discuss why penalty here is none
# Scoring method here is misclassification error rate
log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)


# Manually confirming that score is misclassification rate
y_hat = log_reg.predict(y.to_numpy().reshape(-1, 1))
np.sum(y == y_hat) / len(y)

sklearn_score = log_reg.score(X, y)
misclass_error_rate = np.sum(y == y_hat) / len(y)

np.equal(sklearn_score, misclass_error_rate)

# All y_hats are 0, model is not very good
np.unique(y_hat)

# Plotting model
sns.regplot(x="age", y="survived", data=titanic, logistic=True)


#########################################

# Using statsmodels discrete_model.Logit()
# Discuss pseudo R^2
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
np.sum(y == y_hat) / len(y)


#########################################


## Trying to improve fit with other variables
# scikit-learn

## Let's continue with this to finish it
# Goal:
# Then predict the ticket_class just to show that multiclass/multinomial logistic regression is possible and familiarize myself with it
# Comment in future posts will discuss scikit-learn pipelines

# Discussing upsampling may improve results, but don't do here

# Initializing one-hot encoder
encoder = OneHotEncoder(sparse=False, drop="first")

# Encoding categorical fiedl
X_categorical = titanic[["sex"]]
X_categorical = encoder.fit_transform(X_categorical)

# Nummeric fields
X_numeric = titanic[["age", "fare"]]

# Concatenating NumPy arrays together
X = np.concatenate((X_categorical, X_numeric), axis=1)

log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)

# Now want to do via statsmodels
# This works
model = sm.GLM(endog=y, exog=sm.add_constant(X), family=Binomial())
result = model.fit()
result.summary()
probs = model.predict(params=result.params)
y_hat = (probs >= 0.5).astype(int)
np.sum(y.reshape(-1) == y_hat) / len(y)  # same .77731 value

# From the output we see that age is not significant
# Let's make a 3D graph showing just sex and fare as the predictors


#########################################


# Now creating 3D plot

# Define mash size for prediction surface
mesh_size = .02

# Removing age field because not significant
X_sex_fare = np.delete(X, obj=1, axis=1)

# Fitting model
log_reg.fit(X=X_sex_fare, y=y)

# Define x and y ranges
# Note: y here refers to y-dimension, not response variable
x_min, x_max = X_sex_fare[:, 0].min(), X_sex_fare[:, 0].max()
y_min, y_max = X_sex_fare[:, 1].min(), X_sex_fare[:, 1].max()
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Get predictions for all values of x and y ranges
pred = log_reg.predict(X=np.c_[xx.ravel(), yy.ravel()])

# Reshape predictions to match mesh shape
pred = pred.reshape(xx.shape)

# Plotting
fig = px.scatter_3d(titanic,
                    x='sex',
                    y='fare',
                    z='survived',
                    labels=dict(sex="Sex (1 if male, 0 if female)",
                                fare="Ticket Fare",
                                survived="Survived"))
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
fig.show()

#########################

# Multiclass Logistic Regression with scikit-learn
encoder = OneHotEncoder(sparse=False, drop="first")

# Encoding categorical fiedl
X_categorical = titanic[["sex"]]
X_categorical = encoder.fit_transform(X_categorical)

# Nummeric fields
X_numeric = titanic[["age", "survived"]]

# Concatenating NumPy arrays together
X = np.concatenate((X_categorical, X_numeric), axis=1)

y = titanic[["ticket_class"]].to_numpy().ravel()

# Uses a one-vs-rest scheme by default
log_reg = LogisticRegression(penalty="none")
log_reg.fit(X=X, y=y)
log_reg.score(X, y)

log_reg.predict(X)[:10]



