# Post 5: Python example
# Maybe sample my data down to 10,000 rows to make it easier for plotting
#
# Want to include:
# * Qualitative predictors & dummy encoding of some sort   --> include an example of how to do the dummy encoding myself, even with a different dataset
# * Removing the additive assumption: interaction terms
# * Removing the linear assumption: non-linear relationships

# • How to do one hot encoding in scikit-learn/statsmodels

#
# * A comparison of scikit-learn vs. statsmodels
#
# * Discuss these 6 problems (maybe not all, but some)
# 1. Non-linearity of the response-predictor relationships.  <<-- residual plot
# 2. Correlation of error terms.
# 3. Non-constant variance of error terms.
# 4. Outliers.                <<-- maybe studentized residuals
# 5. High-leverage points.    <<-- leverage statistics
# 6. Collinearity.            <<-- VIF

# Maybe also filter out any hours with low flights traffic


# Plan:
# Read in data
# Summarize per hour
# First look at using the orig airport as the categorical variable
# And the carrier
# Filter for carriers with a certain minimum of flights
# Discuss encoding --> does scikit-learn or statsmodels handle this for us?
# Discuss removing the additive assumption --> interaction terms
# Discuss removing the linear assumption --> polynomial regression

# Might be in my benefit to really just do in scikit-learn, although will proabably want to check the interaction term logic via statsmodels and INFERENCE!

# Need to do some visualizations
# Good resouce for plotting: https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/

import pandas as pd
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
from yellowbrick.regressor import ResidualsPlot


# Load data
boston = datasets.load_boston()

# Printing description
print(boston.DESCR)

# Creating DataFrame instead of X and y numpy.ndarrays for ease of use with patsy
X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
y_df = pd.DataFrame(boston.target, columns=["MEDV"])
boston_df = pd.concat([X_df, y_df], axis=1)

# Using smf.ols first for ease of exploration + inference
# Going to train on MEDV as a function of ZN, CHAS, RM, DIS
# We learn that DIS is not significant, so we should drop it
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + DIS", data=boston_df)
result = model.fit()
result.summary()

# Adding interaction term
# R^2 is 0.537
# Discuss : vs. * here
# All terms with CHAS no longer significant, so dropping
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + ZN:CHAS + CHAS:RM + ZN:RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# R^2 = 0.522
model = smf.ols(formula="MEDV ~ ZN + RM + ZN:RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# Removing interaction term for comparison
# R^2 drops to 0.504
model = smf.ols(formula="MEDV ~ ZN + RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# Using sm.OLS
# Manually creating X with interaction term
# Adding constant to X
# Produces same model as smf.ols
X = boston_df[["ZN", "RM"]].assign(ZN_RM=boston_df["ZN"] * boston_df["RM"])
X = sm.add_constant(X)
model = sm.OLS(endog=boston_df["MEDV"], exog=X)
result = model.fit()
result.summary()

# Using sklearn.linear_model
model = linear_model.LinearRegression()
model.fit(X=X, y=boston_df["MEDV"])
model.score(X=X, y=boston_df["MEDV"])    # Same R^2 value as above

# How to do with sklearn and the polynomial regression stuff
# Exploring
poly = PolynomialFeatures(interaction_only=True)
X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])   # include_bias=True is the default here
model = linear_model.LinearRegression()
model.fit(X=X_poly, y=boston_df["MEDV"])
model.score(X=X_poly, y=boston_df["MEDV"])    # Same R^2 value as above

model.show()  # trying to make residual plot with yellowbrick -> not working
























sm.add_constant(test)

model = smf.ols(formula="MEDV ~ ZN + RM + DIS + ZN:RM + RM:DIS + ZN:DIS", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
model = smf.ols(formula="MEDV ~ ZN + RM + DIS", data=boston_df)  # start with this one
result = model.fit()
result.summary()

#sm.OLS
# will need to add intercept
model = sm.OLS(endog=boston_df["MEDV"], exog=sm.add_constant(boston_df[["CHAS", "RM"]]))
result = model.fit()
result.summary()














boston_data = pd.DataFrame({boston.data, boston.target})
pd.

boston_data.columns = boston.feature_names

X = boston_data.loc[:, ["CHAS", "RM"]]
y = boston.target

from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X, y)
model.score(X, y)

import statsmodels.api as sm

# will need to add intercept
model = sm.OLS(X.to_numpy(), pd.DataFrame(y))
result = model.fit()
result.summary()






X.shape
y.shape
result.

X.to_numpy()
pd.DataFrame.to_numpy(X)


boston.data
boston.target
print(boston.DESCR)

X = boston.data
y = boston.target

boston_df = pd.DataFrame(X)
boston_df.columns = boston.feature_names
boston_df

# maybe predict median value of house
# as a function of charles river dummy variable, per capita crime rate (cut into 3 factors), average number of rooms per dwelling
# whats the interaction between charles river dummy variable and average number of rooms per dwelling

CHAS
RM