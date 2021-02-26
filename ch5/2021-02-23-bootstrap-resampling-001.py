## Writing for loop below
## CLEAN UP THIS LOOP SOME MORE
# delete code above
# next steps: take results, plot, get SE and CI
# do for the lasso C = 0.01 model, just for demo purposes
# but keep all features (cause regression)
# and only do the SE and CI and plot for r2
# then write up supporting post

import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Loading boston dataset
boston = load_boston()

# Selecting just two fields and renaming
X = (pd.DataFrame(boston.data, columns=boston.feature_names)
     .loc[:, ["RM", "AGE"]]
     .rename(columns=dict(RM="mean_rooms_per_dwelling",
                          AGE="prop_built_prior_1940")))

y = pd.DataFrame(boston.target, columns=["median_value"])

data = pd.concat(objs=[X, y], axis=1)

# Defining number of iterations for bootstrap resample
n_iterations = 1000

# Initializing estimator
lin_reg = LinearRegression()

# Initializing DataFrame, to hold bootstrapped statistics
bootstrapped_stats = pd.DataFrame()

# Each loop iteration is a single bootstrap resample and model fit
for i in range(n_iterations):

    # Sampling n_samples from data, with replacement, as train
    # Defining test to be all observations not in train
    train = resample(data, replace=True, n_samples=len(data))
    test = data[~data.index.isin(train.index)]

    X_train = train.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
    y_train = train.loc[:, ["median_value"]]

    X_test = test.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
    y_test = test.loc[:, ["median_value"]]

    # Fitting linear regression model
    lin_reg.fit(X_train, y_train)

    # Storing stats in DataFrame, and concatenating with stats
    intercept = lin_reg.intercept_
    beta_mean_rooms_per_dwelling = lin_reg.coef_.ravel()[0]
    beta_prop_built_prior_1940 = lin_reg.coef_.ravel()[1]
    r_squared = lin_reg.score(X_test, y_test)

    bootstrapped_stats_i = pd.DataFrame(data=dict(
        intercept=intercept,
        beta_mean_rooms_per_dwelling=beta_mean_rooms_per_dwelling,
        beta_prop_built_prior_1940=beta_prop_built_prior_1940,
        r_squared=r_squared
    ))

    bootstrapped_stats = pd.concat(objs=[bootstrapped_stats,
                                         bootstrapped_stats_i])

import seaborn as sns
import matplotlib.pyplot as plt

# plot
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.histplot(bootstrapped_stats["intercept"], color="royalblue", ax=axes[0, 0], kde=True)
sns.histplot(bootstrapped_stats["beta_mean_rooms_per_dwelling"], color="olive", ax=axes[0, 1], kde=True)
sns.histplot(bootstrapped_stats["beta_prop_built_prior_1940"], color="gold", ax=axes[1, 0], kde=True)
sns.histplot(bootstrapped_stats["r_squared"], color="teal", ax=axes[1, 1], kde=True)

# Getting standard deviation of measurements (just showing one)
import scipy.stats as st
st.tstd(bootstrapped_stats["beta_mean_rooms_per_dwelling"])
# Comment: This are the same, tstd="trimmed standard deviation", note ddof=1 by default in st.tstd
import numpy as np
np.std(bootstrapped_stats.intercept, ddof=1)
st.tstd(bootstrapped_stats.intercept)

# Using normal here because 1000 samples, would use t-distribution for smaller sample size (but CLT)
ci = st.norm.interval(alpha=0.95,
                      loc=np.mean(bootstrapped_stats["beta_mean_rooms_per_dwelling"]),  # mean
                      scale=st.tstd(bootstrapped_stats["beta_mean_rooms_per_dwelling"]))  # standard deviation

# Plotting confidence intervals
sns.histplot(bootstrapped_stats["beta_mean_rooms_per_dwelling"], color="olive", kde=True)
plt.axvline(x=ci[0], color="red")
plt.axvline(x=ci[1], color="red")

########
# Same thing, for Lasso and R^2
#######

from sklearn.linear_model import Lasso

# Could have also used this instead of hstack
# np.concatenate([boston.data, boston.target.reshape(-1,1)], axis=1)
data_all = pd.DataFrame(
    data=np.hstack((boston.data, boston.target.reshape(-1, 1))),
    columns=np.concatenate([boston.feature_names, ["median_value"]])
)

# Defining number of iterations for bootstrap resample
n_iterations = 1000

# Initializing estimator
lasso = Lasso()

# Initializing DataFrame, to hold bootstrapped statistics
bootstrapped_lasso_r2 = pd.Series()

# Each loop iteration is a single bootstrap resample and model fit
for i in range(n_iterations):

    # Sampling n_samples from data, with replacement, as train
    # Defining test to be all observations not in train
    train = resample(data_all, replace=True, n_samples=len(data_all))
    test = data_all[~data_all.index.isin(train.index)]

    X_train = train.iloc[:, 0:-1]
    y_train = train.iloc[:, -1]

    X_test = test.iloc[:, 0:-1]
    y_test = test.iloc[:, -1]

    # Fitting linear regression model
    lasso.fit(X_train, y_train)

    # Storing stats in DataFrame, and concatenating with stats
    r_squared = lasso.score(X_test, y_test)

    bootstrapped_lasso_r2_i = pd.Series(data=r_squared)

    bootstrapped_lasso_r2 = pd.concat(objs=[bootstrapped_lasso_r2,
                                            bootstrapped_lasso_r2_i])


##########
# Include the below as well
##########

# Need to update the above to use the same data (do "data" not "data_all")

# Note these values are differnt from the ones created above
# Visually see all the different models we're producing
# This is bagging aggregation
# Include just to show how the different models all look
# With bootstrapping we are estimating the CIs of (in this case) beta_0 and beta_1
# Actual bagging will come in a later post, but show the last line of this just to demonstrating boostrap aggregation for regression prediction

# If you want to use scikit's API for the bootstrap part of the code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor

# Create toy data
x = np.linspace(0, 10, 20)
y = x + (np.random.rand(len(x)) * 10)

# Extend x data to contain another row vector of 1s
X = np.vstack([x, np.ones(len(x))]).T

n_estimators = 50
model = BaggingRegressor(LinearRegression(),
                         n_estimators=n_estimators,
                         bootstrap=True)

model.fit(X, y)

plt.figure(figsize=(12,8))

# Accessing each base_estimator (already fitted)
for m in model.estimators_:
    plt.plot(x, m.predict(X), color='grey', alpha=0.2, zorder=1)

plt.scatter(x,y, marker='o', color='orange', zorder=4)

# "Bagging model" prediction
plt.plot(x, model.predict(X), color='red', zorder=5)