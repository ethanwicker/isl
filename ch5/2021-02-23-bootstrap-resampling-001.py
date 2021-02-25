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

# Initialzing stats DataFrame, to hold bootstrapped stats
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
sns.histplot(bootstrapped_stats.intercept, color="skyblue", ax=axes[0, 0], kde=True)
sns.histplot(bootstrapped_stats.beta_mean_rooms_per_dwelling, color="olive", ax=axes[0, 1], kde=True)
sns.histplot(bootstrapped_stats.beta_prop_built_prior_1940, color="gold", ax=axes[1, 0], kde=True)
sns.histplot(bootstrapped_stats.r_squared, color="teal", ax=axes[1, 1], kde=True)

# Getting standard deviation of measurements
# Maybe just show one
import scipy.stats as st
st.tstd(bootstrapped_stats.intercept)
st.tstd(bootstrapped_stats.beta_mean_rooms_per_dwelling)
st.tstd(bootstrapped_stats.beta_prop_built_prior_1940)
st.tstd(bootstrapped_stats.r_squared)





# Getting confidence intervals
# This are the same, tstd="trimmed standard deviation", note ddof=1 by default in st.tstd
import numpy as np
np.std(bootstrapped_stats.intercept, ddof=1)
st.tstd(bootstrapped_stats.intercept)

# Using normal here because 1000 samples, would use t-distribution for smaller sample size (but CLT)
# Just show one here (do beta_mean_rooms_per_dwelling)
# And then just for that one, show it the CI on the plot
# Then do the lasso regularization stuff
norm.interval(alpha=0.95, loc=np.mean(bootstrapped_stats.intercept), scale=st.tstd(bootstrapped_stats.intercept))

# Sanity check
plt.axvline(x=-34.07275877662597, ax=axes[0, 0])
axes[0, 0].axvline(x=-34.07275877662597, color="red")
axes[0, 0].axvline(x=-16.60852219208411, color="red")


