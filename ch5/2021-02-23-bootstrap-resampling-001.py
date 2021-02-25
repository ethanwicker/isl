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
stats = pd.DataFrame()

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

    stats_i = pd.DataFrame(data=dict(intercept=intercept,
                                     beta_mean_rooms_per_dwelling=beta_mean_rooms_per_dwelling,
                                     beta_prop_built_prior_1940=beta_prop_built_prior_1940,
                                     r_squared=r_squared))

    stats = pd.concat(objs=[stats, stats_i])

import seaborn as sns
import matplotlib.pyplot as plt

# plot
fig, axes = plt.subplots(2, 2, figsize=(7, 7))
sns.histplot(stats.intercept, color="skyblue", ax=axes[0, 0], kde=True)
sns.histplot(stats.beta_mean_rooms_per_dwelling, color="olive", ax=axes[0, 1], kde=True)
sns.histplot(stats.beta_prop_built_prior_1940, color="gold", ax=axes[1, 0], kde=True)
sns.histplot(stats.r_squared, color="teal", ax=axes[1, 1], kde=True)

from scipy.stats import sem
sem(stats.intercept)
sem(stats.beta_mean_rooms_per_dwelling)
sem(stats.beta_prop_built_prior_1940)
sem(stats.r_squared)

scipy.stats.norm.interval

from scipy.stats import norm
import numpy as np

norm.interval(alpha=0.95, loc=np.mean(stats.intercept), scale=sem(stats.intercept))

# I THINK THE ABOVE IS WRONG:
# compare and contrast with this: stats.intercept.describe()
# Stuff does not seem to be adding up
# There has to be an easier way to do this, probably via scipy.stats
# Think through exactly what I want, and maybe manually calculate it as a santity check
















# DELETE BELOW

import numpy
from pandas import read_csv
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
# load dataset
data = read_csv('pima-indians-diabetes.data.csv', header=None)
values = data.values
# configure bootstrap
n_iterations = 1000
n_size = int(len(data) * 0.50)
# run bootstrap
stats = list()
for i in range(n_iterations):
	# prepare train and test sets
	train = resample(values, n_samples=n_size)
	test = numpy.array([x for x in values if x.tolist() not in train.tolist()])
	# fit model
	model = DecisionTreeClassifier()
	model.fit(train[:,:-1], train[:,-1])
	# evaluate model
	predictions = model.predict(test[:,:-1])
	score = accuracy_score(test[:,-1], predictions)
	print(score)
	stats.append(score)
# plot scores
pyplot.hist(stats)
pyplot.show()
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, numpy.percentile(stats, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, numpy.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))