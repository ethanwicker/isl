import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

boston = load_boston()

X = (pd.DataFrame(boston.data, columns=boston.feature_names)
     .loc[:, ["RM", "AGE"]]
     .rename(columns=dict(RM="mean_rooms_per_dwelling",
                          AGE="prop_built_prior_1940")))

y = pd.DataFrame(boston.target, columns=["median_value"])

data = pd.concat(objs=[X, y], axis=1)

train = resample(data, replace=True, n_samples=len(data))   # change 5 here to len(data)
test = data[~data.index.isin(train.index)]

lin_reg = LinearRegression()

X_train = train.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
y_train = train.loc[:, ["median_value"]]

X_test = test.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
y_test = test.loc[:, ["median_value"]]

lin_reg.fit(X_train, y_train)

# Loop through all these and collect 1000 times
# Then do something similar for a different type of model
# Maybe lasso reg with C = 0.1 and the score to drive home point that can be done with any type of model
lin_reg.coef_
lin_reg.intercept_
lin_reg.score(X_test, y_test)

intercept = lin_reg.intercept_
beta_x1 = lin_reg.coef_.ravel()[0]
beta_x2 = lin_reg.coef_.ravel()[1]
r2 = lin_reg.score(X_test, y_test)

stats = pd.DataFrame(data=dict(intercept=intercept,
                               beta_x1=beta_x1,
                               beta_x2=beta_x2,
                               r2=r2))
## Writing for loop below
## CLEAN UP THIS LOOP SOME MORE
# delete code above
# next steps: take results, plot, get SE and CI
# do for the lasso C = 0.01 model, just for demo purposes
# but keep all features (cause regression)
# and only do the SE and CI and plot for r2
# then write up supporting post


n_iterations = 1000
lin_reg = LinearRegression()

# Initialzing stats
stats = pd.DataFrame()

for i in range(n_iterations):
    train = resample(data, replace=True, n_samples=len(data))  # change 5 here to len(data)
    test = data[~data.index.isin(train.index)]

    X_train = train.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
    y_train = train.loc[:, ["median_value"]]

    X_test = test.loc[:, ["mean_rooms_per_dwelling", "prop_built_prior_1940"]]
    y_test = test.loc[:, ["median_value"]]

    lin_reg.fit(X_train, y_train)

    intercept = lin_reg.intercept_
    beta_x1 = lin_reg.coef_.ravel()[0]
    beta_x2 = lin_reg.coef_.ravel()[1]
    r2 = lin_reg.score(X_test, y_test)

    stats_i = pd.DataFrame(data=dict(intercept=intercept,
                                     beta_x1=beta_x1,
                                     beta_x2=beta_x2,
                                     r2=r2))

    stats = pd.concat(objs=[stats, stats_i])













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