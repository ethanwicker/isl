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



# Plan:
# Read in data
# Summarize per hour
# First look at using the orig airport as the categorical variable
# And the carrier
# Filter for carriers with a certain minimum of flights
# Discuss encoding --> does scikit-learn or statsmodels handle this for us?
# Discuss removing the additive assumption --> interaction terms
# Discuss removing the linear assumption --> polynomial regression


# Need to do some visualizations
# Good resouce for plotting: https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/

import numpy as np
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

# Doing one-hot encoding
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

pd.cut(boston_df["CRIM"], bins=3, labels=["low_crime", "medium_crime", "high_crime"])
boston_df = \
    (boston_df
     .assign(crime_label=pd.cut(boston_df["CRIM"],
                               bins=3,
                               labels=["low_crime", "medium_crime", "high_crime"])))

enc = OneHotEncoder()
#enc.fit(boston_df["crime_label"])
X_for_enc = boston_df["crime_label"].to_numpy().reshape(-1, 1)
test = enc.fit_transform(X_for_enc)
test.toarray()

## Could also use pd.get_dummies -> comment on this, but can't use in a scikit-learn pipeline

pd.DataFrame(data=test.toarray(), columns=enc.categories_)
test.categories_
#enc.fit(X_for_enc)
enc.categories_
test = enc.transform(X_for_enc)
enc.get_feature_names()

test = enc.fit_transform(X_for_enc)
test.toarray()
test.get_feature_names()

## Example
oe_style = OneHotEncoder()
oe_results = oe_style.fit_transform(obj_df[["body_style"]])
pd.DataFrame(oe_results.toarray(), columns=oe_style.categories_).head()
##


enc.get_features_names()

pd.DataFrame(test)

enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)



enc.categories_

enc.transform([['Female', 1], ['Male', 4]]).toarray()


enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


enc.get_feature_names(['gender', 'group'])



boston_df.groupby("crime_label").count()

test
boston_df

enc.categories_

enc.categories_

enc.transform([['Female', 1], ['Male', 4]]).toarray()


enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


enc.get_feature_names(['gender', 'group'])




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
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + ZN:CHAS + CHAS:RM + ZN:RM", data=boston_df)
result = model.fit()
result.summary()

# R^2 = 0.522
model = smf.ols(formula="MEDV ~ ZN + RM + ZN:RM", data=boston_df)
result = model.fit()
result.summary()

# Removing interaction term for comparison
# R^2 drops to 0.504
model = smf.ols(formula="MEDV ~ ZN + RM", data=boston_df)
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

# Residual plot shows a slight non-linearity, yellowbrick
model2 = ResidualsPlot(linear_model.LinearRegression())
model2.fit(X=X_poly, y=boston_df["MEDV"])
model2.show()

# Let's try some polynomial regressions in the model see if can get residual plot to improve
# Using smf.ols
# I() sometimes called the identifiy or indicator or "as-is" function
model = smf.ols(formula="MEDV ~ ZN + RM + I(ZN**2) + I(RM**2) + ZN:RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# Trying polynomial regression again with np.power()
# Only trying up to degreee 2
# Exact same result as about with indicator function
model = smf.ols(formula="MEDV ~ ZN + RM + np.power(ZN, 2) + np.power(RM, 2) + ZN:RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# np.power(ZN, 2) not significant, so let's drop it
# Also neither is ZN, but keeping it cause hierarchy principle
# Actually they're all significant at the 0.05 level
model = smf.ols(formula="MEDV ~ ZN + RM + np.power(RM, 2) + ZN:RM", data=boston_df)
result = model.fit()
result.summary()

# Basically doing "Mixed Selection" here, with a sig level of 0.01
model = smf.ols(formula="MEDV ~ RM + np.power(RM, 2)", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

# Let's use scikit-learn to get the same result, for practice
# Then yellow brick again
# Couldn't find any simple way to perform the polynomial regression on just the subset of predictors
# after taking out the ZN variable, leaving it like this for now
poly = PolynomialFeatures()
X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])   # include_bias=True is the default here
model = linear_model.LinearRegression()
model.fit(X=X_poly, y=boston_df["MEDV"])
model.score(X=X_poly, y=boston_df["MEDV"])

model.predict(X=X_poly)

# Let's do the residual plot again on this module
# Similar pattern
model2 = ResidualsPlot(linear_model.LinearRegression())
X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])
model2.fit(X=X_poly, y=boston_df["MEDV"])
model2.show()



## Trying 3D scatter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('2016.csv')
sns.set(style = "darkgrid")

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = boston_df["ZN"]
y = boston_df["RM"]
z = boston_df["MEDV"]

ax.set_xlabel("ZN")
ax.set_ylabel("RM")
ax.set_zlabel("MEDV")

ax.scatter(x, y, z)
ax.scatter(x, y, model.predict(X=X_poly))

# need to add this
# create a wiremesh for the plane that the predicted values will lie
xx, yy, zz = np.meshgrid(x, y, z)
combinedArrays = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
Z = combinedArrays.dot(a)
# and this
ax.plot_trisurf(combinedArrays[:, 0], combinedArrays[:, 1], Z, alpha=0.5)
# Finish with tihs and see if it works
# but the will try plot_ly

ax.legend()


#
# ax.plot_surface(x, y,
#                 model.predict(X=X_poly).reshape(),
#                 rstride=1,
#                 cstride=1,
#                 color='None',
#                 alpha = 0.4)

plt.show()



np.reshape(model.predict(X=X_poly))





## THIS MIGHT HAVE ACTUALY WORKED

# Trying out plotly
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVR

mesh_size = .02
margin = 0


# df = px.data.iris()
#
# X = df[['sepal_width', 'sepal_length']]
# y = df['petal_width']

X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])
X = X_poly
y = boston_df["MEDV"]

# Condition the model on sepal width and length, predict the petal width
# model = SVR(C=1.)
# model.fit(X, y)

poly = PolynomialFeatures()
X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])   # include_bias=True is the default here
model = linear_model.LinearRegression()
model.fit(X=X_poly, y=boston_df["MEDV"])
model.score(X=X_poly, y=boston_df["MEDV"])

# Trying different model
model = smf.ols(formula="MEDV ~ ZN + RM + I(ZN**2) + I(RM**2) + ZN:RM", data=boston_df)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()

exog = pd.DataFrame(columns=dict(ZN=boston_df["ZN"], ZN_squared=boston_df["ZN"]**2))
exog = np.array([[boston_df["ZN"],
                 [boston_df["ZN"]**2],
                 [boston_df["RM"]],
                 [boston_df["RM"]**2],
                 np.multiply([boston_df["ZN"]], [boston_df["ZN"]])]))

np.array([[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]])
array=np.array([boston_df["ZN"],
         boston_df["ZN"]**2,
         boston_df["RM"]]).T

exog = pd.DataFrame(data=array, columns=["a", "b", "c"])
exog = sm.add_constant(exog)

model = sm.OLS(endog=y,exog=exog)  # then this one, and the next one take out the terms that aren't significant
result = model.fit()
result.summary()
# no constant here - being fooled, but just added in the constant

model.predict(params=result.params)

# Create a mesh grid on which we will run our model
x_min, x_max = boston_df.ZN.min() - margin, boston_df.ZN.max() + margin
y_min, y_max = boston_df.RM.min() - margin, boston_df.RM.max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Create a mesh grid on which we will run our model
# x_min, x_max = X.sepal_width.min() - margin, X.sepal_width.max() + margin
# y_min, y_max = X.sepal_length.min() - margin, X.sepal_length.max() + margin
# xrange = np.arange(x_min, x_max, mesh_size)
# yrange = np.arange(y_min, y_max, mesh_size)
# xx, yy = np.meshgrid(xrange, yrange)

# Run model
# The model is just crappy - it's performing poorly at the higher ends
pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
pred = model.predict(np.c_[xx.ravel(), xx.ravel()**2, yy.ravel()])  # do this
pred = model.predict(exog=np.c_[xx.ravel(), xx.ravel()**2, yy.ravel(), yy.ravel()**2, xx.ravel()*yy.ravel()])  # don't think this works
pred = model.predict(params=result.params, exog=np.c_[np.ones(shape=xx.ravel().shape), xx.ravel(), xx.ravel()**2, yy.ravel()])  # this also works for ZN + ZN^2 + RM example
## This is exactly what I want with the one's included in the constant
pred = model.predict(params=result.params, exog=np.c_[np.ones(shape=xx.ravel().shape), xx.ravel(), xx.ravel()**2, yy.ravel()])  # this also works for ZN + ZN^2 + RM example
pred = pred.reshape(xx.shape) # and this

# Generate the plot
fig = px.scatter_3d(boston_df, x='ZN', y='RM', z='MEDV')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
fig.show()

# fig = px.scatter_3d(boston_df, x='ZN', y='RM', z='MEDV')
# fig.update_traces(marker=dict(size=5))
# fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
# fig.show()

# fig = px.scatter_3d(df, x='sepal_width', y='sepal_length', z='petal_width')
# fig.update_traces(marker=dict(size=5))
# fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
# fig.show()







boston_df.info()
boston_df.info(verbose=True)
boston_df.describe(include='all')
boston_df.loc[:, ["ZN", "RM", "MEDV"]].describe()



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