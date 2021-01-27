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
# Good resource for plotting: https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import statsmodels.formula.api as smf
from yellowbrick.regressor import ResidualsPlot   # don't use can delete


# Load data
boston = datasets.load_boston()

# Printing description
print(boston.DESCR)

# Creating DataFrame instead of X and y numpy.ndarrays for ease of use with patsy
X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
y_df = pd.DataFrame(boston.target, columns=["MEDV"])
boston_df = pd.concat([X_df, y_df], axis=1)

# Creating crime_label field
boston_df = \
    (boston_df
     .assign(
        crime_label=pd.cut(boston_df["CRIM"],
                           bins=3,
                           labels=["low_crime", "medium_crime", "high_crime"]))
    )

# Converting crime_label field to NumPy array
crime_labels_ndarray = boston_df["crime_label"].to_numpy().reshape(-1, 1)

# Defining encoder
encoder = OneHotEncoder()

# Fitting encoder on array, and transforming
crime_labels_encoded = encoder.fit_transform(crime_labels_ndarray)

# Converting encoded array to DataFrame
crime_labels_df = pd.DataFrame(data=crime_labels_encoded.toarray(),
                               columns=encoder.get_feature_names())

# Concatenating with boston_df
boston_df = pd.concat(objs=[boston_df, crime_labels_df], axis=1)

# Could also use this to drop the first column
# Necessary when fitting unregularized linear models, so as to not create linear dependencies
encoder = OneHotEncoder(drop="first")


# Using smf.ols first for ease of exploration + inference
# Going to train on MEDV as a function of ZN, CHAS, RM, DIS
# We learn that DIS is not significant, so we should drop it
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM + DIS", data=boston_df)
result = model.fit()
result.summary()

# For post
model = smf.ols(formula="MEDV ~ ZN + CHAS + RM", data=boston_df)
result = model.fit()
result.summary()

# Looking at residual plot of above model
y = boston_df["MEDV"]
y_hat = result.predict()
resid = y - y_hat

sns.scatterplot(x=y_hat, y=resid).set(xlabel="Predicted Value",
                                      ylabel="Residual")

# Creating plot of residuals vs predictors to diagnosis any error term correlation

sns.regplot(x='value', y='wage', data=df_melt, ax=axs[0])
sns.regplot(x='value', y='wage', data=df_melt, ax=axs[1])

fig, axs = plt.subplots(nrows=2)
sns.scatterplot(x=boston_df["ZN"], y=resid, ax=axs[0]).set(ylabel="Residual")
sns.scatterplot(x=boston_df["RM"], y=resid, ax=axs[1]).set(ylabel="Residual")



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
# model2 = ResidualsPlot(linear_model.LinearRegression())
# model2.fit(X=X_poly, y=boston_df["MEDV"])
# model2.show()

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


sm.graphics.influence_plot(result, size=6)

model = smf.ols("MEDV ~ ZN + I(ZN**2) + RM + I(RM**2) + ZN*RM + I(ZN*RM**2)", data=boston_df)
result = model.fit()
studentized_resids = result.outlier_test()
studentized_resids = studentized_resids.assign(student_resid_abs=abs(studentized_resids["student_resid"]))
outliers = studentized_resids.sort_values(by="student_resid_abs", ascending=False).head(6)

#df = df[df.index.isin(df1.index)]

boston_df_no_outliers = boston_df[~boston_df.index.isin(outliers.index)]

model_no_outliers = smf.ols("MEDV ~ ZN + ZN**2 + RM + I(RM**2) + ZN*RM + I(ZN*RM**2)", data=boston_df_no_outliers)
result_no_outliers = model.fit()
result_no_outliers.summary()

from statsmodels.stats.outliers_influence import variance_inflation_factor

vifs = [variance_inflation_factor(exog=X_poly, exog_idx=i) for i in range(X_poly.shape[1])]
predictors = ["ZN", "ZN^2", "RM", "RM^2", "ZN*RM", "(ZN*RM)^2"]
pd.DataFrame(dict(predictor=predictors, vif=vifs))

X = boston_df[["ZN", "RM"]].to_numpy()

vifs = [variance_inflation_factor(exog=X, exog_idx=i) for i in range(X.shape[1])]
predictors = ["ZN", "RM"]
pd.DataFrame(dict(predictor=predictors, vif=vifs))

sm.add_constant(X_poly)

y, X = d
matrices('annual_inc ~' + features, df, return_type='dataframe')
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

sm.stats.var

# Trying Breusch-Pagan test
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(results.resid, results.model.exog)
lzip(name, test)

import statsmodels.stats.api as sms
from statsmodels.compat import lzip

returned_names = ['Lagrange multiplier statistic', 'p-value',
                  'f-value', 'f p-value']

bp_test_results = sms.het_breuschpagan(resid, X_poly)
lzip(returned_names, bp_test_results)


sms.lzip(name, test)

pd.compat.lzip(name, test)



y = boston_df["MEDV"]
y_hat = model.predict(X=X_poly)
resid = y - y_hat

sns.scatterplot(x=y_hat, y=resid).set(xlabel="Predicted Value",
                                      ylabel="Residual")


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

## Below works
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Define mash size for prediction surface
mesh_size = .02

# Train model
model = smf.ols(formula="MEDV ~ ZN + RM + ZN:RM", data=boston_df)
result = model.fit()

# Define x and y ranges
# Note: y here refers to y-dimension, not response variable
x_min, x_max = boston_df["ZN"].min(), boston_df["ZN"].max()
y_min, y_max = boston_df["RM"].min(), boston_df["RM"].max()
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Get predictions for all values of x and y ranges
pred = model.predict(params=result.params, exog=np.c_[np.ones(shape=xx.ravel().shape), xx.ravel(), yy.ravel(), xx.ravel()*yy.ravel()])

# Reshape predictions to match mesh shape
pred = pred.reshape(xx.shape)

# Plotting
fig = px.scatter_3d(boston_df, x='ZN', y='RM', z='MEDV')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='predicted_MEDV'))
fig.show()

## Above works



## THIS MIGHT HAVE ACTUALY WORKED


## BELOW
# Trying out plotly
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

mesh_size = .02
margin = 0


poly = PolynomialFeatures()
X_poly = poly.fit_transform(X=boston_df[["ZN", "RM"]])   # include_bias=True is the default here
model = linear_model.LinearRegression()
model.fit(X=X_poly, y=boston_df["MEDV"])
model.score(X=X_poly, y=boston_df["MEDV"])


x_min, x_max = boston_df.ZN.min() - margin, boston_df.ZN.max() + margin
y_min, y_max = boston_df.RM.min() - margin, boston_df.RM.max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

pred = model.predict(X=np.c_[xx.ravel(), xx.ravel()**2, yy.ravel(), yy.ravel()**2, xx.ravel()*yy.ravel(), xx.ravel()*yy.ravel()**2])

pred = pred.reshape(xx.shape) # and this


fig = px.scatter_3d(boston_df, x='ZN', y='RM', z='MEDV')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
fig.show()

## ABOVE


# Condition the model on sepal width and length, predict the petal width
# model = SVR(C=1.)
# model.fit(X, y)



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