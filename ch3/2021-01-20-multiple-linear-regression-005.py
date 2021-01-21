# Post 5: Python example
# Maybe sample my data down to 10,000 rows to make it easier for plotting
#
# Want to include:
# * Qualitative predictors & dummy encoding of some sort
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
import pandas as pd

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


# Reading nyc data
nyc = pd.read_csv("~/python/isl/data/nyc.csv")

# Getting count of flights by carriers
nyc.groupby("carrier_name").agg(count_flights=('timestamp_hour', 'count')).sort_values(by="count_flights", ascending=False)

# Filter for carriers that have at least 10,000 flights in dataset
nyc_high_vol_carriers = \
    nyc\
    .groupby("carrier_name")\
    .filter(lambda x: x["carrier_name"].count() >= 10000)

nyc_per_hour_per_carrier = \
    nyc_high_vol_carriers \
    .groupby(["timestamp_hour", "carrier_name"]) \
    .agg(mean_departure_delay=('departure_delay', 'mean'),
         mean_wind_speed_mph=('wind_speed_mph', 'mean'),
         mean_precipitation_inches=('precipitation_inches', 'mean'),
         mean_visibility_miles=('visibility_miles', 'mean')) \
    .reset_index()

nyc_per_hour_per_orig_airport = \
    nyc_high_vol_carriers \
    .groupby(["timestamp_hour", "orig_airport"]) \
    .agg(mean_departure_delay=('departure_delay', 'mean'),
         mean_wind_speed_mph=('wind_speed_mph', 'mean'),
         mean_precipitation_inches=('precipitation_inches', 'mean'),
         mean_visibility_miles=('visibility_miles', 'mean')) \
    .reset_index()

# Using nyc_per_hour_per_orig_airport
# Can I predict mean departure delay as a function of weather and which airport
# Then use an interaction term on precipitation inches and the airports

import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols(formula="mean_departure_delay ~ mean_wind_speed_mph + mean_precipitation_inches + mean_visibility_miles + orig_airport", data=nyc_per_hour_per_orig_airport)
result = model.fit()
result.summary()

# Explain what the : vs * is in the formula

model = smf.ols(formula="mean_departure_delay ~ mean_precipitation_inches + orig_airport", data=nyc_per_hour_per_orig_airport)
model = smf.ols(formula="mean_departure_delay ~ mean_precipitation_inches*carrier_name", data=nyc_per_hour_per_carrier)
result = model.fit()
result.summary()





## Working on boston data below, easier to work with





from sklearn import datasets
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf


boston = datasets.load_boston()

X_df = pd.DataFrame(boston.data, columns=boston.feature_names)
y_df = pd.DataFrame(boston.target, columns=["MEDV"])

boston_df = pd.concat([X_df, y_df], axis=1)

#sklearn.linear_model
model = linear_model.LinearRegression()
model.fit(X=boston_df[["CHAS", "RM"]], y=boston_df["MEDV"])
model.score(X, y)

test = boston_df[["CHAS", "RM"]]
sm.add_constant(test)

#sm.OLS
# will need to add intercept
model = sm.OLS(endog=boston_df["MEDV"], exog=sm.add_constant(boston_df[["CHAS", "RM"]]))
result = model.fit()
result.summary()

#smf.ols
model = smf.ols(formula="MEDV ~ CHAS + RM", data=boston_df)
result = model.fit()
result.summary()

#smf.ols with interaction
model = smf.ols(formula="MEDV ~ CHAS + RM + CHAS:RM", data=boston_df)
# model = smf.ols(formula="MEDV ~ CHAS*RM", data=boston_df)  # equivalent
result = model.fit()
result.summary()

# Trying to find a good interaction term
model = smf.ols(formula="MEDV ~ CHAS + RM + ZN + DIS + CHAS:DIS + ZN:DIS", data=boston_df)    ## <<-- works
# model = smf.ols(formula="MEDV ~ CHAS*RM", data=boston_df)  # equivalent
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