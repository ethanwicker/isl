# Weather data downloaded from R "weather" (https://www.rdocumentation.org/packages/nycflights13/versions/1.0.1/topics/weather)
# flights data from nycflights13 dataset (https://pypi.org/project/nycflights13/)


# Lets predict if the flight was delayed using
# Quantitative: distance (in miles), visib (visibility in mile), precip (precipitation in inches), wind_speed (in mph)
# Qualitative: carrier (maybe online the top ones, can join with airlines to get names), origin( the origin airport)

# I should look at the interaction between wind_speed, precipitation and visibilitiy, and also maybe the interaction with carrier as well


# Importing Libraries ---------------------------------------------------------

import numpy as np
import pandas as pd
import seaborn as sns
from nycflights13 import airlines, flights, weather
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Organizing airlines ---------------------------------------------------------

# Renaming fields for clarity
airlines = airlines.rename(columns={'carrier':'carrier_abbreviation',
                                    'name': 'carrier_name'})


# Organizing flights ----------------------------------------------------------

# Renaming fields for clarity
flights = flights.rename(columns={'dep_time': 'departure_time',
                                  'sched_dep_time': 'scheduled_departure_time',
                                  'dep_delay': 'departure_delay',
                                  'arr_time': 'arrival_time',
                                  'sched_arr_time': 'scheduled_arrival_time',
                                  'arr_delay': 'arrival_delay',
                                  'carrier': 'carrier_abbreviation',
                                  'origin': 'orig_airport',
                                  'dest': 'dest_airport',
                                  'distance': 'distance_miles',
                                  'time_hour': 'timestamp_hour'})

# Assigning timestamp_hour field to datetime type
flights = flights.assign(timestamp_hour=pd.to_datetime(flights.timestamp_hour, utc=True))


# Organizing weather ----------------------------------------------------------

# Renaming fields for clarity
weather = weather.rename(columns={'origin': 'orig_airport',
                                  'wind_speed': 'wind_speed_mph',
                                  'precip': 'precipitation_inches',
                                  'visib': 'visibility_miles',
                                  'time_hour': 'timestamp_hour'})

# Assigning timestamp_hour field to datetime type
weather = weather.assign(timestamp_hour=pd.to_datetime(weather.timestamp_hour, utc=True))


# Joining Data ----------------------------------------------------------------

# Joining
nyc = \
    flights \
    .merge(right=airlines, how='left', on='carrier_abbreviation') \
    .merge(right=weather, how='left', on=['orig_airport', 'timestamp_hour'])

# Selecting relevant fields and reordering
# NOTE: Likely don't need all of these fields, can probably remove some
nyc = nyc[['timestamp_hour',
           'orig_airport',
           'dest_airport',
           'carrier_name',
           'carrier_abbreviation',
           'scheduled_departure_time',
           'departure_time',
           'departure_delay',
           'scheduled_arrival_time',
           'arrival_time',
           'arrival_delay',
           'air_time',
           'distance_miles',
           'wind_speed_mph',
           'precipitation_inches',
           'visibility_miles']]

## EDA: Plotting Data ----------------------------------------------------------

# Getting count of flights per day per origin
nyc_flight_counts = \
    nyc \
    .assign(date=nyc.timestamp_hour.dt.floor(freq='d')) \
    .groupby(['date', 'orig_airport'], as_index=False) \
    .agg(count=('orig_airport', 'count'))

# Plotting
sns.lineplot(data=nyc_flight_counts, x='date', y='count', hue='orig_airport')

# NOTE: Should do some additional plotting here, depending on what regressions I do

# Cleaning Data Prior to Regression -------------------------------------------

# Removing NaN values
# In a true analysis, should investigate this.  In this toy analysis, can simply remove
nyc = nyc.dropna(subset=['departure_delay',
                         'distance_miles',
                         'wind_speed_mph',
                         'precipitation_inches',
                         'visibility_miles'])


# Quantitative: distance (in miles), visib (visibility in mile), precip (precipitation in inches), wind_speed (in mph)
# Qualitative: carrier (maybe online the top ones, can join with airlines to get names), origin( the origin airport)


# Multivariable Linear Regression via scikit-learn ----------------------------

# Question: What do I want to predict?
# Options: Does the flight leave delayed? Does the flight arrive delayed? Is the flight time longer than it should be?
# Focus on departure delay as the response, and visibility_miles, precipitation_inches and wind_speed_mpg as the quantitation variables
# With carrier and orig_airport as the qualitative variables



linear_model = LinearRegression()
X = np.array(nyc.wind_speed_mpg).reshape(-1, 1)
y = np.array(nyc.departure_delay).reshape(-1, 1)

linear_model.fit(X=X, y=y)

linear_model.predict(X=X)
linear_model.coef_
linear_model.intercept_

linear_model.score(X, y)


# Refitting with multiple variables
linear_model = LinearRegression()
X = nyc[['wind_speed_mph', 'precipitation_inches', 'visibility_miles']]
y = np.array(nyc.departure_delay).reshape(-1, 1)

linear_model.fit(X=X, y=y)
linear_model.coef_
linear_model.intercept_



# Multivariable Linear Regression via statsmodels ----------------------------

# Fitting with sm.OLS
X = nyc[['wind_speed_mph', 'precipitation_inches', 'visibility_miles']]
X = sm.add_constant(X)


mod = sm.OLS(endog=nyc.departure_delay, exog=X)
result = mod.fit()
result.summary()      # This produces pretty good results, low R-squared but also low p-values
# indicating the variables have an affect on the response, but the model has poor predictive performance


# Fitting with smf.ols
# Using patsy and R style formulas
mod = smf.ols('departure_delay ~ wind_speed_mph + precipitation_inches + visibility_miles', data = nyc)  # exact same as sm.OLS
result = mod.fit()
result.summary()

# From SOF
res_ols = sm.OLS(y, X).fit()
res_ols.params


sm.OLS



# Exploring Multivariable linaer regression via scikit-learn ------------------
# Do I need to make delay field?
# arrival_delay is if the flight is delayed
# if I want to look at whether the flight takes long that is should, need to look at arrival_time-departure_time
# and compare to the scheduled times
# Use scikit-learn pipelines!



# Some NA value in arrival_delay
# this is a much better way of checking
nyc.query('arrival_delay.isna()')

# get not na values in arrival_delay
# Removing NaN arrival_delay values
nyc = nyc.query('arrival_delay.notna()')



linear_model = LinearRegression()
X = np.array(nyc.distance_miles).reshape(-1, 1)
y = np.array(nyc.arrival_delay).reshape(-1, 1)

# GETTING error here: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
linear_model.fit(X=X, y=y)

linear_model.predict(X=X)
linear_model.coef_
linear_model.rank_
linear_model.singular_
linear_model.intercept_

linear_model.get_params()
linear_model.score(X, y)

# blog post #1: multivariable linear regresssion with scikit-learn
# blog post #2: the exact same regresssion with statsmodels, and dive into the regression summary table

# maybe even switch the order since it makes more since to do it with statsmodels first and find a good model, then
# do it in scikit-learn

# Tuesday: do the Regression again with multiple variables in both scikit-learn and statsmodels
# How to make 3d plot with plane?



# Cannot get regression summary from scikit-learn, but statsmodels provides this:
#from statsmodels.formula.api  import ols
##you need a Pandas dataframe df with columns labeled Y, X, & X2
#est = ols(formula = 'Y ~  X + X2', data = df).fit()
#est.summary()

#
# nyc.distance_miles.isnull().values.any()
# nyc.arrival_delay.isnull().values.any()
# nyc.isnull()
#
# nyc.isnull().values.any()  # indicates that values are missing
#
# sns.scatterplot(data=nyc, x='distance_miles', y='arrival_delay')
#
# X = nyc['distance_miles']
# nyc.distance_miles.to_numpy.shape
#
#
# nyc.distance_miles.to_numpy.array.reshape(-1, 1)
# nyc.array.reshape(1,-1)
#
# nyc.distance_miles.to_numpy
#
# nyc.distance_miles..
#
# np.array(nyc.distance_miles).reshape(-1, 1)
#
# pd.DataFrame(nyc.distance_miles).shape
# nyc.distance_miles.values.reshape
#
#
#
# nyc.hist()
#
# X = pd.DataFrame(df[‘OAT (F)’])
# y = pd.DataFrame(df[‘Power (kW)’])
# model = LinearRegression()
# scores = []
# kfold = KFold(n_splits=3, shuffle=True, random_state=42)
# for i, (train, test) in enumerate(kfold.split(X, y)):
#  model.fit(X.iloc[train,:], y.iloc[train,:])
#  score = model.score(X.iloc[test,:], y.iloc[test,:])
#  scores.append(score)
# print(scores)
#
# lin_reg = LinearRegression()
# # GETTING error here: ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
# lin_reg.fit(X=nyc.distance_miles.shape, y=nyc.arrival_delay.shape)
#
# lin_reg.fit(X=nyc[['distance_miles']], y=nyc[['departure_delay']])
#
# nyc.distance_miles.values.reshape()
# nyc.distance_miles.values.reshape(length, 1)
# 
#

## Next steps:
# Join datasets together via carrier_abbreviation, origin DONE
# Perform multivariable linear reg on just two quantative variables and first and make a 3D scatter plot
# link up with github
# figure out how to write markdown file in pycharm with LaTex
# Look at note on first three blog posts




# ##############
#
# nyc.plot(x='timestamp_hour', y='distance_miles')
#
# nyc[[0:9], [:]]
#
# nyc[0:9]
#
#
#
# sns.lineplot(data=nyc[0:9], x='timestamp_hour', y='distance_miles')
#
# nyc.date = nyc.timestamp_hour.dt.floor(freq='d')
#
# nyc.assign(date=nyc.timestamp_hour.dt.floor('d'))
#
#
# a = nyc.value_counts(subset=['timestamp_hour', 'origin'])
#
# sns.load_dataset(data=a, x='timestamp_ho')
#
# sns.load_dataset('flights')
# sns.load_dataset("flights")
#
#
#
#
# agg(min_height=('height', 'min')
#
#
# nyc.origin.unique()
#
# pd.plot()
#
# nyc
#
#
