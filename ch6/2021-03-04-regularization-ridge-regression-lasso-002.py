# This blog has a great example of making the plot:
# https://towardsdatascience.com/ridge-regression-for-better-usage-2f19b3a202db
# show the comparison to statsmodels https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.fit_regularized.html

# Boston housing data

# Show either or in statsmodels, and comment that the other is just changing one value

# Plan:
# Load boston as X, y
# Standardize variables
# Run Ridge with single C
# Run a grid search with standardizing in a pipeline
# Fit the best model using _best_params and get MSE
# Plot the graph showing the coefficients for different values of C (will need to run a for loop)

# Repeat the above grid search with Lasso
# Fit the best model using _best_params and get MSE and copare to Ridge

# Show Lasso fitting with statsmodels and comment on how to do ridge regression

# Make sure to look at Ridge and Lasso documentation for defaults, etc

# Question: What are RidgeCV and LassoCV?

#########
# =====================================
#########

### Start here

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Loading boston housing data
X, y = load_boston(return_X_y=True)


standard_scaler = StandardScaler()
# X = standard_scaler.fit(X)  # needs changing?











