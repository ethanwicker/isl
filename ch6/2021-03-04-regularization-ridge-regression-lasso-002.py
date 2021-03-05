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
# Instead of plotting the graph as in the above medium link, I can probably use cross_val_score or a grid search param to get the stored coef values for each alpha, (RidgeCV with array of alphas)
# Also use RidgeCV which defaults to LOOCV (can set cv=10 to perform 10-fold CV)

# Repeat the above grid search with Lasso
# Fit the best model using _best_params and get MSE and copare to Ridge

# Show Lasso fitting with statsmodels and comment on how to do ridge regression

# Make sure to look at Ridge and Lasso documentation for defaults, etc

# Question: What are RidgeCV and LassoCV?

# =====================================
# =====================================
# =====================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =====================================
# Loading data
# =====================================

# Loading boston housing data
X, y = load_boston(return_X_y=True)

# =====================================
# Scaling features
# =====================================

# Scaling input features
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

# =====================================
# Ridge
# =====================================

# Initialzing Ridge estimator, defining lambda (alpha) to be 0.1
ridge_reg = Ridge(alpha=0.1)

# Fitting estimator
ridge_reg.fit(X, y)

# Viewing results
ridge_reg.score(X, y)  # training R^2 square
ridge_reg.coef_        # coefficients
ridge_reg.intercept_   # intercept
ridge_reg.predict(X)   # getting predictions

# =====================================
# k-Fold Cross-Validation
# =====================================

# Using cross_val_score (evaluating model on mean and std of R^2 values)
from sklearn.model_selection import cross_val_score

# Initializing cross_val_score estimator, 5-fold cross-validation
cv_scores = cross_val_score(estimator=ridge_reg, X=X, y=y, cv=5)

# Mean and standard deviation of training scores across folds
cv_scores.mean()
cv_scores.std()

# =====================================
# RidgeCV
# =====================================

# Question: Is RidgeCV with an array of alphas and cv=5 performing nested cross-validation under the hood?  Not immediately clear.



# =====================================
# GridSearchCV
# =====================================

from sklearn.model_selection import GridSearchCV

# Defining grid of candidate alpha values (powers of 10, from 0.00001 to 1000000)
param_grid = {"alpha": 10.0 ** np.arange(-5, 6)}

# Initializing Ridge and GridSearchCV estimators
ridge_reg = Ridge()
grid_search = GridSearchCV(estimator=ridge_reg, param_grid=param_grid)

# Fitting grid search object
grid_search.fit(X, y)

# Results
grid_search.best_params_     # best alpha=100
grid_search.best_estimator_  # best estimator object
grid_search.best_score_      # highest mean 5-fold cross-validated test score (corresponds where alpha=100)

grid_search.predict(X)       # predictions using best model, refit on all folds
grid_search.score(X, y)      # training score of best model, refit on all folds

# Detailed results as pandas DataFrame
grid_search_results = pd.DataFrame(grid_search.cv_results_)
grid_search_results









