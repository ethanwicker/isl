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
# Also use RidgeCV which defaults to LOOCV (can set cv=10 to perform 10-fold CV)
# Show the comparison to statsmodels

# Repeat the above grid search with Lasso
# Fit the best model using _best_params and get MSE and compare to Ridge

# Show Lasso fitting with statsmodels and comment on how to do ridge regression

# Make sure to look at Ridge and Lasso documentation for defaults, etc

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

# Initializing Ridge estimator, defining lambda (alpha) to be 0.1
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
# GridSearchCV (with Ridge)
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

# =====================================
# RidgeCV
# =====================================

# Using RidgeCV to perform same function as GridSearchCV (since only searching over alphas in GridSearchCV
from sklearn.linear_model import RidgeCV

# Initializing estimator
# RidgeCV defaults to leave-one-out-cross-validation, setting cv=5 for 5-fold cross-validation
ridge_reg_cv = RidgeCV(alphas=10.0 ** np.arange(-5, 6), cv=5)

# Fitting
ridge_reg_cv.fit(X, y)

# Results
ridge_reg_cv.alpha_       # best alpha=100
ridge_reg_cv.best_score_  # highest mean 5-fold cross-validated test score (corresponds where alpha=100)
ridge_reg_cv.coef_        # standardized coefficients when alpha=100
ridge_reg_cv.intercept_   # intercept when alpha=100


# =====================================
# Plotting Standardized Coefficients as Function of lambda
# =====================================

import seaborn as sns

# Defining alpha (lambda) values (powers of 10, from 0.00001 to 1000000)
alphas = np.arange(-5, 6)

# Initiating lit to hold results
coefficients_list = []

# For each value of alpha, performing ridge regression and storing coefficients
for alpha in 10.0 ** alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X, y)
    coefficients_df = pd.DataFrame(ridge_reg.coef_).T
    coefficients_list.append(coefficients_df)

# Organizing coefficients DataFrame for plotting
coefficients = \
    (pd.concat(coefficients_list)
     .rename(columns=pd.Series(load_boston().feature_names))
     .assign(lambda_value=10.0 ** alphas)
     .set_index("lambda_value")
     .unstack()
     .reset_index()
     .rename(columns={"level_0": "field",
                      0: "standardized_coefficients"})
     .loc[:, ["lambda_value", "field", "standardized_coefficients"]])

# Viewing DataFrame
coefficients

# Plotting standardized coefficients as function of lambda
sns.lineplot(data=coefficients,
             x="lambda_value",
             y="standardized_coefficients",
             hue="field")


# =====================================
# GridSearchCV with Lasso
# =====================================

# Initializing Lasso and GridSearchCV estimators
lasso = Lasso()
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid)

# Fitting grid search object
grid_search_lasso.fit(X, y)

# Results
grid_search_lasso.best_params_     # best alpha=0.1
grid_search_lasso.best_score_      # highest mean 5-fold cross-validated test score (corresponds where alpha=0.1)
grid_search_lasso.score(X, y)      # training score of best model, refit on all folds

# =====================================
# Lasso with statsmodels' OLS.fit_regularized
# =====================================










