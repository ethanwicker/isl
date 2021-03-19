# Good resource: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html

import pandas as pd
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Loading data as NumPy arrays
X, y = load_diabetes(return_X_y=True)

# Initializing estimator
standard_scaler = StandardScaler()

# Scaling predictors
X = standard_scaler.fit_transform(X)

# Initializing estimator
pca = PCA()

# Performing PCA
pc_scores = pca.fit_transform(X)

# Initializing estimator
lin_reg = LinearRegression()

# neg_mean_squared_error instead of just mean_squared_error because cross_val_score tries to maximize scores.
# But we want a low MSE, or a high negative MSE

# Creating lists to hold results
components_used = []
mean_squared_errors = []

# Performing 10-fold cross validation on sequential amount of principal components
for i in range(1, 11):

    # 10-fold cross-validation
    cv_scores = cross_val_score(estimator=lin_reg,
                                X=pc_scores[:, 0:i],
                                y=y,
                                cv=10,
                                scoring="neg_mean_squared_error")

    # Calculating average of negative mean squared error, and turning positive
    cv_mean_squared_error = cv_scores.mean() * -1

    # Appending results
    components_used.append(i)
    mean_squared_errors.append(cv_mean_squared_error)

# Organizing cross-validation results into DataFrame
mse_by_n_components = \
    pd.DataFrame(data=dict(components_used=components_used,
                           mean_squared_errors=mean_squared_errors))

# Plotting
sns.lineplot(x="components_used",
             y="mean_squared_errors",
             data=mse_by_n_components)
