# Good resource: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html

import pandas as pd

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
X_scores = pca.fit_transform(X)

pca.components_

lin_reg = LinearRegression()

lin_reg.fit(X=pca.fit_transform(X), y=y)

# neg_mean_squared_error instead of just mean_squared_error because cross_val_score tries to maximize scores.
# But we want a low MSE, or a high negative MSE
cross_val_score(lin_reg, X_scores[:, 0:2], y, cv=10, scoring="neg_mean_squared_error")

components_used = []
mean_squared_errors = []

for i in range(1, 11):

    cv_scores = cross_val_score(estimator=lin_reg,
                                X=X_scores[:, 0:i],
                                y=y,
                                cv=10,
                                scoring="neg_mean_squared_error")

    cv_mean_squared_error = cv_scores.mean() * -1

    components_used.append(i)
    mean_squared_errors.append(cv_mean_squared_error)

pd.DataFrame(data=dict(components_used=components_used,
                       mean_squared_errors=mean_squared_errors))
