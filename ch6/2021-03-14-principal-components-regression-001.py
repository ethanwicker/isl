# Good resource: http://www.science.smith.edu/~jcrouser/SDS293/labs/lab11-py.html

import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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
X_projected = pca.fit_transform(X)

lin_reg = LinearRegression()







