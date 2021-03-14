# Explore PCA in scikit-learn
# Is it available in other package?
# Creating 2D plot showing principal components with data
# Create biplot and discuss that the loadings on the principal components relate to the arrows (rape centered at 0.54, 0.17)
# Scree plot and cumulative PVE plot
# Maybe just use the iris dataset?  Would be easy for plotting
# Also include some an example of the statsmodels version https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html

# Use the pca package to plot biplots (2d and 3d)
# https://pypi.org/project/pca/
# https://stackoverflow.com/questions/39216897/plot-pca-loadings-and-loading-in-biplot-in-sklearn-like-rs-autoplot

# For an example of plotting a biplot via matplotlib exclusively...
# Good example: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

from sklearn.decomposition import PCA

# use fit_transform

# score method returns the average log-likelihood of each sample?

## Start Here

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Loading Iris
X, y = load_iris(return_X_y=True)

# Scaling features
X = StandardScaler().fit_transform(X)

# Initializing PCA estimator
# n_components=None by default, so all components will be returned
pca = PCA()

# Performing PCA
pca.fit_transform(X)  # projected X values, aka principal component scores
pca.components_  # each row represents a principal component, while each value is a loading
pca.explained_variance_ratio_  # proportion of variance explained per component

# Scree plot
(sns
 .lineplot(x=np.arange(1, 5),
           y=pca.explained_variance_ratio_,
           marker="o")
 .set(title="PVE vs. Principal Component",
      xlabel="Principal Component",
      ylabel="Proportion of Variance Explained",
      ylim=(-0.05, 1.05)))

# Cumulative PVE vs. Principal Component plot
(sns
 .lineplot(x=np.arange(1, 5),
           y=np.cumsum(pca.explained_variance_ratio_),
           marker="o")
 .set(title="Cumulative PVE vs. Principal Component",
      xlabel="Principal Component",
      ylabel="Cumulative Proportion of Variance Explained",
      ylim=(-0.05, 1.05)))



# Creating biplot and 3D biplot with pca package
import pca

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Loading iris as NumPy arrays
X, y = load_iris(return_X_y=True)

# Scaling features
X = StandardScaler().fit_transform(X)

# Creating DataFrame of X values
X_df = pd.DataFrame(X, columns=load_iris().feature_names)

# Creating DataFrame of y values
target_names = pd.DataFrame(data=dict(target=np.unique(y),
                                      target_names=load_iris().target_names))
y_df = pd.merge(pd.DataFrame(data=dict(target=y)), target_names)

# Initializing pca.pca estimator
model = pca.pca()

# Performing PCA
results = model.fit_transform(X_df)

# Plotting biplot
fig, ax = model.biplot(y=y_df["target_names"])




# Exploring Python Data Science Handbook Code Snippet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Setting plot theme
sns.set_theme()


# Defining function to draw principal component vectors
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops = dict(arrowstyle="->",
                      linewidth=2,
                      shrinkA=0,
                      shrinkB=0,
                      color="black")
    ax.annotate("", v1, v0, arrowprops=arrowprops)


# Generating data
rng = np.random.RandomState(123)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
pca = PCA(n_components=2)
pca.fit(X)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# Plotting data and principal component vectors
ax[0].scatter(X[:, 0], X[:, 1], alpha=0.3)
for length, vector in zip(pca.explained_variance_, pca.components_):
    draw_vector(pca.mean_, pca.mean_ + vector, ax=ax[0])
ax[0].axis("equal")
ax[0].set(xlabel="x",
          ylabel="y",
          title="Random 2D Data Points")

# Plotting point with max x value as red
max_x_index = pd.DataFrame(X)[0].idxmax()
ax[0].plot(X[max_x_index, 0], X[max_x_index, 1], "ro")

# Plotting project points onto first and second principal components
X_pca = pca.transform(X)
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax[1])
draw_vector([0, 0], [3, 0], ax=ax[1])
ax[1].axis("equal")
ax[1].set(xlabel="First Principal Component",
          ylabel="Second Principal Component",
          title="Principal Components",
          xlim=(-5, 5), ylim=(-3, 3.1))

# Plotting point with max x value as red (same point as above)
ax[1].plot(X_pca[max_x_index, 0], X_pca[max_x_index, 1], "ro")


# Exploring statsmodels


from statsmodels.multivariate.pca import PCA

# Initializing, standarize=False because X already standarized
results = PCA(X, standardize=False)

results.loadings  # principal components represented vertically (vs. horizontally in scikit-learn)
