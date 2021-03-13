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

X, y = load_iris(return_X_y=True)

# NOTE: I NEED TO SCALE MY DATA ****


pca = PCA()

# n_components=None by default, so all components returned
pca.fit_transform(X)  # projected X values (I think)
pca.components_       # principal components
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

np.cumsum(pca.explained_variance_ratio_)



# Exploring Python Data Science Handbook Code Snippet

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# plot data
ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
ax[0].axis('equal');
ax[0].set(xlabel='x', ylabel='y', title='input')

# plot principal components
X_pca = pca.transform(X)
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax[1])
draw_vector([0, 0], [3, 0], ax=ax[1])
ax[1].axis('equal')
ax[1].set(xlabel='component 1', ylabel='component 2',
          title='principal components',
          xlim=(-5, 5), ylim=(-3, 3.1))

fig.savefig('figures/05.09-PCA-rotation.png')


