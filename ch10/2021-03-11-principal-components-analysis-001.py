# Explore PCA in scikit-learn
# Is it available in other package?
# Creating 2D plot showing principal components with data
# Create biplot and discuss that the loadings on the principal components relate to the arrows (rape centered at 0.54, 0.17)
# Scree plot and cumulative PVE plot
# Maybe just use the iris dataset?  Would be easy for plotting
# Also include some an example of the statsmodels version https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html

# Good example: https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html

from sklearn.decomposition import PCA

# use fit_transform

# score method returns the average log-likelihood of each sample?