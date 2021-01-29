# Notes for blog post on this: https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62

# Use the titanic dataset
# Transform some categorical variables for use in a ridge regression model
#  • OneHotEncoding with sparse=dense
#  • Impute some missing values if they're there using SimpleImputer
# Transform some numeric variables for use in a ridge regression model
#  • Impute some missing values using (e.g. "age") with the median via SimpleImputer
#  • Standardize the numeric values using StandardScaler
# Perform a 10-fold cross validation similar to the blog post, including a
# Grid search to search over the values of C and ["mean", "median"]
