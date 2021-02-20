#####
# Start below
#####

# • Explain what cross validation is
# • In a previous post, I introduced much of the theory behind CV
# • In this post, I'll explore some working example using CV from scikit-learn
# • In particular, I'll explore simple validation sets using train_test_split, `KFold` CV, `LeaveOneOut`, `StratifiedKFold` CV,
# • GroupKFold         # will just discuss this, won't actually do it
# • TimeSeriesSplit    # will just discuss this, won't actually do it

####
# Getting Iris Dataset (classification task)
####

import pandas as pd
from sklearn.datasets import load_iris

# Loading Iris as pandas DataFrame
iris = load_iris(as_frame=True)

# Creating DataFrame to match up target label with target_name
iris_target_names = pd.DataFrame(data=dict(target=[0, 1, 2],
                                           target_name=iris.target_names))

# Merging predictor and response DataFrames, via left join
data = (iris["frame"]
        .merge(right=iris_target_names,
               on="target",
               how="left"))

####
# Plotting Iris as a scatter plot and show all the classes with different colors
####

import seaborn as sns

sns.scatterplot(x="sepal length (cm)",
                y="sepal width (cm)",
                hue="target_name",
                data=data)

####
# train_test_split and LDA
####

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

# Loading Iris again, returning X and y as NumPy arrays
X, y = load_iris(return_X_y=True)

# Simple validation set split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.25,
                                                    random_state=1234,
                                                    shuffle=True)

# Initializing and fitting on training observations
lda = LinearDiscriminantAnalysis()
lda.fit(X=X_train, y=y_train)

lda.predict(X_train)  # predictions on training set
lda.predict(X_test)   # predictions on test set
lda.score(X=X_train, y=y_train)  # correct classification rate on training set
lda.score(X=X_test, y=y_test)    # correct classification rate on test set

####
# KFold/really Stratified KFold
####

from sklearn.model_selection import cross_val_score

# 5-fold k-fold cross-validation by default, doing 10-fold below
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default
# StratifiedKFold is used if estimator is a classifier
# Performing stratified k-Fold CV here --> keeping class labels roughly same in each fold
# Comment: Above had to use _train and _test sets, but with k-Fold can use entirety of X and y
scores = cross_val_score(estimator=lda, X=X, y=y, cv=10)
scores  # view the scores
scores.mean()
scores.std()

# Using StratifiedKFold class explicitly
from sklearn.model_selection import StratifiedKFold

cv_stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=False)
cross_val_score(estimator=lda, X=X, y=y, cv=cv_stratified_k_fold)

# cross_validate

from sklearn.model_selection import cross_validate

scores = cross_validate(estimator=lda,
                        X=X,
                        y=y,
                        cv=10,
                        scoring=("roc_auc_ovo", "accuracy"),
                        return_train_score=True)

from pprint import pprint
pprint(scores)


####
# "Comments on Repeated Cross-Validation"
# Paper: https://limo.libis.be/primo-explore/fulldisplay?docid=LIRIAS1655861&context=L&vid=Lirias&search_scope=Lirias&tab=default_tab&lang=en_US&fromSitemap=1

# This paper arguments that repeated k-fold CV does not do much good
# We are interested in sigma_2, which is the predictive accuracy on a fixed sample S taken from a Population P.  We want to know how well our model will perform basically, only using our sampel S
# However, this sigma_2 value has both bias (because S is a subset, S_2 will be different) and
# it has high variance because of the nature of taking k random folds.  Doing the CV again will give a different value of sigma_2
# The paper argues that repeated k-fold CV can reduce this variance but not the bias
# repeated k-fold CV is useful to accurately estimate u_k, the mean of all CV results across all possible ways to split
# But u_k is not necessary an accurate estimate of sigma_2.  We're interested in sigma_2, and we still get a biased estimate because S itself is a random sample from the population P
# Repeated CV is at the best a waste of computation resources, and at the worse misleading
####

# We can also repeat the above stratified k-fold procedure multiple times
# Look at above notes. Will show this code because it's interesting, but not useful in practice.
# Wanted to give the code a try just to figure out
from sklearn.model_selection import RepeatedStratifiedKFold
cv_repeated_stratified_k_fold = RepeatedStratifiedKFold(n_splits=10,
                                                        n_repeats=5,
                                                        random_state=1234)
scores = cross_val_score(lda, X, y, cv=cv)
scores
scores.mean()
scores.std()  # This standard residual lower than the above value of 0.0427

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Creating individual pipelines for each model
# With LDA, QDA and non-regularized logistic regression,
# Scaling doesn't affect results
# But doing so here so the LDFGS solver in the below quadratic logistic regression works
# logistic regression (non-regularized), LDA, and QDA
# Comment: Logistic regression models here are is one-vs-rest multi-class logistic regression classifier
pipeline_lda = \
    ("LDA", (Pipeline([
        ("standard_scaler", StandardScaler()),
        ("lda", LinearDiscriminantAnalysis())
    ])))

pipeline_qda = \
    ("QDA", Pipeline([
        ("standard_scaler", StandardScaler()),
        ("qda", QuadraticDiscriminantAnalysis())
    ]))

pipeline_log_reg = \
    ("Logistic Regression", Pipeline([
        ("standard_scaler", StandardScaler()),
        ("log_reg", LogisticRegression(penalty="none"))
    ]))

pipeline_quadratic_log_reg = \
    ("Quadratic Logistic Regression", Pipeline([
        ("standard_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=False)),
        ("logistic_regression", LogisticRegression(penalty="none"))
    ]))

# Constructing list of pipelines
pipelines = [
    pipeline_lda,
    pipeline_qda,
    pipeline_log_reg,
    pipeline_quadratic_log_reg
]

# Initializing DataFrame for results
results = pd.DataFrame()

# Looping through pipelines, storing results
for pipe, model in pipelines:

    # Getting cross validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv_repeated_stratified_k_fold)

    # Calculating mean CV scores (approximation for mu_1 in paper's notation)
    cv_scores_mean = pd.DataFrame(data=cv_scores.reshape(10, 5)).mean()

    # Organizing results into DataFrame
    results_per_model = pd.DataFrame(data=dict(mean_cv_score=cv_scores_mean,
                                               model=pipe,
                                               repeat=["Repeat " + str(i) for i in range(1, 6)]))

    # Concatenating results
    results = pd.concat([results, results_per_model])

# Plotting results as boxplot, just for demonstration purposes
(sns
 .boxplot(data=results, x="model", y="mean_cv_score")
 .set(title='Model Comparison via \n Repeated Stratified k-Fold Cross Validation',
      xlabel="Model",
      ylabel="Mean Cross-Validation Score"))


####
# Same as above with LeaveOneOut
####

from sklearn.model_selection import LeaveOneOut
cv = LeaveOneOut()
scores = cross_val_score(lda, X, y, cv=cv)
scores.mean()

####
# Discuss GroupKFold, TimeSeriesSplit, the above link
# Link: https://scikit-learn.org/stable/modules/cross_validation.html
####




