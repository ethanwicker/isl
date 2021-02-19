# Note: Do a quadratic logistic regression in this post

# Note for future: I think I like the idea of making myself a notes section with useful code snippets
# Maybe even call it that: "useful code snippets" Nah but notes if more general

# Plan:
# Do a simple validation set
# Then a LOOCV example
# Then a 10-fold CV example and compare the below models
# Using a toy dataset, train a LDA, QDA, logistic regression and quartic logistic regression model

# train_test_split helper https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split

# Discuss cross_val_score
# When the cv argument is an integer, cross_val_score uses the KFold or StratifiedKFold strategies by default, (think StratifiedKFold here if doing classifier)
# talk about pipelines and using to prevent data leak
# the cross_validate function
# KFold
# LeaveOneOut
# StratifiedKFold to keep same percentage of class labels as in original dataset
# GroupKFold
# TimeSeriesSplit
# mention many more can be found here: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv

#####
# Start below
#####

# • Explain what cross validation is
# • In a previous post, I introduced much of the theory behind CV
# • In this post, I'll explore some working example using CV from scikit-learn
# • In particular, I'll explore simple validation sets using train_test_split, `KFold` CV, `LeaveOneOut`, `StratifiedKFold` CV,
# • GroupKFold   # will just discuss this, won't actually do it
# • TimeSeriesSplit    # will just discuss this, won't actually do it
# • mention many more can be found here: https://scikit-learn.org/stable/modules/cross_validation.html#group-cv

## Note on scaling:
# For logistic regression (non-regularized), LDA, and QDA, I shouldn't need to scale my observations
# But going to anyway just because it typically doesn't hurt
# And compare results afterwards to see if equal or not
# Also all of Iris in the same units (cm)
##

####
# Getting Iris Dataset (classification task)
####

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris

# X, y = load_iris(return_X_y=True)

# Loading Iris as pandas DataFrame
iris = load_iris(as_frame=True)

# Creating target_names DataFrame to match up target with target_name
iris_target_names = pd.DataFrame(data=dict(target=[0, 1, 2], target_name=iris.target_names))

# Merging
data = (iris["frame"]
        .merge(right=iris_target_names,
               on="target",
               how="left"))

####
# Plot Iris as a scatter plot and show all the classes with different colors
####

import seaborn as sns

sns.scatterplot(x="sepal length (cm)",
                y="sepal width (cm)",
                hue="target_name",
                data=data)

####
# Try LDA by itself, don't scale, and use train_test_split
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

lda.predict(X_train)  # training predictions
lda.predict(X_test)   # training predictions
lda.score(X=X_train, y=y_train)  # training error rate
lda.score(X=X_test, y=y_test)    # test error rate

####
# KFold/really Stratified KFold
# But also do some by hand for practice and such, and future notes to myself
# Create a pipeline to scale data and then train log reg, quadratic log reg, LDA, QDA with KFold/really Stratified KFold
# Logistic regression here is going to be a one-vs-rest multi-class logistic regression classifier

# Have a section "Comments on Repeated Cross-Validation"
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

from sklearn.model_selection import cross_val_score

# Comment: Above had to use _train and _test sets, but with k-Fold can use entirety of X and y

lda = LinearDiscriminantAnalysis()

# 5-fold k-fold cross-validation by default, doing 10-fold below
scores = cross_val_score(lda, X, y, cv=10)   # Performing stratified k-Fold CV here --> keeping class labels roughly same in each fold
scores # view the scores
scores.mean()
scores.std()

# We can also repeat the above stratified k-fold procedure multiple times
# Look at above notes. Will show this code because it's interesting, but not useful in practice.
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=1234)
scores = cross_val_score(lda, X, y, cv=cv)
scores
scores.mean()
scores.std()

mean_cv_scores = pd.DataFrame(data=scores.reshape(10, 5)).mean()
pd.DataFrame(data=dict(mean_cv_score=mean_cv_scores,
                       model="Linear Discriminant Analysis",
                       repeat=["Repeat " + str(i) for i in range(1, 6)]))


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

# I want pipeline steps for
# LDA
# QDA
# Logistic Regression            --> set the necessary argument to "none"
# Quadratic Logistic Regression  --> in this one use Polynomial Features step & set the necessary argument to "none"

# step_simple_imputer = ("simple_imputer", SimpleImputer(strategy="constant"))
# step_encoder = ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))
#
# # Creating pipeline
# pipeline = Pipeline([step_simple_imputer, step_encoder])


from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Creating individual pipelines
# Not scaling, just for demonstration purposes of code
# I SHOULD JUST STANDARDIZE EVERYTHING
pipeline_lda = ("LDA", (Pipeline([("lda", LinearDiscriminantAnalysis())])))
pipeline_qda = ("QDA", Pipeline([("qda", QuadraticDiscriminantAnalysis())]))
pipeline_log_reg = ("Logistic Regression", Pipeline([("log_reg", LogisticRegression(penalty="none"))]))
pipeline_quadratic_log_reg = \
    ("Quadratic Logistic Regression", Pipeline([
        # Standizing here so LDFGS solver works
        ("standard_scaler", StandardScaler()),
        ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=False)),
        ("logistic_regression", LogisticRegression(penalty="none"))
    ]))

# Constructing list of pipelines
pipelines = [
    pipeline_lda,
    pipeline_qda,
    pipeline_log_reg,
    # Might just take out the quad log reg here
    pipeline_quadratic_log_reg
]

# Initializing DataFrame for results
results = pd.DataFrame()

# Looping through pipelines, storing results
for pipe, model in pipelines:

    # Getting cross validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv)

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
 .set(title='Model Comparison',
      xlabel="Model",
      ylabel="Mean Cross-Validation Score \n (Repeated Stratified k-Fold)"))


















poly = PolynomialFeatures(degree=2, interaction_only=False)
X_poly = poly.fit_transform(X_train)
log_reg = LogisticRegression(penalty="none")
log_reg.fit(X_poly, y_train)
log_reg.score(poly.transform(X_test), y_test)

# This works
from sklearn.preprocessing import StandardScaler
pipe = Pipeline([
    ("standard_scaler", StandardScaler()),
    ("polynomial_features", PolynomialFeatures()),
    ("logistic_regression", LogisticRegression(penalty="none"))
])
#
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)

cross_val_score(pipe, X, y, cv=cv)
#
#
# lr = LogisticRegression()
# lr.fit(X_poly, y_train)
#
# lr.score(poly.transform(X_test), y_test)

# pipe = Pipeline([('polynomial_features', poly), ('logistic_regression', lr)])
# pipe.fit(X_train, y_train)
# pipe.score(X_test, y_test)



# this is what i want
# It's a list of pipelines that is then iterated through
# From here: https://www.kaggle.com/rakesh2711/multiple-models-using-pipeline
pipelines = []
pipelines.append(('scaledLR', (Pipeline([('scaled', StandardScaler()), ('LR', LogisticRegression())]))))
pipelines.append(('scaledKNN', (Pipeline([('scaled', StandardScaler()), ('KNN', KNeighborsClassifier())]))))
pipelines.append(('scaledDT', (Pipeline([('scaled', StandardScaler()), ('DT', DecisionTreeClassifier())]))))
pipelines.append(('scaledSVC', (Pipeline([('scaled', StandardScaler()), ('SVC', SVC())]))))
pipelines.append(('scaledMNB', (Pipeline([('scaled', StandardScaler()), ('MNB', GaussianNB())]))))

model_name = []
results = []
for pipe, model in pipelines:
    kfold = KFold(n_splits=10, random_state=42)
    crossv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(crossv_results)
    model_name.append(pipe)
    msg = "%s: %f (%f)" % (model_name, crossv_results.mean(), crossv_results.std())
    print(msg)

# Compare different Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(model_name)
plt.show()


# Comparing with
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# Something like this what I want
# From: https://stackoverflow.com/questions/51629153/more-than-one-estimator-in-gridsearchcvsklearn

# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.datasets import load_iris
# iris_data = load_iris()
# X, y = iris_data.data, iris_data.target
#
#
# # Just initialize the pipeline with any estimator you like
# pipe = Pipeline(steps=[('estimator', SVC())])
#
# # Add a dict of estimator and estimator related parameters in this list
# params_grid = [{
#                 'estimator':[SVC()],
#                 'estimator__C': [1, 10, 100, 1000],
#                 'estimator__gamma': [0.001, 0.0001],
#                 },
#                 {
#                 'estimator': [DecisionTreeClassifier()],
#                 'estimator__max_depth': [1,2,3,4,5],
#                 'estimator__max_features': [None, "auto", "sqrt", "log2"],
#                 },
#                # {'estimator':[Any_other_estimator_you_want],
#                #  'estimator__valid_param_of_your_estimator':[valid_values]
#
#               ]
#
# grid = GridSearchCV(pipe, params_grid)

####
# Same as above with LeaveOneOut
####

####
# Discuss GroupKFold, TimeSeriesSplit, the above link
####




