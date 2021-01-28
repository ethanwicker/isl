# Ethan Wicker

# Python for Logistic Regression #2 Post

# Things to Include:
# • 1 predictor logistic regression in scikit-learn and statsmodels
# • A plot showing the results
# • Comment on maximum likelihood estimation
# • Multiple Logistic Regression in scikit-learn and statsmodels, using qualitative variables
# • Multiclass Logistic Regression (hopefully using both scikit-learn and statsmodels)

# Plan
# • Use age + sex + fare to predict survived
# • Then use age + sex + survived to predict ticket_class (multiclass logistic  regression)


import pandas as pd
import janitor   # just for fun

titanic = pd.read_csv("~/python/isl/data/titanic-train-kaggle.csv")

titanic.info()
titanic.head()

titanic = (janitor
               .clean_names(titanic)
               .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
               .rename(columns={"pclass": "ticket_class"}))


