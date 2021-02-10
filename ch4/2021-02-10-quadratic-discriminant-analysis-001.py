# Ethan Wicker

import janitor
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

titanic = pd.read_csv("data/titanic-train-kaggle.csv")

titanic = (janitor
          .clean_names(titanic)
          .loc[:, ["sex", "age", "fare", "pclass", "survived"]]
          .rename(columns={"pclass": "ticket_class"}))

titanic = titanic.query("age.notna()").reset_index()


X = titanic[["age", "fare"]]
y = titanic["survived"]

qda = QuadraticDiscriminantAnalysis()
qda.fit(X, y)
qda.score(X, y)

print(classification_report(y_true=y, y_pred=qda.predict(X)))

lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda.score(X, y)

print(classification_report(y_true=y, y_pred=lda.predict(X)))

roc_qda = plot_roc_curve(estimator=qda, X=X, y=y)
roc_log_reg = plot_roc_curve(estimator=lda, X=X, y=y, ax=roc_qda.ax_)
plt.title("ROC Curve Comparison")
plt.show()



