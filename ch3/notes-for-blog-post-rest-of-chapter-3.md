* Planning to do a theoretical post (or two) and a Python example post (or two)
* Comment the structure of this post was influence by ISL chapter 2


### 3.3 Other Considerations in the Regression Model

#### Qualitative Predictors

* So far, I have only discussed the Multiple Linear Regression model in relation to *quantitative predictors*.  However, it can also be extended to *qualitative predictors*.

Qualitative variable can sometimes be known as categorical or factor variables.

##### Predicting with Only Two Levels

* When we only have two levels, it is simple to create an *indicator* or *dummy variable* that takes on two possible numeric values, 0 or 1
* See formulas 3.26 and 3.27, and the interpretation about beta_0, beta_1 immediately following.
* We can still interpret p-values the same for dummy variables - to determine significance
* Alternatively, we could also create a new dummy variable that takes on the values -1 and 1 (see formulas 3.26 and 3.27.)
* Import to note, no matter how we encode our qualitative predictor, the predictors will be equivalent, but the interpretation of the coefficients will change.

##### Qualitative Predictors with More than Two Levels

* Verbatim from book: "When a qualitative predictor has more than two levels, a single dummy variable cannot represent all possible values. In this situation, we can create additional dummy variables."
* See formulas 3.28, 3.29 and 3.30 and the below interpretation
* Verbatim from book: There will always be one fewer dummy vari- able than the number of levels. The level with no dummy variable—African American in this example—is known as the baseline.
* Interpretation of p-values is the same.  However, we can use the F-test to test H0 : β1 = β2 = 0, which does not depend on the coding.
* With the dummy variable approach, we can also incorporate both quantitative and qualitative predictors.  We simply fit a multiple regression model on the quantitative variables and the encoded qualitative variables.  Graphically, this is represented as two hyperplanes parallel to each other.
* Note, there are other methods of coding qualitative variables, beside the dummy variable approach here, but they all produce equivalent model fits.

#### Extensions of the Linear Model

##### Removing the Additive Assumption (interaction terms)

##### Non-linear Relationships

#### Potential Problems

##### Non-linearity of the Data

##### Correlation of Error Terms

##### Non-constant Variance of Error Terms

##### Outliers

##### High Leverage Points

##### Collinearity