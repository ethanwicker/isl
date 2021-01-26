
**Notes below**

### Big header

#### Small header

$$
\begin{aligned} 
Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_pX_p + \epsilon 
\end{aligned}
$$

**Notes above**``

* Planning to do a theoretical post (or two) and a Python example post (or two)
* Comment the structure of this post was influence by ISL chapter 2


### 3.3 Other Considerations in the Regression Model

#### Qualitative Predictors

* So far, I have only discussed the Multiple Linear Regression model in relation to *quantitative predictors*.  However, it can also be extended to *qualitative predictors*.

Qualitative variable can sometimes be known as categorical or factor variables.

##### Qualitative Predictors Predicting with Only Two Levels

* When we only have two levels, it is simple to create an *indicator* or *dummy variable* that takes on two possible numeric values, 0 or 1
* See formulas 3.26 and 3.27, and the interpretation about beta_0, beta_1 immediately following.
* We can still interpret p-values the same for dummy variables - to determine significance
* Alternatively, we could also create a new dummy variable that takes on the values -1 and 1 (see formulas 3.26 and 3.27.)
* Import to note, no matter how we encode our qualitative predictor, the predictors will be equivalent, but the interpretation of the coefficients will change.

##### Qualitative Predictors with More than Two Levels

* Verbatim from book: "When a qualitative predictor has more than two levels, a single dummy variable cannot represent all possible values. In this situation, we can create additional dummy variables."
* See formulas 3.28, 3.29 and 3.30 and the below interpretation
* Verbatim from book: There will always be one fewer dummy variable than the number of levels. The level with no dummy variable—African American in this example—is known as the baseline.
* Interpretation of p-values is the same.  However, we can use the F-test to test H0 : β1 = β2 = 0, which does not depend on the coding.
* With the dummy variable approach, we can also incorporate both quantitative and qualitative predictors.  We simply fit a multiple regression model on the quantitative variables and the encoded qualitative variables.  Graphically, this is represented as two hyperplanes parallel to each other.
* Note, there are other methods of coding qualitative variables, beside the dummy variable approach here, but they all produce equivalent model fits.
* Related concept is 1-hot encoding 

#### Extensions of the Linear Model

* Standard linear model provides interpretable results, and is quite useful in on many real-world problems
* However, it makes several strict assumptions that are often not true in practice.
* Verbatim: Two of the most important assumptions state that the relationship between the predictors and response are additive and linear.
* Explain what the additive and liner assumptions due - will explore other methods to relax these two assumptions but here are two classical ways

##### Removing the Additive Assumption (interaction terms)

*interaction* terms
* formulas 3.31 and 3.32, and the interpretation below
* A change in X_2 will change the impact of X_1 on Y --> the relationship between X_1 and Y is no longer linear
* We interpret the p-value of the interaction term just like any other p-value for a coefficient
* Here a significant p-value indicates indicates the true relationship is non-linear
* In the event the interaction term is significant but the main effects are not, we should still include the main effects via the *hierarchy principle*
* Verbatim from book: The hierarchical principle states that if we include an interaction in a model, we should also include the main effects, even if the p-values associated with their coefficients are not significant.
* The concept of interaction also applies to qualitative variables, or a combination of quantitative and qualitative variables
* Interaction between a quantitative and qualitative variable has a nice interpretation --> instead of two ``parallel hyperplanes, we get two intersecting hyperplanes.

##### Non-linear Relationships

* Here, a simple way to extend the linear model to accommodate non-linear relationships is explored --> polynomial regression.  In future posts, I'll explore more sophisticated approaches "for performing non-linear fits in more general settings".
* Verbatim: A simple approach for incorporating non-linear associations in a linear model is to include transformed versions of the predictors in the model.
* Write formula 3.36 in general terms
* Comment: This is still a linear model --> it is just a multiple linear regression model where X_2 = X_1^2

** >>> Going to write up the above, and make include the below, or just might publish the in 2 posts <<< **


#### Potential Problems     <-- maybe new blog post here

* When fitting a linear regression model to a particular dataset, many problems can occur.  
* Below, I'll briefly discuss some of the most common issues:
  (Put below in italics)
1. Non-linearity of the response-predictor relationships. 
2. Correlation of error terms.
3. Non-constant variance of error terms.
4. Outliers.
5. High-leverage points.
6. Collinearity.

##### Non-linearity of the Data

##### Correlation of Error Terms

##### Non-constant Variance of Error Terms

##### Outliers

##### High Leverage Points

##### Collinearity


#### Notes for post 5 and 6
Post 5: Python example
Maybe sample my data down to 10,000 rows to make it easier for plotting

Want to include: 
* Qualitiative predictors & dummy encoding of some sort
* Removing the additive assumption: interaction terms
* Removing the linear assumption: non-linear relationships

* A comparison of scikit-learn vs. statsmodels

* Discuss these 6 problems (maybe not all, but some)
1. Non-linearity of the response-predictor relationships.  <<-- residual plot
2. Correlation of error terms.
3. Non-constant variance of error terms.
4. Outliers.                <<-- maybe studentized residuals
5. High-leverage points.    <<-- leverage statistics 
6. Collinearity.            <<-- VIF
