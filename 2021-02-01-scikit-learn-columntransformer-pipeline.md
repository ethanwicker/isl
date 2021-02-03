---
layout: post
title: "scikit-learn: `ColumnTransformer` and `Pipeline`"
subtitle: Summarizing an Effective Workflow with pandas
comments: false
---

I recently read through this [excellent Medium article](https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62) about the `ColumnTransformer` class in scikit-learn and how it can be used in tandem with `Pipeline`s and the `OneHotEncoder` class.  To strengthen my own understanding of the concept, I decided to follow the post with my own working example, and summarize the concepts along the way.

### Introduction

In scikit-learn's 0.20 release, the `ColumnTransformer` estimator was released.  This estimator allows different transformers to be applied to different fields of the data in parallel, before concatenating them together.  In particular, this estimator is attractive when processing a pandas DataFrame with categorical fields.

For this working example, I'll be using the same slimmed down Titanic dataset from my previous [logistic regression post](https://ethanwicker.com/2021-01-27-logistic-regression-002/).

Below, I'll make use of the `ColumnTransformer` estimator to encode two categorical fields via scikit-learn's `OneHotEncoder`.  Because scikit-learn machine learning models require their input to be two-dimensional numerical arrays, an encoding preprocessing step is required.

I'll also use `SimpleImputer` estimator to preform some preprocessing of numerical fields.  Lastly, I'll perform a ridge regression and wrap all of these steps into a reusable and convenient `Pipeline`.

For example purposes further along in this post, I'll take the last 10 rows of `titanic` as a test dataset, and allocate the remaining rows as my training data.

```python
titanic_train = titanic.iloc[:-10]
titanic_test = titanic.iloc[-10:]
```

### OneHotEncoder

Let's first encode `sex` to demonstrate some functionality of `OneHotEncoder`.

```python
from sklearn.preprocessing import OneHotEncoder

# Initializing one-hot encoder
# Forcing dense matrix to be returned
encoder = OneHotEncoder(sparse=False)

# Encoding categorical field
X_categorical = titanic_train[["sex"]]
X_categorical_encoded = encoder.fit_transform(X_categorical)

>>> X_categorical_encoded
array([[0., 1.],
       [1., 0.],
       [1., 0.],
       ...,
       [0., 1.],
       [0., 1.],
       [1., 0.]])
```

From the output, we can see that the `male` and `female` values of `sex` have been encoded into two binary columns.

Notice that a NumPy array was returned.  We can access column names indicating which feature of `sex` is represented by each column using the `get_feature_names()` method.

```python
>>> feature_names = encoder.get_feature_names()
>>> feature_names
array(['x0_female', 'x0_male'], dtype=object)
```

We can also use the `inverse_transform()` method to return the original categorical label from the `sex` column.  Notice the brackets around `X_categorical[0]` that force a list to returned instead of a NumPy array.

```python
# Inverse transforming the first row of X_categorical_encoded
>>> encoder.inverse_transform([X_categorical_encoded[0]])
array([['male']], dtype=object)

# Inverse transforming all rows of X_categorical_encoded
>>> encoder.inverse_transform(X_categorical_encoded)
array([['male'],
       ['female'],
       ['female'],
       ...,
       ['male'],
       ['male'],
       ['female']], dtype=object)

# Verifying arrays are equivalent after inverse transforming X_categorical_encoded
>>> np.array_equal(encoder.inverse_transform(X_categorical_encoded), 
                   X_categorical)
True
```

### Applying Transformations to Training & Test Sets

Whenever we transform a training field, we must also transform the corresponding field in the test set.  We must do this *after* splitting up the training and test datasets, instead of performing the transformation first and then splitting the data.  If we used the latter method here, some information from our test set would leak over into our training set.  This mistake is something referred to as *data leakage*.

```python
# Encoding the sex column of our test dataset
>>> X_categorical_test = titanic_test[["sex"]].copy()

>>> encoder.transform(X_categorical_test)
array([[1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

The above transformation works great.  Because we have already initialized our one-hot encoder, we do not need to do so again.  

Although the above example worked well, occasionally we'll run into problems when transforming our test set.

#### Problem #1: ValueError: Found unknown categories

Occasionally, a value will be present in our test set that is not present in our training set.  This can present a problem, as our initialized one-hot encoder is expecting the same unique label value as the training set.

Let's suppose the first value of `X_categorical_test` was misspelled `fmale` instead of `female`.

```python
>>> X_categorical_test.iloc[0, 0] = "fmale"

>>> X_categorical_test.head()
        sex
704   fmale
705    male
706  female
707    male
708    male
```

If we attempt the transformation on this new `DataFrame`, we'll get an error indicating an unknown category was found.

```python
>>> encoder.transform(X_categorical_test)
ValueError: Found unknown categories ['fmale'] in column 0 during transform
```

In practice, we should investigate this issue further.  However, for this example, let's initialize a new `OneHotEncoder` with the `handle_unknown` parameter set to `"ignore"`, and fit it on the training observations again. Then, when we attempt to transform the test observations, the unknown values will be encoded as a row of all 0's.

```python
# Initializing new encoder
>>> encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

# Fitting on training observations
>>> encoder.fit(X_categorical)

# Transforming test observations
# Notice first row is all 0's
>>> encoder.transform(X_categorical_test)
array([[0., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

#### Problem #2: Missing Values

Handling missing values in our test set is similar to handling unknown values in our test set.  If we initialize our encoder with `handle_unknown="ignore"`, these missing observations will be gracefully handled and encoded as rows of all 0's.

Note, it appears the 0.24.1 release of scikit-learn included an upgrade to `OneHotEncoder` allowing it to handle `None` values, but not `NaN` values.

## NOTE: I downgraded back to 0.23 here, make sure this still works

```python
# Assigning None to some elements of X_categorical_test
>>> X_categorical_test.iloc[1, 0] = None
>>> X_categorical_test.iloc[2, 0] = None
>>> X_categorical_test.head()

# Encoding
# Notice first three rows are all 0's
>>> encoder.transform(X_categorical_test)
array([[0., 0.],
       [0., 0.],
       [0., 0.],
       [0., 1.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [1., 0.],
       [0., 1.],
       [0., 1.]])
```

#### Imputing Missing Values

In the event where we do have missing data, it can be useful to *impute* the missing values.  We can use scikit-learn's `SimpleImputer` transformer for this from the `impute` module.

Let's artificially create an `NaN` value in `X_categorical_copy` below, and impute this missing values.  We can use the `strategy` parameter to control how the imputation is done.  For numerical data, `strategy` can be set to either `mean` or `median`.   For categorical data, we can set `strategy` to `constant`, which will allow us to set a constant string value to convert missing values to.  We can also let `strategy=most_frequent` for both numerical and categorical observations, which will replace missing values with the most frequent observation in that column.

When `strategy` is equal to `"constant"`, we can optionally use the `fill_value` parameter to create the constant string value.  Below we'll just use the default value of `missing_value`.

```python
from sklearn.impute import SimpleImputer

# Assigning first element as NaN
X_categorical_copy = X_categorical.copy()
X_categorical_copy.iloc[0, 0] = np.nan

# Initializing SimpleImputer
# fill_value="missing_value" by default
simple_imputer = SimpleImputer(strategy="constant")

# Fitting and transforming
X_categorical_copy_imputer = simple_imputer.fit_transform(X_categorical_copy)
X_categorical_copy_imputer
```

Now, we can use the `fit_transform()` method as before for encoding.

```python
>>> encoder.fit_transform(X_categorical_copy_imputer)
array([[0., 0., 1.],
       [1., 0., 0.],
       [1., 0., 0.],
       ...,
       [0., 1., 0.],
       [0., 1., 0.],
       [1., 0., 0.]])
```

### scikit-learn `Pipeline`s

Instead of manually applying multiple fitting and transformation steps, we can instead use a `Pipeline`.  A `Pipeline` allows a list of transformations to be successively run, and a model can be trained as the last estimator.  `Pipeline`s are especially useful for reproducibe workflows, such as applying the same transformation to training and test sets or different subsets of a dataset during cross validation.

Each step in the `Pipeline` consist of a two-item tuple.  The first element of the tuple is a string that labels the step, and the second element is an initialized estimator.  The output of each previous step will be the input to the next step.

```python
from sklearn.pipeline import Pipeline

# Creating steps
step_simple_imputer = ("simple_imputer", SimpleImputer(strategy="constant"))
step_encoder = ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))

# Creating pipeline
pipeline = Pipeline([step_simple_imputer, step_encoder])

# Fitting and transforming
pipeline.fit_transform(X_categorical_copy)
```

After fitting the `Pipeline` on the training data, it is easy to transform the test data.  Notice that because the pipeline has already been fit, we do not need to refit it below, and can instead just call `transform()`. 

```python
pipeline.transform(X_categorical_test)
```

#### Transforming Multiple Categorical Columns

It is simple as well to use our `Pipeline` on multiple categorical columns.  Just refit the `Pipeline` and run the transformation.

```python
X_categorical = titanic_train[["sex", "ticket_class"]]
pipeline.fit_transform(X_categorical)
```

#### Accessing Steps in our Pipeline

We can also access the individual steps of our `Pipeline` via the `named_steps` attribute.  For example, we can access the feature names via `get_feature_names()` method after specifying `encoder`.

```python
encoder = pipeline.named_steps["encoder"]
encoder.get_feature_names()
```

### scikit-learn's `ColumnTransformer`

The `ColumnTransformer` estimator from the `compose` module allows the user to control which columns get which transformation.  This is especially useful when we consider the different transformations categorical and numerical fields will need.

The `ColumnTransformer` takes a three-item tuple of the following structure:

```python
("name_of_column_transformer", "SomeTransformer(parameters), columns_to_transform")
```

The `columns_to_transform` could be a list of column names or integer indexs, a boolean array, or a function that resolves to a selection of columns.

#### Passing a `Pipeline` to a `ColumnTransformer`

We can also pass a `Pipeline` to the `SomeTransformer(parameters)` input above.  Notice the below `Pipeline` is equivalent to the one described above, but `_categorical` has been appended throughout.  We'll have a numerical equivalent version shortly.

```python
from sklearn.compose import ColumnTransformer

# Creating steps
step_simple_imputer_categorical = ("simple_imputer", SimpleImputer(strategy="constant"))
step_encoder_categorical = ("encoder", OneHotEncoder(sparse=False, handle_unknown="ignore"))

# Creating pipeline
pipeline_categorical = Pipeline([step_simple_imputer_categorical, step_encoder_categorical])

# Defining categorical columns
columns_categorical = ["sex", "ticket_class"]

# Defining column transformer
column_transformer_categorical = [("column_transformer_categorical",
                                   pipeline_categorical,
                                   columns_categorical)]

# Creating column transformer
column_transformer = ColumnTransformer(transformers=column_transformer_categorical)
```

Because our `ColumnTransformer` selects columns, we can pass the entire `titanic_train` `DataFrame` to it.  The defined columns will be select and transformed as appropriate.  We could of course pass `titanic_test` to our `ColumnTransformer` as well.

```python
column_transformer.fit_transform(titanic_train)
array([[0., 1., 0., 0., 1.],
       [1., 0., 1., 0., 0.],
       [1., 0., 0., 0., 1.],
       ...,
       [0., 1., 0., 0., 1.],
       [0., 1., 0., 0., 1.],
       [1., 0., 1., 0., 0.]])
```

To get the feature names of our encoded variables as we have done before, we need to use the `named_transformers_` attribute of our `ColumnTransformer`.  After doing so, we can then `named_steps` attribute of our pipeline as before.

```python
(column_transformer
    .named_transformers_["column_transformer_categorical"]
    .named_steps["encoder"]
    .get_feature_names())
```

#### Transforming Numeric Columns




