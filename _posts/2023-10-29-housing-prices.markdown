---
layout: post
title:  "Kansas City Home Prices"

date:   2023-10-29 1:00:00 -0400
categories: blog
---

## Introducing the Problem and Dataset

In the realm of real estate and housing, numerous features of a home can
influence property prices. This project hopes to shed some light on which
features might be more influential than others. By using a comprehensive
dataset from Kaggle titled
[KC_House_Data](https://www.kaggle.com/datasets/astronautelvis/kc-house-data/),
we can build a linear regression model to accurately estimate home prices in
Kansas City. This dataset contains many details about each property that will
be used. I've listed all the features I found most useful below:
- `bedrooms` : Number of bedrooms

- `bathrooms` : Number of bathrooms

- `sqft_living` : Square footage of the home

- `sqft_lot` : Square footage of the lot

- `floors` : Number of floors

- `waterfront` : Waterfront view or not

- `view` : View from the home ranked 0-4

- `condition` : Condition of the home ranked 1-5

- `grade` : Grade or quality of the construction and design ranked 1-13

- `yr_built` : Year the home was built

- `price` : Selling price of the home


## What is Regression? How does it work?

Linear Regression is a machine learning technique used to model the
relationship between a dependent variable and independent variable(s) with a
linear equation. It essentially attempts to plot a line of best fit that
minimizes the differences between the predicted and actual values.

The line of best fit can be found using the following equation: `y = mx + b` \
Where `m` is the slope, `x` is the independent variables, and `b` is the
intercept.

## Experiment 1
# Data Understanding

To initially understand the data we are working with, I took a deeper dive into
the dataset. Using Python, I started with loading the data into a Pandas
dataframe called `raw_data` to get a closer look at the features and values.

I display the first 5 rows of the csv here, which gives an idea of the features
and values the dataset offers:
```
raw_data = pd.read_csv('kc_final.csv')
raw_data.head()
```
![head](/assets/p3/head.png)

Then, I printed out some statistical information from each feature:
```
raw_data.describe()
```
![head](/assets/p3/describe.png)

Next, I sought to find some patterns in the data by displaying a heatmap of the
correlation matrix (data variable introduced in next section):
```
correlation_matrix = data.corr()
```
<p align="center">
    <img src="/assets/p3/correlation_heatmap.png" width="700">
</p>

Looking at specific features, I created pairplots with columns that were
relevant, like the bedroom and bathroom count, square footage of living space
and lot, and the year the home was built (zoom in or open image in new tab to
view closer):
<p align="center">
<img src="/assets/p3/price_pairplots.png" width="700">
</p>

Lastly, I created some boxplots with the features that have smaller ranges of
values:
<p align="center">
    <img src="/assets/p3/floors_v_price.png" width="500">
    <img src="/assets/p3/waterfront_v_price.png" width="500">
    <img src="/assets/p3/view_v_price.png" width="500">
    <img src="/assets/p3/condition_v_price.png" width="500">
    <img src="/assets/p3/grade_v_price.png" width="500">
</p>

# Pre-processing

Before scrubbing the data, I filtered the dataset down to the columns that I
found most relevant:
```
important_features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'yr_built']
data = raw_data[important_features]
data
```

Then I checked if there are any nulls in the dataset:
```
null_counts = data.isna().sum()
null_counts_gtzero = null_counts[null_counts > 0]
```
Fortunately, the dataset came back without any null values.

# Modeling

For the modeling process, I used a linear regression model using sklearn's
tools. I started by dividing the set into `X` and `y`, with `X` including all
the features and `y` including just the target variable, price. The data is
then split up into training and test sets with an 80-20 split. Next, we
instantiate the linear regression model, and fit it to the training data from
the split. Finally, we use the model to generate predictions into the `pred`
variable. This all forms the foundation of the linear regression analysis.

```
X = data.drop(columns=['price'])
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

We can visualize these predicted values using a scatter plot, with the line of
best fit in red:
<p align="center">
    <img src="/assets/p3/actual_v_predicted.png" width="600">
</p>

# Evaluation

To evaluate the dataset, we can find the Root Mean Squared Error (RMSE) and
R-squared values using sklearn's metrics library. The RMSE value defines the
average difference between the model's predicted values and actual values,
while R-squared measures how well the model predicts the price, with 0 meaning
the model did not explain any of the variability and 1 meaning the model
perfectly predicts the price.
```
mse = mean_squared_error(y_test, pred, squared=True)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, pred)
```
As a result, our RMSE value comes back with $229,315.39. This is high, but it
is important to note that average cost of a home in this dataset is over
$500,000. So on average, the model was off the actual price by $229,315.39.

Additionally, the R-squared metric comes back to 0.65. The model is a
reasonably good fit to the data, but could use improvements. The remaining 0.35
is unexplained by the model, which could have been due to features not included
in the dataset. We will dive some improvements in the next 2 experiments.

## Experiment 2

For the second experiment, I changed two things regarding the features in the
datset. First, I added ALL the usable features in the dataset to use in the
model:
```
new_features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
exp2_data = raw_data[new_features]
```
Additionally, I attempted to apply Principal Component Analysis (PCA) to the
dataset to try and increase our accuracy. PCA reduces the number of features
but retains all the important ones (called principal components). Because PCA
is affected heavily by scale, I decided to scale the data first using sklearn's
StandardScaler:
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_exp2_scaled = scaler.fit_transform(X_train_exp2)
X_test_exp2_scaled = scaler.transform(X_test_exp2)
```
Then, I applied PCA:
```
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)
X_train_exp2_pca = pca.fit_transform(X_train_exp2_scaled)
X_test_exp2_pca = pca.transform(X_test_exp2_scaled)
```

Lastly, I fit the data to the model and evaluated the results.
```
model_exp2 = LinearRegression()
model.fit(X_train_exp2_pca, y_train_exp2)
pred_exp2 = model.predict(X_test_exp2_pca)

mse_exp2 = mean_squared_error(y_test_exp2, pred_exp2, squared=True)
rmse_exp2 = np.sqrt(mse_exp2)
r2_exp2 = r2_score(y_test_exp2, pred_exp2)
```

As a result of experiment two, our RMSE value lowers to $214,681.47, a
reduction in $15,000 per home. Additionally, our R-squared value increased to
0.70. This tells us that both using all the features and applying PCA gave us a
noticable improvement in accuracy!

## Experiment 3

For the third experiment, I decided to try different regression models, ending
with the DecisionTreeRegressor:
```
from sklearn.tree import DecisionTreeRegressor
model_exp3 = DecisionTreeRegressor()
model.fit(X_train_exp3, y_train_exp3)
pred_exp3 = model.predict(X_test_exp3)
```
Unfortunately, none of the other regression models improved our results much.
Lastly, I was able to change the test size to be 25% instead of 20%, which kept
the R-squared value at 0.70, but brought the RSME down again to $210,145.91.

## Impact

This project can be beneficial to visually see which features of a home might
influence the price the most. While specific to Kansas City, one could still
use the knowledge from this project to estimate their own home value.

Negatively, if someone has a feature that is important to their home value that
is not included in this dataset, this model will not estimate the home
correctly. Even something as small as paint color can drastically affect the
value of a home.

## Conclusion

While the model proved to be a fairly good fit of the data (roughly 70%
confident), there are definitely limitations. I found that the biggest
improvement to the model was in experiment two, where I added in a few extra
features and applied PCA. Using different regression models was not as
beneficial to the model as I would have hoped. In future experiments, I might
do more fine tuning to the model and it's features, and maybe try using
ensemble methods to combine several algorithms.

## References
[Kaggle Data Set](https://www.kaggle.com/datasets/astronautelvis/kc-house-data/)

## Code
You can view all the code for this project [here](/assets/p3/project_three_code.html).
