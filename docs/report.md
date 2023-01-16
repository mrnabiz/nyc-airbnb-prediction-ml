# Zero to Hero Machine Learning Pipeline: London Airbnb Open Data

### Table of Content
- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Data Cleaning](#data-cleaning)
  - [Data Dictionary](#data-dictionary)
- [EDA](#eda)
  - [Null Values](#null-values)
  - [Numeric Columns](#numeric-columns)
  - [Correlation Matrix and Correlation Plot](#correlation-matrix-and-correlation-plot)
  - [Target Column](#target-column)
- [Feature Engineering](#feature-engineering)
  - [Date Column: last_review, and host_since](#date-column-last_review-and-host_since)
  - [Name Column](#name-column)
- [Preprocessing and Transformations](#preprocessing-and-transformations)
  - [Categorical and Binary Features](#categorical-and-binary-features)
  - [Ordinal Features](#ordinal-features)
  - [Numerical Features](#numerical-features)
  - [Text Features](#text-features)
  - [Drop Feature](#drop-features)
- [Baseline and Linear Models](#baseline-and-linear-models)
- [Non-Linear Models](#non-linear-models)
  - [LGBM Regressor](#lightgbm-regressor)
  - [XGB Regressor](#xgb-regressor)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Interpretation and Feature Importance](#interpretation-and-feature-importance)
  - [Permutation Feature Importance](#permutation-feature-importance)
  - [SHAP](#shap)
- [Deployment Performance](#deployment-performance)
- [Conclusion](#conclusion)

## Introduction
**"He was no one, Zero, Zero
Now he's a honcho, He's a hero, hero!"**

Isn't this familiar? Absolutely it is. No way you could forger about Disney's cool animation, Hercules.


<p>
  <img src="https://static.wikia.nocookie.net/disney/images/f/fa/Zero_to_Hero.png" alt="Hercules" width="50%"/>
</p>

Fig 1: Charming Hercules ([Source](https://disney.fandom.com/wiki/Zero_to_Hero))

"Zero to Hero" is a phrase that is often used to describe a transformation from a novice or beginner to an expert or accomplished individual. It can also refer to a person who starts with very little and through hard work and determination, becomes successful. Throughout this journey, I want to take you to a step-by-step guide on building a regression Machine Learning pipeline.

This mini-project is about building a regression pipeline to predict the number of `reviews per month` as a proxy for the popularity of the listing, trained based on a dataset collected on September 2022 in London.

## Dataset
Airbnb is an online marketplace that connects people who need a place to stay with people who have a spare room or an entire home to share. The platform allows property owners to rent out their properties to travelers, who can book the properties through the Airbnb website or mobile app. 
The dataset to build this pipeline was captured from [Detailed Airbnb Listing Data (London, Sep 2022)](https://www.kaggle.com/datasets/mrnabiz/detailed-airbnb-listing-data-london-sep-2022) on Kaggle. The original data was prepared by [Inside Airbnb project](http://insideairbnb.com/). The mission of Inside Airbnb is to empower residential communities with data and information that enables them to understand, make decisions and have control over the effects of Airbnb's presence in their neighborhoods.

Some of the data that is typically included in the Inside Airbnb dataset includes property information (like the property's location, price, number of bedrooms and bathrooms, and amenities) and review information (like the date of the review, the rating given by the guest, and the text of the review).
The Inside Airbnb dataset can be used for various purposes like market analysis, demand forecasting, price optimization, and much more.

After retrieving the dataset with the [`pull_data.py`](https://github.com/mrnabiz/zero-to-hero-ml-pipline/blob/main/src/data_wrangling/pull_data.py) script by running [Step 1 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline), it is the time to clean the data.

### Data Cleaning
In this step, two scripts [`clean_data.py`](https://github.com/mrnabiz/zero-to-hero-ml-pipline/blob/main/src/data_wrangling/clean_data.py) and [`preprocessing.py`](https://github.com/mrnabiz/zero-to-hero-ml-pipline/blob/main/src/data_wrangling/preprocessing.py) for data cleaning and data preprocessing purposes by running [Step 2 and 3 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline). The major cleaning task are:
- Removing irrelevant or ID-based columns like `listing_url`, `neighborhood_overview`, `host_id`, `host_url`, `host_name`, and etc.
- Dropping the null values
- Transferring `last_review` and `host_since` date columns into datetime object
- Removing the `$` and `,` from the `price` column and changing its type to float
- Removing the `[]` and `"` characters from `amenities` column to prepare it for countvectorizer.

### Data Dictionary
This raw dataset includes 69351 entries and has 52 columns. After the data cleaning and data wrangling step we end of having 51726 observations and 23 columns. Some of the most important columns are:
- `host_since`: The date when he host joined AirBnb
- `room_type`: The type of the rental property, Entire home or Private room
- `neighbourhood_cleansed`: The name of the neighbourhood
- `minimum_nights`: The minimum required nights for booking
- `maximum_nights`: Maximum allowed nights for booking
- `minimum_nights_avg_ntm`: Average minimum nights booked
- `maximum_nights_avg_ntm`: Average maximum nights booked
- `host_listing_count`: The total number of active listing for the host
- `number_of_reviews`: Total number of current reviews of the property
- `last_review`: The date of the last review the property received
- `reviews_per_month` (target) and `last_review` show some null values related to the properties with zero reviews. So the null values of both of these features will be dropped since we know that for `reviews_per_month` equal to zero, we observe the null values in the aforementioned feature.

## EDA
You can reproduce the results of this EDA by running [Step 4 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).
### Null Values
First thing first! At the very first step of the EDA we need to identify the null value and their potential threat. Null values (also known as missing values) can sometimes cause problems like introducing bias into the model and skewing the distribution of the data. So handling the missing values correctly can improve the performance and accuracy of the model. The plot plot shows the occurrence of the null values in sample (10% of the total size) of the data set.
<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/missing_values_plot.png" width="85%"/>
</p>

Fig 2: Missing values occurrence

We see an interesting pattern here that the review-related data are missing all together in the dataset which is not bad at all. In this case we are lucky and dropping the null values would be a better option here.

### Numeric Columns
As this stage we initially plot histograms of the numeric columns in the dataset. Since histograms can help to visualize the distribution of the data, they can reveal important insights about the data such as outliers, skewness, multimodality, etc.
Understanding the distribution of the data also can help to choose the appropriate machine learning algorithm. For example, a histogram showing a normal distribution might suggest using a linear regression model, while a histogram showing a skewed distribution might suggest using a decision tree algorithm. The grid plot below shows the histograms of important numeric features.
<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/numeric_hist_plot.png" width="85%"/>
</p>
Fig 3: Numeric columns histogram

At the first glance, we notice difference in the scale of the numeric values which signals the need for using an scaling method on the numeric values. As more properties located in the center of the city the distribution of the `latitude` and `longitude` are bell-shaped. There is a significant skewness obvious on the `reviews_per_month`, `review_scores_rating`, and `accommodates` , so we need to take a deeper look to these columns and be aware of their probable corelation.

### Correlation Matrix and Correlation Plot
Before running any ML model, checking the correlation between different columns is essential. Correlation matrix and plot can help to identify relationships between different features, which can be useful for feature selection. Features that are highly correlated with the target variable are likely to be more informative than those that are not. On the other hand we can find the relationship between two features. For example, if two features are highly correlated, they may be combined into a single feature.

Correlation can also help to identify any multicollinearity issues, which can negatively impact the performance of linear models, it is important to address them before modeling.
Understanding the correlation structure of your data can also help to decide on the appropriate machine learning algorithm. For example, if the data is highly correlated, a linear model may be appropriate, while if the data is not highly correlated, a decision tree algorithm may be more appropriate.
<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/corr_matrix.png" width="100%"/>
</p>
Fig 4: Numeric columns correlation matrix


It seems that the target column `reviews_per_month` is showing a more strong correlation with current `number_of_reviews`, `price`, and `host_listing_count`. Let' investigate more by plotting the correlations. The plot below also reveals the same pattern. So we need to be aware of this correlation and verify that when reporting the pipeline's performance and feature importance results. The properties with a higher number of reviews are the ones that were more popular and active, so they probably have a higher `reviews_per_month`.


<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/corr_plot.png" width="100%"/>
</p>
Fig 5: Numeric columns correlation plots

### Target Column
Based on an initial EDA of the dataset, it seems that the target column `reviews_per_month` is showing a skewed distribution, so target column transformation might be necessary for this problem.
<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/untrans_target.png" width="45%"/>
</p>

Fig 6: How the target values (`reviews_per_month`) are distributed.

So normalizing the target values would be a good idea here. Normalizing the target values in machine learning can improve the performance of a model for a few reasons like:
- Helping to ensure that the optimization algorithms converge more quickly and effectively
- Helping to prevent numerical stability issues, as well as reduce the impact of outliers
- Preventing the features of the model from dominating the optimization process

After normalizing the target column values with NumPy's `np.log10` function, their histogram is indicating a bell-shaped distribution.
<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/eda_plots/trans_target.png" width="55%"/>
</p>

Fig 7: Target values (`reviews_per_month`) distribution after normalization with `np.log10` function.

## Feature engineering
Feature engineering can be one of the most important stages of building a machine learning pipeline. It can help to improve the predictive power of a model by creating new features that are more informative or relevant to the problem. Feature engineering can also help to reduce the dimensionality of the data by combining or removing features that are redundant or not informative.

In the current dataset there are some feature that we can extract relevant features from them including `last_review`, `host_since`, and `amenities`.
You can reproduce the feature engineered dataset of this pipeline by running [Step 5 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).

### Date Column: last_review, and host_since
Based on the initial evaluation of the `last_review` column, it is observed that this feature is a date column to report the late review date. The null value represents no review for a rental property. On other hand, there is another useful date column `host_since` which reports the first day of the host joining AirBnb. Experienced hosts who have been using AirBnb for a longer time probably have more reviews and higher ratings. 

To take these features into account and make the date column more interpretable for the model, we create a new feature called `time_diff` that show the duration between `last_review` and `host_since` features in days. We believe a higher `time_diff`, results in higher `reviews_per_month`.

### Name Column
`name` of a rental property is influential when users are looking for places. According to the [rental tips](https://www.igms.com/airbnb-titles/), a well-crafted Airbnb title can attract up to 5X more bookings, and we know that more bookings result in more reviews and higher `reviews_per_month`. 

To extract a probably useful feature out of the `name` feature, we'll use `SentimentIntensityAnalyzer` from `nltk.sentiment.vader` to analyze the sentiment of the listing. Based on the initial evaluation, there are about 50% of zero sentiment scores for the data. So we need to note that the focus here is to identify their properties, either positive or negative sentiment. Positive sentiment results in a higher booking rate and a higher review rate. So the output of this function is three categories `Positive`, `Negative`, and `Neutral`.

## Preprocessing and Transformations
Preprocessing in machine learning refers to the steps taken to prepare raw data for use in a model. This can include cleaning and formatting the data, as well as normalizing or scaling the values. It is an essential stage of a building an stable ML pipeline, since these step is important for improving the performance and accuracy of a model, and can also help to reduce the risk of overfitting. 

By using from `sklearn`'s `make_column_transformer`, a preprocesser object has been created to address different types of the features. We also have transformed the target with log transformation earlier.

### Categorical and Binary Features
One-hot encoding is a technique used to convert categorical variables into numerical variables. In machine learning, many algorithms cannot operate on categorical data directly and require numerical input. One-hot encoding is a way to convert categorical data into numerical data by creating a new binary column for each unique category and assigning a 1 or 0 to indicate the presence or absence of that category in the original data.
It is also useful for handling categorical variables with many levels, as it avoids the problem of assigning arbitrary numerical values to each level, which can introduce bias into the model. Features that I decided to pass into `sklearn`'s `OneHotEncoder` are:
- neighbourhood_cleansed
- room_type
- host_is_superhost
- instant_bookable

### Ordinal Features
Ordinal encoding is a technique used to convert categorical variables into numerical variables, similar to one-hot encoding. However, unlike one-hot encoding, ordinal encoding preserves the ordinal relationship between the categories.

In ordinal encoding, each category is assigned a unique integer value, with the assumption that the categories have an inherent ordinal relationship. For example, if you have a categorical variable with the levels "low", "medium", and "high", you would assign the values 1, 2, and 3 respectively.

Ordinal encoding can be useful in situations where the categorical variable has an inherent order or ranking, such as "low", "medium", "high" or "cold", "cool", "warm", "hot". However, ordinal encoding can introduce bias into the model if the categorical variable does not have an inherent ordinal relationship.

In some cases, ordinal encoding can be less efficient than one-hot encoding because it creates fewer columns, but it can introduce bias or make the model assume a particular ordering or relationship that may not be true for the data. In this case the column `name_sent` with the ordinal levels of `Positive`, `Neutral`, and `Negative` are passed to `sklearn`'s `OrdinalEncoder` 

### Numerical Features
I suspect this problem doesn't have a 100% linear or non-linear nature. So using polynomial transformation on then numeric values might lead to producing useful features (of course with the cost of higher model complexity). You may ask why Polonomial transformation?

Polynomial transformation is a technique used to add non-linearity to a model. In machine learning, many models, such as linear regression, make assumptions about the linearity of the relationship between the input variables and the output variable. However, in some cases, the relationship between the variables may be non-linear. Polynomial transformation can help to capture these non-linear relationships by adding polynomial terms to the feature set.

For example, if you have a single input variable, x, and you want to model a non-linear relationship, you can add higher-order terms such as x^2, x^3, etc. to the feature set. This can help to increase the flexibility of the model and improve its ability to capture non-linear patterns in the data.

Polynomial transformation can also help to reduce the risk of overfitting by adding new features that can help to explain the variation in the output variable without simply memorizing the training data.

It is important to note that applying polynomial transformation to a feature will increase the feature dimensionality, which can increase the complexity of the model and make it more prone to overfitting. Therefore, it's important to use cross validation and regularization methods to avoid overfitting.

After applying the polynomial trnasformation, it is the time to normalize all the numerical features with `StandardScaler`. Standard scaling, also known as normalization or Z-score normalization, is a technique used to adjust the values of a feature so that they have a mean of zero and a standard deviation of one. This is done by subtracting the mean of the feature from each value and then dividing by the standard deviation.

Standard scaling is often used in machine learning to ensure that all features are on the same scale and have similar properties. This is important because many machine learning algorithms, such as linear regression and neural networks, are sensitive to the scale of the input variables. Features that are on a different scale can dominate the model and make it difficult for the other features to have any impact on the output.

Standard scaling can also help to improve the performance and accuracy of a model by making it easier for the algorithm to find patterns in the data. For example, if a feature has a large range of values, it can be difficult for an algorithm to detect patterns in that feature. Scaling the feature can help to make the patterns more visible to the algorithm.

In addition, standard scaling can also help to improve the interpretability of the model by making it easier to compare the importance of different features. By scaling the features, we can ensure that the magnitude of the coefficients of the model represents the importance of each feature in the prediction. The numeric features are passed to Polynomiad feature and StandardScaler are:
- host_listings_count
- accommodates
- price
- minimum_nights
- minimum_nights
- number_of_reviews
- time_diff
- review_scores_rating


### Text Features
By using the bag of words and CountVectorizer technique we process the `amenities` feature. `amenities` of a rental property shows its quality level and it affect customers' satisfaction. Usually more luxury AirBnbs with many amenities are getting higher review rating and the will have a higher chance of being booked if they are available. Having kitchen accessories, pool, separate beds, and etc. also might affect the frequency of the booking and reviews. 
CountVectorizer is a feature extraction method in natural language processing (NLP) that is used to convert a collection of text documents into a numerical feature matrix. It is commonly used as a step in the preprocessing of text data.

The basic idea behind CountVectorizer is to create a vocabulary of words from the text documents and then represent each document by a vector of the frequency of each word in the vocabulary.

The vector created by CountVectorizer is commonly known as a bag of words representation, and it discards grammar and word order information, only considering the occurence of each word.

### Drop Features
Lastly, the are some features that they don't add any value to the model predictions and we will drop them, including:
- id
- latitude
- longitude
- minimum_nights_avg_ntm
- maximum_nights_avg_ntm
- name (switched by name_sent)
- description

## Baseline and Linear Models
Based on the nature of the problem (regression) and the nature of the target, the metrics below will be chosen to assess the models are:
- $R^2$: The coefficient of determination to calculate the ratio between the variance of the model prediction and total variance. I will use this score for the model and hyperparameter optimization.
- NRMSE: This will also be used for reporting purposes. Here the RMSE is the root square mean of the standard error that it's mode interpretable MSE.

You can reproduce the linear model results by running [Step 6 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).

As the baseline model, I picked the `LinearRegression` which is a linear model without any regularization. This model is a simple a simple model that we used as a starting point for comparison when developing more complex models. It serves as a benchmark or reference point against which the performance of other models can be measured. This model reports a training score of about 63.8% and a test score of about 58.2% which doesn't see very appealing for a start point-to-start model, feature selection, and hyperparameter tuning.

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/01_baseline_results_df.png" width="45%"/>
</p>

Table 1: Performance of baseline and Linear models

At the next level, using regularization on the linear regression model might help with improving the scores. Here `Ridge` (Linear Regression with L2 regularization) model is used. 

L2 regularization, also known as weight decay, is a technique used to prevent overfitting in machine learning models. It works by adding a penalty term to the cost function that the model is trying to optimize. The penalty term is the sum of the squares of the model's weights (coefficients). The idea behind L2 regularization is that it will encourage the model to have smaller weights, which can help to prevent overfitting by reducing the complexity of the model.

Based on the initial analysis, it seems that both linear regression and ridge (linear model with L2 regularization) are reporting close results but the difference between train and test score has been decreased, and this sings that the L2 regularization had already smoothened the model coefficients. But how L2 regularization improved the model interoperability?
By taking a deeper look to the coefficients of the features in Baseline model, we realize that they don't make any sense!

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/02_baseline_coeff_plt.png" width="100%"/>
</p>

Fig 8: Highest coefficients for the Linear Regression model and features 

As It is indicated above the top features for the baseline model are some random words generated by the OneHotEncoder of the neighborhood and CountVectorizer of the amenities. As it was expected the coefficient of features in LinearRegression model without any regularization are sharp and in the scale of -2 to 2. But how about the most important features afre regularization by using `Ridge` model?


<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/03_ridge_coeff_plt.png" width="98%"/>
</p>

Fig 9: Highest coefficients for the Ridge model and features 

As we are expecting the magnitude of the coefficients have dropped dramatically to -0.3 to 0.2 which means the model responds smoother and the result are more interpretable. Interestingly, we see `number_of_reviews` and `review_scores_ratings`.

The bar plots below present the range of the coefficients for each model, and we see a lower range of the coefficient by choosing a higher hyperparameter of $\alpha$ (in this case $\alpha = 100$) for the `Ridge` model (`Best_Ridge` model). Increasing this hyperparameter intensifies the penalty term in the cost term of the ridge model and results in a lower range of the coefficients in the cost function.

## Non-Linear Models
In this step we will try a couple of non-linear, or let's say ensemble models, LGBM Regressor, and XGB Regressor.
You can reproduce the ensemble model results by running [Step 7 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).

### LightGBM Regressor
LightGBM Regressor is a gradient boosting framework that uses tree-based learning algorithms. It is designed to be efficient and scalable, and it is particularly well-suited for large datasets and high-dimensional datasets.

Gradient boosting is a method of ensemble learning that combines multiple weak learners (such as decision trees) to create a strong learner (a model that can make accurate predictions). LightGBM is an efficient implementation of gradient boosting that uses a tree-based learning algorithm. It builds the trees leaf-wise, which is different from the traditional level-wise approach. This allows LightGBM to focus on the more important features and build more accurate models.

### XGB Regressor
XGBoost Regressor is an implementation of the gradient boosting algorithm for regression problems. It is a powerful and popular machine learning algorithm that is designed for both efficiency and performance. It is an optimized version of the gradient boosting algorithm and is known for its speed and accuracy.

Like other gradient boosting algorithms, XGBoost Regressor creates an ensemble of decision trees to make predictions. Each tree is trained to correct the mistakes of the previous tree, and the final ensemble is a combination of all the trees. This allows XGBoost to capture non-linear relationships in the data, and it can handle large datasets and high-dimensional datasets.

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/04_linear_nonlinear_results_df.png" width="80%"/>
</p>

Table 2: Performance of baseline, Linear, and Ensemble models

Regarding their results on the problem, LGBM and XGB showed better performance on the cross-validation scores. The test score increased from a range of 60% increase to the range of 80% for both `LGBM Regressor` and `XGB Regressor`. On the other hand both of these models tend towards overfitting with a training score of 84% and 88% and a test score of 81%. These models generally perform better than linear models. In terms of speed, the `XGB Regressor` model takes much longer to run un compare to the `LGBM` model.

When it comes to the interpretability of the results, we need to consider how the model takes the features into account. The table below presents the most important features (top 20) for both `LGBM Regressor` and `XGB Regressor`.

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/06_top_feat_df.png" width="60%"/>
</p>

Table 3: Top 20 features for XGB and LGBM Regressors

We also observe more interpretable results here for LGBM model, leading to a better generalization. Moving forward to the hyperparameter optimization and model interoperability, `LGBM Regressor` will be chosen as the final model.

## Hyperparameter optimization

Like any other essential stages of the building a pipeline, we need to pass this stage to seeking for best possible combination of the hyperparameters. By finding the best combination of hyperparameters, the model can perform better on the test data. Hyperparameter optimization can help to find the best trade-off between bias and variance and can help the model to generalize better. In this stage the below range is chosen for hyperparameter optimization:
- n_estimators: [10, 50, **100**, 500]
- learning_rate: [0.0001, 0.001, **0.01**, 0.1, 1.0]
- subsample: [**0.5**, 0.7, 1.0]
- max_depth [3, 4, **5**]

After running the hyperparameter optimization we reach to the best combination of them which is highlighted above. This combination leads to an increase in the test score, but makes to model prone to the overfitting, since the difference between the train and test scores is higher.

## Interpretation and Feature Importance

In this section, I have used the `permutation` feature importance and `SHAP` (SHapley Additive exPlanations) to explain the importance of the features. 

### Permutation Feature Importance
Permutation feature importance is a method for determining the importance of individual features in a machine learning model. The method works by randomly shuffling the values of a single feature and measuring the impact on the model's performance. The idea is that if a feature is important, then shuffling its values should result in a significant decrease in performance. This can be done for each feature in the dataset, allowing for a ranking of feature importance. You can reproduce the permutation feature importance results by running [Step 8 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/07_perm_imp_plt.png" width="80%"/>
</p>

Fig 10: Feature importances with permutation method

What we observe in the permutation feature importance results is closely aligned with what we observed with Ridge model. The `number_of_reviews`, `amenities`, and `time_diff` are the most important features for these two ensemble models.

### SHAP
On the other hand, SHAP (SHapley Additive exPlanations) is a method for explaining the output of any machine learning model. It is based on the concept of Shapley values from cooperative game theory. SHAP values provide a way to assign importance to each feature (or input) in a model's prediction for a specific instance. The method gives an explanation of the prediction in terms of the contribution of each feature, and it guarantees that the sum of the feature importance values for any given prediction is equal to the difference between the prediction and the expected value of the model's predictions. Additionally, SHAP is model-agnostic, which means it can be used to explain the predictions of any kind of machine learning model. You can reproduce the SHAP analysis results by running [Step 9 in the usage section](https://github.com/mrnabiz/zero-to-hero-ml-pipline#option-1-running-step-by-step-scripts-to-build-th-pipeline).

<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/08_shap_sum_plt.png" width="70%"/>
</p>

Fig 11: Feature importances with SHAP method


Based on the feature weight we observe in the SHAP method, features like price, number_of_reviews, and some amenities like heating are affecting the SHAP value more and the are more important for the model. Unlike our expectations the ensemble models are performing and explaining the result better than linear models, which reveals the non-linear nature of the dataset. The final trained model object has also been saved for deployment purposes.

## Deployment Performance

The test score is about 81% for LGBM model which we decided to move forward and also 82% for LGBM Regressor with hyperparameter optimization and this shows that this model generalizes fairly well. 

Using LGBM Regressor model seems more intuitive and easy to understand since we saw some features like `number_of_reviews`, and `time_diff` are contributing most in the prediction weights. In reality having more reviews signals that the rental is active and more people renting it and leaving their comments, while this naturally affects the target which is the number of reviews per month. Having lower minimum night shows again represents more activity of the property and a higher number of reviews per month. Also receiving more recent reviews signals the model's properties activity in the past and hints at more reviews per month.


<p>
  <img src="https://raw.githubusercontent.com/mrnabiz/zero-to-hero-ml-pipline/main/results/model_results/05_scores_df.png" width="25%"/>
</p>

Fig 12: Various models score on deployment data (test data)

## Conclusion

| Stage | Important Result
| --- | -----------
| EDA and Data Wrangling | In this mini-project, I tried a variety of different linear and non-linear models on a regression problem. We were specifically interested in predicting `reviews_per_month` per month for Airbnb rentals in London. Predicting the `reviews_per_month` and presenting it to the hosts play a critical role in their effort to collect reviews and boost their listing. By performing the initial EDA, it can be inferred that the `number_of_reviews` shows a good correlation with `reviews_per_month` and it's quite interpretable too. The higher the `number_of_reviews`, the higher the properties activity, and the higher `reviews_per_month`. The target values are skewed and transforming them with the `log` function, increased the model's accuracy by about 40%.
| Baseline and Linear Models | Running `linear regression` without and with `L2` regularization shows that the performance of the linear models is about 61% while offering interpretable results with the regularization making sense in the real world. `L2` regularization offers the most promising results in comparison to the other linear models.
| Non-Linear Models | Using non-linear models like `LGBM` and `XGB` regressors opens another door to the model's performance while sacrificing the simplicity and interpretability of the results for `XGB` model. These models on average increase the performance is about accuracy by 20%.
| Hyperparameter Optimization | Carrying out hyperparameter optimization results in very close train and test scores which in fact ensures us that we are not leaning toward the optimization bias.


Throughout this mini-project, there were several times that the interpretability-accuracy trade-off was obvious and the main lesson learned from this project was seeing how different models are responding to the real problem we are trying to solve. For this particular problem using non-linear would be a better choice due to the nature of the problem (building a recommendation engine to motivate the hosts to collect more reviews). To improve the results, using model combinations lie stacking, or voting systems would be helpful.

I hope you have enjoyed this journey and I would love to hear your feedback. Feel free to email me at [hello@nabi.me](mailto:hello@nabi.me)
