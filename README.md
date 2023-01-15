# Zero to Hero Machine Learning Pipeline: London Airbnb Open Data 
## Introduction
**"He was no one, Zero, Zero
Now he's a honcho, He's a hero, hero!"**
Isn't this familiar? Absolutely it is. No way you could forger about Disney's
cool animation, Hercules.

![hercules.png](https://static.wikia.nocookie.net/disney/images/f/fa/Zero_to_Hero.png)
[Source](https://disney.fandom.com/wiki/Zero_to_Hero)

"Zero to Hero" is a phrase that is often used to describe a transformation from 
a novice or beginner to an expert or accomplished individual. It can also refer 
to a person who starts with very little and through hard work and determination,
 becomes successful. Throughout this journey, I want to take you to a 
step-by-step guide on building a regression Machine Learning pipeline.

This mini-project is about building a regression pipeline to predict the 
`number of reviews per month` of each Airbnb property, trained based on a 
dataset collected on September 2022 in London.

## Dataset
The dataset to build this pipeline was captured from [Detailed Airbnb Listing Data (London, Sep 2022)](https://www.kaggle.com/datasets/mrnabiz/detailed-airbnb-listing-data-london-sep-2022) on
 Kaggle. The originad data was prepared by [Inside Airbnb project](http://insideairbnb.com/). 
 The mission of Inside Airbnb is to empower residential communities with data 
 and information that enables them to understand, make decisions and have 
 control over the effects of Airbnb's presence in their neighborhoods.

## Proposed ML pipeline
This regression ML pipeline is built for predicting `number of reviews per month` 
on London's Airbnb data.
Predicting this number is important because it can help AirBnb to better 
understand and manage customer engagement. This can have a number of benefits, 
such as improving customer satisfaction, optimizing marketing efforts, increased revenue, and forecasting.

## Sharing the Results
This pipeline is created in Python, using a group of ML models from [scikit-learn](https://scikit-learn.org/stable/index.html) library.
For the sake of reproducibility, all of the pipeline is scripted and the results will be stored in relevant directories.
There is a notebook to report the performance of the various models which uses the stored results along with presenting them through visualizations.
In order to render all the plots properly, a HTML version of this report has been updated too. [The report can be found here.](https://ubc-mds.github.io/eurovision_contest_rank_analysis/doc/report.html)

## Usage
This pipeline can be reproduced by cloning the GitHub repository, installing the dependencies listed below and running the following commands at the terminal from the root directory of this project. To run this model follow the below steps:

## Download and store the raw data
Download the data by running `src/data_wrangling/pull_data.py`:\
    `--file_path` should be the path where the data will be saved,\
    `--url` should be the link to the data

    python src/data_wrangling/pull_data.py --file_path="data/raw/raw_df.csv" --url="http://data.insideairbnb.com/united-kingdom/england/london/2022-09-10/data/listings.csv.gz"

## Clean the raw data frame to remove unnecessary columns
Download the data by running `src/data_wrangling/clean_data.py`:\
    `--output_file_path` should be the path where the clean data will be saved,\
    `--input_file_path` should be the path where the raw data is stored

    python src/data_wrangling/clean_data.py --output_file_path="data/raw/clean_df.csv" --input_file_path="data/raw/raw_df.csv"

## Preprocess the cleaned to prepare it for the model building
Download the data by running `src/data_wrangling/preprocessing.py`:\
    `--output_file_path` should be the path where the clean data will be saved,\
    `--input_file_path` should be the path where the raw data is stored

    python src/data_wrangling/preprocessing.py --output_file_path="data/preprocessed/preprocessed_df.csv" --input_file_path="data/raw/clean_df.csv"

## Run the EDA analysis
Download the data by running `src/model_building/eda.py`:\
    `--output_dir`  Path to folder where the EDA results will be saved, `in quotes.\
    `--clean_df_file_path`  Path to folder and file name where the clean df is stored, including the name, `in quotes.\
    `--preprocessed_df_file_path`  Path to folder and file name where the preprocessed df is stored, including the name, `in quotes.

    python src/model_building/eda.py --output_dir="results/eda_plots" --clean_df_file_path="data/raw/clean_df.csv"  --preprocessed_df_file_path="data/preprocessed/preprocessed_df.csv"

## Run the feature engineering script
Download the data by running `src/model_building/feat_eng.py`:\
    `--output_file_path` should be the path where the file with new features will be saved,\
    `--input_file_path`  should be the path where the preprocessed data is stored.

    python src/model_building/feat_eng.py --output_file_path="data/preprocessed/feat_eng_df.csv" --input_file_path="data/preprocessed/preprocessed_df.csv"

## Run the linear models
Download the data by running `src/model_building/linear_models.py`:\
    `--output_dir`  Path to folder where the model results dictionary will be saved, `in quotes.\
    `--input_file_path`  Path to folder and file name where the preprocessed file with engineered features is stored, including the name, `in quotes.

    python src/model_building/linear_models.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

## Run the non linear models (ensembles)
Download the data by running `src/model_building/ensembles.py`:\
    `--output_dir`  Path to folder where the model results dictionary will be saved, `in quotes.\
    `--input_file_path`  Path to folder and file name where the preprocessed file with engineered features is stored, including the name, `in quotes.

    python src/model_building/ensembles.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

## Run the permutation feature importances analysis 
Download the data by running `src/model_building/perm_feat_importances.py`:\
    `--output_dir`  Path to folder where the model results dictionary will be saved, `in quotes.\
    `--input_file_path`  Path to folder and file name where the preprocessed file with engineered features is stored, including the name, `in quotes.

    python src/model_building/perm_feat_importances.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

## Run the SHAP analysis 
Download the data by running `src/model_building/shap_feat_importances.py`:\
    `--output_dir`  Path to folder where the model results dictionary will be saved, `in quotes.\
    `--input_file_path`  Path to folder and file name where the preprocessed file with engineered features is stored, including the name, `in quotes.

    python src/model_building/shap_feat_importances.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"



## References
[Scikit-learn: Machine Learning in Python](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html), Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.