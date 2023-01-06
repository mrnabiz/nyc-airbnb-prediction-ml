# London Airbnb Prediction ML Model


## Download and store the raw data
Download the data by running `src/pull_data.py`:\
    `--file_path` should be the path where the data will be saved,\
    `--url` should be the link to the data

    python src/pull_data.py --file_path="data/raw/raw_df.csv" --url="http://data.insideairbnb.com/united-kingdom/england/london/2022-09-10/data/listings.csv.gz"

## Clean the raw data frame to remove unnecessary columns
Download the data by running `src/clean_data.py`:\
    `--output_file_path` should be the path where the clean data will be saved,\
    `--input_file_path` should be the path where the raw data is stored

    python src/clean_data.py --output_file_path="data/raw/clean_df.csv" --input_file_path="data/raw/raw_df.csv"

## Preprocess the cleaned to prepare it for the model building
Download the data by running `src/preprocessing.py`:\
    `--output_file_path` should be the path where the clean data will be saved,\
    `--input_file_path` should be the path where the raw data is stored

    python src/preprocessing.py --output_file_path="data/preprocessed/preprocessed_df.csv" --input_file_path="data/raw/clean_df.csv"

## Run the EDA analysis
Download the data by running `src/eda.py`:\
    `--output_dir`  Path to folder where the EDA results will be saved, `in quotes.\
    `--clean_df_file_path`  Path to folder and file name where the clean df is stored, including the name, `in quotes.\
    `--preprocessed_df_file_path`  Path to folder and file name where the preprocessed df is stored, including the name, `in quotes.

    python src/eda.py --output_dir="results/eda_plots" --clean_df_file_path="data/raw/clean_df.csv"  --preprocessed_df_file_path="data/preprocessed/preprocessed_df.csv"

## Run the feature engineering script
Download the data by running `src/feat_eng.py`:\
    `--output_file_path` should be the path where the file with new features will be saved,\
    `--input_file_path`  should be the path where the preprocessed data is stored.

    python src/feat_eng.py --output_file_path="data/preprocessed/feat_eng_df.csv" --input_file_path="data/preprocessed/preprocessed_df.csv"