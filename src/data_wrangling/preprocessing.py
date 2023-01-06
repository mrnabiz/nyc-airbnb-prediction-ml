# author: Mohammad Reza Nabizadeh
# date: 2023-01-04

"""This script preprocesses the Airbnb listing data and prepares it for the model building.
This preprocessing phase is all about removing unnecessary columns and prepare the data for EDA.

Usage: preprocessing.py --output_file_path=<output_file_path> --input_file_path=<input_file_path>
 
Options:
--output_file_path=<output_file_path>		Path to folder and file name where the preprocessing data will be saved, including the name, `in quotes.
--input_file_path=<input_file_path>			Path to folder and file name where the clean file is stored, including the name, `in quotes.
"""

# Import dependencies
import pandas as pd
from docopt import docopt
import os
import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)

opt = docopt(__doc__)


def main(output_file_path, input_file_path):
    # Read the csv file of the raw dataset
    clean_df = pd.read_csv(input_file_path, low_memory=False)

    # Clean the raw dataframe columns
    preprocessed_drop_columns = ['listing_url', 'neighborhood_overview', 'host_id', 'host_url',
                                 'host_name', 'host_location', 'host_about', 'host_response_time',
                                 'host_response_rate', 'host_acceptance_rate', 'host_picture_url',
                                 'host_total_listings_count', 'host_verifications', 
                                 'host_identity_verified', 'property_type', 'bathrooms_text', 'bedrooms',
                                 'beds', 'number_of_reviews_ltm', 'review_scores_accuracy',
                                 'review_scores_cleanliness', 'review_scores_checkin', 
                                 'review_scores_communication', 'review_scores_location', 
                                 'review_scores_value'
                                 ]
    
    clean_df_reduced = clean_df.drop(columns=preprocessed_drop_columns)

    # Drop Null Target rows
    preprocessed_df = clean_df_reduced.dropna(subset=['last_review','description','name',
                                                      'minimum_nights_avg_ntm'])

    # Fix the date time columns
    preprocessed_df['host_since'] = pd.to_datetime(preprocessed_df['host_since'])
    preprocessed_df['first_review'] = pd.to_datetime(preprocessed_df['first_review'])
    preprocessed_df['last_review'] = pd.to_datetime(preprocessed_df['last_review'])

    # Fix the price column and change its type to float
    preprocessed_df['price'] = preprocessed_df['price'].str.replace(
                                                                    '$', '').str.replace(
                                                                    ',', '')
    
    preprocessed_df['price'] = preprocessed_df['price'].astype(float)

    # Fix the amenities column for the countvectorizer
    preprocessed_df['amenities'] = preprocessed_df['amenities'].str.replace(
                                                                            '[', '').str.replace(
                                                                            ']', '').str.replace(
                                                                            '"', '')
    
    # Write the csv file
    try:
        preprocessed_df.to_csv(output_file_path, index = False)
    except:
        os.makedirs(os.path.dirname(output_file_path))
        preprocessed_df.to_csv(output_file_path, index = False)

if __name__ == "__main__":
    main(opt["--output_file_path"], opt["--input_file_path"])