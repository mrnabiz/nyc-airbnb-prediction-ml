# author: Mohammad Reza Nabizadeh
# date: 2023-01-04

"""This script cleans the Airbnb listing data.
This cleaning phase is all about removing unnecessary columns and prepare the data for EDA.

Usage: clean_data.py --output_file_path=<output_file_path> --input_file_path=<input_file_path>
 
Options:
--output_file_path=<output_file_path>		Path to folder and file name where the cleaned data will be saved, including the name, `in quotes.
--input_file_path=<input_file_path>			Path to folder and file name where the raw file is stored, including the name, `in quotes.
"""

# Import dependencies
import pandas as pd
from docopt import docopt
import os

opt = docopt(__doc__)


def main(output_file_path, input_file_path):
    # Read the csv file of the raw dataset
    raw_df = pd.read_csv(input_file_path, low_memory=False)

    # Clean the raw dataframe columns
    drop_columns = ['scrape_id', 'last_scraped', 'source', 'picture_url', 'host_thumbnail_url',
                    'host_neighbourhood', 'host_has_profile_pic', 'neighbourhood', 
                    'neighbourhood_group_cleansed', 'bathrooms', 'minimum_minimum_nights',
                    'maximum_minimum_nights', 'minimum_maximum_nights', 'maximum_maximum_nights',
                    'calendar_updated', 'has_availability', 'availability_30', 'availability_60',
                    'availability_90', 'availability_365', 'calendar_last_scraped',
                    'number_of_reviews_l30d', 'license', 'calculated_host_listings_count',
                    'calculated_host_listings_count_entire_homes',
                    'calculated_host_listings_count_private_rooms',
                    'calculated_host_listings_count_shared_rooms'
                    ]
    clean_df = raw_df.drop(columns=drop_columns)

    # Drop Null ID rows
    clean_df = clean_df[pd.to_numeric(clean_df['id'], errors='coerce').notnull()]

    # Fix the date time columns
    clean_df['host_since'] = pd.to_datetime(clean_df['host_since'])
    clean_df['first_review'] = pd.to_datetime(clean_df['first_review'])
    clean_df['last_review'] = pd.to_datetime(clean_df['last_review'])

    # Write the csv file
    try:
        clean_df.to_csv(output_file_path, index = False)
    except:
        os.makedirs(os.path.dirname(output_file_path))
        clean_df.to_csv(output_file_path, index = False)

if __name__ == "__main__":
    main(opt["--output_file_path"], opt["--input_file_path"])