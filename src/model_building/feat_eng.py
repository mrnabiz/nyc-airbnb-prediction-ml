# author: Mohammad Reza Nabizadeh
# date: 2023-01-05

"""This script runs the feature engineering and extracts two features from 
the preprocessed data.


Usage: feat_eng.py --output_file_path=<output_file_path> --input_file_path=<input_file_path>
 
Options:
--output_file_path=<output_file_path>   Path to folder and file name where the 
file with new features will be saved, including the name, `in quotes.
--input_file_path=<input_file_path>		Path to folder and file name where the 
preprocessed file is stored, including the name, `in quotes.
"""

# Import dependencies
import pandas as pd
from docopt import docopt
import os
from utils import diff_days, get_sentiment


import warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=FutureWarning)


opt = docopt(__doc__)

def main(output_file_path, input_file_path):
    # Read the csv file of the raw dataset
    preprocessed_df = pd.read_csv(input_file_path, low_memory=False)

    # Apply the feature extractign functions
    preprocessed_df['host_since'] = pd.to_datetime(
                                        preprocessed_df['host_since']
                                        )
    preprocessed_df['last_review'] = pd.to_datetime(
                                        preprocessed_df['last_review']
                                        )

    preprocessed_df = preprocessed_df.assign(time_diff=diff_days(
                            last_review_date=preprocessed_df['last_review'],
                            host_join_date=preprocessed_df['host_since'])
                            )
                                         
    preprocessed_df = preprocessed_df.assign(
        name_sent=preprocessed_df['name'].apply(get_sentiment)
        )

    # Clean the initial columns
    drop_columns = ['last_review', 'host_since', 'first_review']
    preprocessed_df = preprocessed_df.drop(columns=drop_columns)


    # Write the csv file
    try:
        preprocessed_df.to_csv(output_file_path, index = False)
    except:
        os.makedirs(os.path.dirname(output_file_path))
        preprocessed_df.to_csv(output_file_path, index = False)

if __name__ == "__main__":
    main(opt["--output_file_path"], opt["--input_file_path"])