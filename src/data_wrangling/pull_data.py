# author: Mohammad Reza Nabizadeh
# date: 2022-12-11

"""This script pulls London's most recent Airbnb listing data.
The latest version is released on September 10th, 2022

Usage: pull_data.py --file_path=<file_path> --url=<url>
 
Options:
--file_path=<file_path>		Path to folder and file name where the file will be
 saved, including the name, `in quotes.
--url=<url>			        URL to the online dataset, in quotes.
"""

# Import dependencies
import pandas as pd
from docopt import docopt
import os

opt = docopt(__doc__)


def main(file_path, url):
    #Pull the dataset from the source and read the csv file
    raw_df = pd.read_csv(url, parse_dates=True)

    # Write the csv file
    try:
        raw_df.to_csv(file_path, index = False)
    except:
        os.makedirs(os.path.dirname(file_path))
        raw_df.to_csv(file_path, index = False)

if __name__ == "__main__":
    main(opt["--file_path"], opt["--url"])