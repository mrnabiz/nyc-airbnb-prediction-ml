# author: Mohammad Reza Nabizadeh
# date: 2023-01-06

"""

This script runs the baseline linear regression model and 
report the model scores.

Usage: baseline_model.py --output_dir=<output_dir> --input_file_path=<input_file_path>
 
Options:
--output_dir=<output_dir>
Path to folder where the baseline model results (R2 and Neg Root Mean Squared 
Error will be saved, `in quotes.

--input_file_path=<input_file_path>
Path to folder and file name where the preprocessed file with engineered
features is stored, including the name, `in quotes.

"""


# Import dependencies
import numpy as np
import pandas as pd
from docopt import docopt
import os
import dataframe_image as dfi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from utils import mean_std_cross_val_scores
from preprocessor import preprocessor
import pickle


opt = docopt(__doc__)

def main(output_dir, input_file_path):

    # Read the preprocessed file
    prep_feat_df = pd.read_csv(input_file_path, low_memory=False)

    # Split the file for training and testing the model
    train_df, test_df = train_test_split(prep_feat_df,
                                         test_size=0.7,
                                         random_state=123)
    
    # Form the X and y of the training and test dataframes
    X_train = train_df.drop(columns=['reviews_per_month'])
    y_train_untrans = train_df['reviews_per_month']
    X_test = test_df.drop(columns=['reviews_per_month'])
    y_test_untrans = test_df['reviews_per_month']

    # Transform the target column
    y_train = np.log10(y_train_untrans)
    y_test = np.log10(y_test_untrans)

    # Form the model pipeline
    results = {}
    scoring = {
        "R2": "r2",
        "NRMSE": "neg_root_mean_squared_error"
        }
    lr_model = LinearRegression()
    lr_pipeline = make_pipeline(preprocessor(), lr_model)
    lr_pipeline.fit(X_train, y_train)

    # Run the cross validation ans store the results
    results["baseline_lr"] = mean_std_cross_val_scores(lr_pipeline,
                                                       X_train,
                                                       y_train,
                                                       return_train_score=True,
                                                       scoring=scoring,
                                                       cv=5)
    results_df = pd.DataFrame(results)

    # Save the results dictionary to a pickle file
    outpt_results_dict = output_dir + '/results_dict.pkl'
    outpt_results_df = output_dir + '/01_baseline_results_df.png'
    
    try:
        dfi.export(results_df, outpt_results_df, dpi=200)
        with open(outpt_results_dict, 'wb') as f:
            pickle.dump(results, f)

    except:
        os.makedirs(os.path.dirname(outpt_results_dict))
        dfi.export(results_df, outpt_results_df, dpi=200)
        with open(outpt_results_dict, 'wb') as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    main(opt["--output_dir"], opt["--input_file_path"])    