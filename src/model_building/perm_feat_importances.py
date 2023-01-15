# author: Mohammad Reza Nabizadeh
# date: 2023-01-08

"""

This script runs the a set of ensemble models and 
report the model scores. The models are Random Forest Regressor,
Gradient Boosting Regressor, LGBM Regressor and XGB Regressor. Besides running
the models, it also runs permutation feature importance to help the model's 
interoperability.

Usage: perm_feat_importances.py --output_dir=<output_dir> --input_file_path=<input_file_path>
 
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
import matplotlib.pyplot as plt
from docopt import docopt
import os
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from preprocessor import preprocessor


opt = docopt(__doc__)

def main(output_dir, input_file_path):

    # Read the preprocessed file
    prep_feat_df = pd.read_csv(input_file_path, low_memory=False)

    # Split the file for training and testing the model
    train_df, test_df = train_test_split(prep_feat_df,
                                         test_size=0.9,
                                         random_state=123)
    
    # Form the X and y of the training and test dataframes
    X_train = train_df.drop(columns=['reviews_per_month'])
    y_train_untrans = train_df['reviews_per_month']
    X_test = test_df.drop(columns=['reviews_per_month'])
    y_test_untrans = test_df['reviews_per_month']

    # Transform the target column
    y_train = np.log10(y_train_untrans)
    y_test = np.log10(y_test_untrans)


    # LGBM model
    lgbm_model = LGBMRegressor(random_state=123)
    lgbm_pipeline = make_pipeline(preprocessor(), lgbm_model)
    lgbm_pipeline.fit(X_train, y_train)

    # XGB model
    xgb_model = XGBRegressor(random_state=123)
    xgb_pipeline = make_pipeline(preprocessor(), xgb_model)
    xgb_pipeline.fit(X_train, y_train)

    # Run the permutation feature imporance
    gbm_prm = permutation_importance(lgbm_pipeline,
                                     X_train, 
                                     y_train, 
                                     n_repeats=10, 
                                     random_state=123
                                     )
    xgb_prm = permutation_importance(xgb_pipeline, 
                                     X_train, 
                                     y_train, 
                                     n_repeats=10, 
                                     random_state=123
                                     )

    # Plot the feature importance
    fig_perm, axes = plt.subplots(ncols=2,
                         nrows=1,
                         figsize=(10, 7),
                         layout="constrained")

    # Permutation feature imporance plot for LGBM
    perm_sorted_idx = gbm_prm.importances_mean.argsort()
    axes[0].boxplot(
        gbm_prm.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_train.columns[perm_sorted_idx],
        )
    axes[0].set_title('LGBM Regressor')

    # Permutation feature imporance plot for XGB
    perm_sorted_idx = xgb_prm.importances_mean.argsort()
    axes[1].boxplot(
        xgb_prm.importances[perm_sorted_idx].T,
        vert=False,
        labels=X_train.columns[perm_sorted_idx],
        )
    axes[1].set_title('XGB Regressor')

    fig_perm.suptitle('Permutation Feature Importance', fontsize=16)

    # Save the results 
    output_perm_plt = output_dir + '/07_perm_imp_plt.png'
    
    try:
        fig_perm.savefig(output_perm_plt)

    except:
        os.makedirs(os.path.dirname(output_perm_plt))
        fig_perm.savefig(output_perm_plt)


if __name__ == "__main__":
    main(opt["--output_dir"], opt["--input_file_path"])   
    