# author: Mohammad Reza Nabizadeh
# date: 2023-01-06

"""

This script runs the a set of linear regression models and 
report the model scores. The models are simple Linear Regression Model,
Ridge (Linear Regression with L2 Regularization), and the optimized version of
Ridge for it's hyperparameter alpha.

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
import altair as alt
from utils import save_chart
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
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
                                         test_size=0.3,
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
    # Scoring mterics
    scoring = {
        "R2": "r2",
        "NRMSE": "neg_root_mean_squared_error"
        }
    
    # Form the results dictionary to save the linear model results along with
    # baseline model
    results = {}
    scores = {}

    # Simple baseline LR model
    lr_model = LinearRegression()
    lr_pipeline = make_pipeline(preprocessor(), lr_model)
    lr_pipeline.fit(X_train, y_train)
    scores['Baseline_LR'] = str(round(100*lr_pipeline.score(X_test, 
                                                            y_test), 3)) + ' %'

    # Ridge Model
    ridge_model = Ridge(random_state=123)
    ridge_pipeline = make_pipeline(preprocessor(), ridge_model)
    ridge_pipeline.fit(X_train, y_train)
    scores['Ridge'] = str(round(100*ridge_pipeline.score(X_test, 
                                                         y_test), 3)) + ' %'

    # Best Ridge Model (Hyperparameter alpha, optimized)
    # Hyper parameter search
    param_grid = {"ridge__alpha": 10.0 ** np.arange(-5, 5, 1)}
    param_search_ridge = GridSearchCV(ridge_pipeline,
                                      param_grid,
                                      return_train_score=True,
                                      n_jobs=-1,
                                      scoring="r2"
                                      )
    param_search_ridge.fit(X_train, y_train)
    best_alpha_ridge = param_search_ridge.best_params_['ridge__alpha']

    # Model building
    best_ridge_model = Ridge(random_state=123, alpha=best_alpha_ridge)
    best_ridge_pipeline = make_pipeline(preprocessor(), best_ridge_model)
    best_ridge_pipeline.fit(X_train, y_train)
    scores['Best_Ridge'] = str(round(100*best_ridge_pipeline.score(X_test, 
                                                                   y_test), 
                                                                   3)) + ' %'

    # Form the transformed columns and the coeffs list
    transformer_stage = lr_pipeline.named_steps['columntransformer']
    
    categorical_features_trans = transformer_stage.named_transformers_[
        'onehotencoder-1'].get_feature_names_out()
    binary_features_trans = transformer_stage.named_transformers_[
        'onehotencoder-2'].get_feature_names_out()
    ordinal_features_trans = transformer_stage.named_transformers_[
        'ordinalencoder'].get_feature_names_out()
    numerical_features_trans = transformer_stage.named_transformers_[
        'pipeline'].named_steps['polynomialfeatures'].get_feature_names_out()
    countvec_features_trans_amen = transformer_stage.named_transformers_[
        'countvectorizer'].get_feature_names_out()

    features_transformed = categorical_features_trans.tolist() + \
                            binary_features_trans.tolist() + \
                            ordinal_features_trans.tolist() + \
                            numerical_features_trans.tolist() + \
                            countvec_features_trans_amen.tolist()

    # Form the coeffs dataframes
    model_stage_lr = lr_pipeline.named_steps['linearregression']
    model_stage_ridge = ridge_pipeline.named_steps['ridge']
    model_stage_ridge_best = best_ridge_pipeline.named_steps['ridge']

    coefs_lr = pd.DataFrame(
        data=model_stage_lr.coef_.T,
        index=features_transformed,
        columns=["coef"]
        ).reset_index().rename(columns= {'index' : 'variable'})
    coefs_lr['model'] = 'Baseline_LR'

    coefs_ridge = pd.DataFrame(
        data=model_stage_ridge.coef_.T,
        index=features_transformed,
        columns=["coef"]
        ).reset_index().rename(columns= {'index' : 'variable'})
    coefs_ridge['model'] = 'Ridge'

    coefs_best_ridge = pd.DataFrame(
        data=model_stage_ridge_best.coef_.T,
        index=features_transformed,
        columns=["coef"]
        ).reset_index().rename(columns= {'index' : 'variable'})
    coefs_best_ridge['model'] = 'Best_Ridge'

    coefs = pd.concat([coefs_lr, coefs_ridge, coefs_best_ridge], axis=0)
    
    # Subset the top coefs for linear and ridge model for plotting purposes
    top_linear_coeffs_list = coefs.query("model == 'Baseline_LR'"
                                    ).sort_values(by='coef', key=abs, 
                                                  ascending=False
                                    )['variable'][0:20].to_list()

    top_ridge_coeffs_list = coefs.query("model == 'Best_Ridge'"
                                    ).sort_values(by='coef', key=abs, 
                                                  ascending=False
                                    )['variable'][0:20].to_list()

    plt_coeffs_linear = coefs[coefs['variable'].isin(top_linear_coeffs_list)]
    plt_coeffs_ridge = coefs[coefs['variable'].isin(top_ridge_coeffs_list)]

    sorted_feat_names_linear = plt_coeffs_linear.query("model == 'Baseline_LR'"
                                    ).sort_values(by='coef', 
                                                  ascending=False
                                    )['variable'].to_list()
    sorted_feat_names_ridge = plt_coeffs_ridge.query("model == 'Best_Ridge'"
                                    ).sort_values(by='coef', 
                                                  ascending=False
                                    )['variable'].to_list()

    # Run the cross validation and store the results for LR model
    results['Baseline_LR'] = mean_std_cross_val_scores(lr_pipeline,
                                                       X_train,
                                                       y_train,
                                                       return_train_score=True,
                                                       scoring=scoring,
                                                       cv=5
                                                       )

    # Run the cross validation and store the results for Ridge model
    results["Ridge"] = mean_std_cross_val_scores(ridge_pipeline,
                                                 X_train,
                                                 y_train,
                                                 return_train_score=True,
                                                 scoring=scoring,
                                                 cv=5
                                                 )

    # Run the cross validation and store the results for Ridge model
    results["Best_Ridge"] = mean_std_cross_val_scores(best_ridge_pipeline,
                                                      X_train,
                                                      y_train,
                                                      return_train_score=True,
                                                      scoring=scoring,
                                                      cv=5
                                                    )                                    
    results_df = pd.DataFrame(results)

    # Produce the coeffs barplot for top coeffs of baseline LR
    coefs_barplot_baseline_lr = alt.Chart(plt_coeffs_linear).mark_bar().encode(
        alt.X('coef', title = 'Coefficients'),
        alt.Y('variable', title = 'Feature', sort = sorted_feat_names_linear),
        color=alt.condition(
            alt.datum.coef > 0,
            alt.value("blue"),
            alt.value("red")
            )).properties(
                width=250).facet(column='model').properties(
                    title="What are the most important features for baseline \
                        Linear Regression model and what are their \
                            coefficients in Ridge and optimized Ridge model?"
                    ).resolve_scale(x='independent')

    # Produce the coeffs bar plot for top coeffs of Ridge
    coefs_barplot_ridge = alt.Chart(plt_coeffs_ridge).mark_bar().encode(
        alt.X('coef', title = 'Coefficients'),
        alt.Y('variable', title = 'Feature', sort = sorted_feat_names_ridge),
        color=alt.condition(
            alt.datum.coef > 0,
            alt.value("blue"),
            alt.value("red")
            )
        ).properties(
                width=250).facet(column='model').properties(
                    title="What are the most important features for optimized \
                        Ridge model and what are their coefficients in Ridge \
                            and Baseline Linear Regression model?"
                    ).resolve_scale(x='independent')


    # Save the results dictionary to a pickle file
    output_results_dict = output_dir + '/results_dict.pkl'
    output_scores_dict = output_dir + '/scores_dict.pkl'
    output_results_df_png = output_dir + '/01_baseline_results_df.png'
    output_coeffs_df = output_dir + '/lr_coeffs_df.csv'
    output_coeffs_baseline_lr_plt = output_dir + '/02_baseline_coeff_plt.png'
    output_coeffs_ridge_plt = output_dir + '/03_ridge_coeff_plt.png'
    
    try:
        dfi.export(results_df, output_results_df_png, dpi=200)
        with open(output_results_dict, 'wb') as f:
            pickle.dump(results, f)
        with open(output_scores_dict, 'wb') as f:
            pickle.dump(scores, f)
        coefs.to_csv(output_coeffs_df, index = False)
        save_chart(coefs_barplot_baseline_lr, output_coeffs_baseline_lr_plt, 2)
        save_chart(coefs_barplot_ridge, output_coeffs_ridge_plt, 2)

    except:
        os.makedirs(os.path.dirname(output_results_dict))
        dfi.export(results_df, output_results_df_png, dpi=200)
        with open(output_results_dict, 'wb') as f:
            pickle.dump(results, f)
        with open(output_scores_dict, 'wb') as f:
            pickle.dump(scores, f)
        coefs.to_csv(output_coeffs_df, index = False)
        save_chart(coefs_barplot_baseline_lr, output_coeffs_baseline_lr_plt, 2)
        save_chart(coefs_barplot_ridge, output_coeffs_ridge_plt, 2)

if __name__ == "__main__":
    main(opt["--output_dir"], opt["--input_file_path"])    