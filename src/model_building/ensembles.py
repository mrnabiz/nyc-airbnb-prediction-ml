# author: Mohammad Reza Nabizadeh
# date: 2023-01-08

"""

This script runs the a set of ensemble models and 
report the model scores. The models are Random Forest Regressor,
Gradient Boosting Regressor, LGBM Regressor and XGB Regressor.

Usage: ensembles.py --output_dir=<output_dir> --input_file_path=<input_file_path>
 
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
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm.sklearn import LGBMRegressor
from xgboost import XGBRegressor
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
    # Scoring metrics
    scoring = {
        "R2": "r2",
        "NRMSE": "neg_root_mean_squared_error"
        }
    
    # Form the results dictionary to save the linear model results along with
    # baseline and linear models.
    output_results_dict = output_dir + '/results_dict.pkl'
    with open(output_results_dict, 'rb') as f:
        results = pickle.load(f)
    
    output_scores_dict = output_dir + '/scores_dict.pkl'
    with open(output_scores_dict, 'rb') as f:
        scores = pickle.load(f)


    # LGBM model
    lgbm_model = LGBMRegressor(random_state=123)
    lgbm_pipeline = make_pipeline(preprocessor(), lgbm_model)
    lgbm_pipeline.fit(X_train, y_train)
    scores['LGBMRegressor'] = str(round(100*lgbm_pipeline.score(
                                                            X_test, 
                                                            y_test), 
                                                            3)) + ' %'

    # XGB model
    xgb_model = XGBRegressor(random_state=123)
    xgb_pipeline = make_pipeline(preprocessor(), xgb_model)
    xgb_pipeline.fit(X_train, y_train)
    scores['XGBRegressor'] = str(round(100*xgb_pipeline.score(
                                                            X_test, 
                                                            y_test), 
                                                            3)) + ' %'


    # LGBM model hyperparameter optimization
    lgbm_search_grid = {
        'lgbmregressor__n_estimators': [10, 50, 100, 500],
        'lgbmregressor__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'lgbmregressor__subsample': [0.5, 0.7, 1.0],
        'lgbmregressor__max_depth': [3, 4, 5]
        }

    pram_search_lgbm = GridSearchCV(lgbm_pipeline,
                                lgbm_search_grid,
                                return_train_score=True,
                                n_jobs=-1,
                                scoring="r2",
                                cv=2
                                )
    pram_search_lgbm.fit(X_train, y_train)

    n_estimators_lgbm = pram_search_lgbm.best_params_[
        'lgbmregressor__n_estimators'
        ]
    learning_rate_lgbm = pram_search_lgbm.best_params_[
        'lgbmregressor__learning_rate'
        ]
    subsample_lgbm = pram_search_lgbm.best_params_[
        'lgbmregressor__subsample'
        ]
    max_depth_lgbm = pram_search_lgbm.best_params_[
        'lgbmregressor__max_depth'
        ]

    # Best LGBM model
    best_lgbm_model = LGBMRegressor(random_state=123,
                                    n_estimators=n_estimators_lgbm,
                                    learning_rate=learning_rate_lgbm,
                                    max_depth=max_depth_lgbm,
                                    subsample=subsample_lgbm)
    best_lgbm_pipeline = make_pipeline(preprocessor(), best_lgbm_model)
    best_lgbm_pipeline.fit(X_train, y_train)
    scores['Best_LGBM'] = str(round(100*best_lgbm_pipeline.score(
                                                            X_test, 
                                                            y_test), 
                                                            3)) + ' %'

    # Run the cross validation and store the results for LGBM model
    results["LGBMRegressor"] = mean_std_cross_val_scores(lgbm_pipeline,
                                                    X_train,
                                                    y_train,
                                                    return_train_score=True,
                                                    scoring=scoring,
                                                    cv=5
                                                    )

    # Run the cross validation and store the results for XGB model
    results["XGBRegressor"] = mean_std_cross_val_scores(xgb_pipeline,
                                                    X_train,
                                                    y_train,
                                                    return_train_score=True,
                                                    scoring=scoring,
                                                    cv=5
                                                    )

    # Run the cross validation and store the results for best LGBM model
    results["Best_LGBM"] = mean_std_cross_val_scores(best_lgbm_pipeline,
                                                    X_train,
                                                    y_train,
                                                    return_train_score=True,
                                                    scoring=scoring,
                                                    cv=5)

    results_df = pd.DataFrame(results)
    scores_df = pd.DataFrame(scores.items(), columns=['Model', 'Score'])

    # Get the list of transformed columns for getting the feature importance
    transformer_stage = lgbm_pipeline.named_steps['columntransformer']

    categorical_features_trans = transformer_stage.named_transformers_[
        'onehotencoder-1'
        ].get_feature_names_out()
    binary_features_trans = transformer_stage.named_transformers_[
        'onehotencoder-2'
        ].get_feature_names_out()
    ordinal_features_trans = transformer_stage.named_transformers_[
        'ordinalencoder'
        ].get_feature_names_out()
    numerical_features_trans = transformer_stage.named_transformers_[
        'pipeline'
        ].named_steps[
            'polynomialfeatures'
            ].get_feature_names_out()
    countvec_features_trans_amen = transformer_stage.named_transformers_[
        'countvectorizer'
        ].get_feature_names_out()

    features_transformed = categorical_features_trans.tolist() + \
                                binary_features_trans.tolist() + \
                                ordinal_features_trans.tolist() + \
                                numerical_features_trans.tolist() + \
                                countvec_features_trans_amen.tolist()


    # Get the feature importance for each model to identify the top 20 important
    # feature of each model

    # LGBM feature importance
    feat_imp_lgbm_df = pd.DataFrame(
        data={
        "Importance": lgbm_pipeline.named_steps[
            'lgbmregressor'
            ].feature_importances_,
        },
        index=features_transformed).sort_values(
            by="Importance", ascending=False
            )
    feat_imp_lgbm_df['rank'] = feat_imp_lgbm_df[
        'Importance'
        ].rank(ascending=False)
    feat_imp_lgbm_df['model'] = 'LGBMRegressor'

    # XGB feature importance
    feat_imp_xgb_df = pd.DataFrame(
        data={
        "Importance": xgb_pipeline.named_steps[
            'xgbregressor'
            ].feature_importances_,
        },
        index=features_transformed).sort_values(
            by="Importance", ascending=False
            )
    feat_imp_xgb_df['rank'] = feat_imp_xgb_df[
        'Importance'
        ].rank(ascending=False)
    feat_imp_xgb_df['model'] = 'XGBRegressor'

    top_feat_ranking = pd.DataFrame(
            {'Rank': feat_imp_lgbm_df[:20]['rank'].to_list(),
            'LGBMRegressor': feat_imp_lgbm_df[:20].index.to_list(),
            'XGBRegressor': feat_imp_xgb_df[:20].index.to_list()
            }
            ).set_index('Rank')

    # Save the results 
    output_results_dict = output_dir + '/results_dict.pkl'
    output_results_df_png = output_dir + '/04_linear_nonlinear_results_df.png'
    output_scores_df_png = output_dir + '/05_scores_df.png'
    output_top_feat_png = output_dir + '/06_top_feat_df.png'
    output_top_feat_df = output_dir + '/top_feat_df.csv'
    
    try:
        with open(output_results_dict, 'wb') as f:
            pickle.dump(results, f)
        dfi.export(results_df, output_results_df_png, dpi=200)
        dfi.export(scores_df, output_scores_df_png, dpi=200)
        dfi.export(top_feat_ranking, output_top_feat_png, dpi=200)
        top_feat_ranking.to_csv(output_top_feat_df, index = False)

    except:
        os.makedirs(os.path.dirname(output_results_dict))
        with open(output_results_dict, 'wb') as f:
            pickle.dump(results, f)
        dfi.export(results_df, output_results_df_png, dpi=200)
        dfi.export(scores_df, output_scores_df_png, dpi=200)
        dfi.export(top_feat_ranking, output_top_feat_png, dpi=200)
        top_feat_ranking.to_csv(output_top_feat_df, index = False)

if __name__ == "__main__":
    main(opt["--output_dir"], opt["--input_file_path"])   
    