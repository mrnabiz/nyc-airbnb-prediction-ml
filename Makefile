# Makefile
# Date: 2023-01-14
# Makefile written by Mohammad Reza Nabizadeh

#Run this Make file using your CLI from the root of the project directory
#Example Commands:

#Run all scripts in sequence:
# Make all

#Delete all results:
# Make clean

#Run an individual script:
#Use script names below. Assumes dependencies have already been run. 

#all the major output files
all:	data/raw/raw_df.csv data/raw/clean_df.csv data/preprocessed/preprocessed_df.csv results/eda_plots/corr_matrix.png results/eda_plots/corr_plot.png results/eda_plots/missing_values_plot.png results/eda_plots/numeric_hist_plot.png results/eda_plots/trans_target.png results/eda_plots/untrans_target.png data/preprocessed/feat_eng_df.csv results/model_results/results_dict.pkl results/model_results/scores_dict.pkl results/model_results/01_baseline_results_df.png results/model_results/lr_coeffs_df.csv results/model_results/02_baseline_coeff_plt.png results/model_results/03_ridge_coeff_plt.png results/model_results/results_dict.pkl results/model_results/04_linear_nonlinear_results_df.png results/model_results/05_scores_df.png results/model_results/06_top_feat_df.png results/model_results/top_feat_df.csv results/model_results/lgbm_model.joblib results/model_result/07_perm_imp_plt.png results/model_results/08_shap_sum_plt.png

# Download the raw data
data/raw/raw_df.csv:	src/data_wrangling/pull_data.py	
	python src/data_wrangling/pull_data.py --file_path="data/raw/raw_df.csv" --url="http://data.insideairbnb.com/united-kingdom/england/london/2022-09-10/data/listings.csv.gz"
    
# Clean the raw data frame to remove unnecessary columns
data/raw/clean_df.csv:	src/data_wrangling/clean_data.py data/raw/raw_df.csv
	python src/data_wrangling/clean_data.py --output_file_path="data/raw/clean_df.csv" --input_file_path="data/raw/raw_df.csv"

# Preprocesse the cleaned to prepare it for the model building by running
data/preprocessed/preprocessed_df.csv:	src/data_wrangling/preprocessing.py data/raw/clean_df.csv
	python src/data_wrangling/preprocessing.py --output_file_path="data/preprocessed/preprocessed_df.csv" --input_file_path="data/raw/clean_df.csv"

# Run the EDA analysis
results/eda_plots/corr_matrix.png results/eda_plots/corr_plot.png results/eda_plots/missing_values_plot.png results/eda_plots/numeric_hist_plot.png results/eda_plots/trans_target.png results/eda_plots/untrans_target.png: src/model_building/eda.py data/raw/clean_df.csv data/preprocessed/preprocessed_df.csv
	python src/model_building/eda.py --output_dir="results/eda_plots" --clean_df_file_path="data/raw/clean_df.csv"  --preprocessed_df_file_path="data/preprocessed/preprocessed_df.csv"

# Run the feature engineering
data/preprocessed/feat_eng_df.csv: src/model_building/feat_eng.py data/preprocessed/preprocessed_df.csv
	python src/model_building/feat_eng.py --output_file_path="data/preprocessed/feat_eng_df.csv" --input_file_path="data/preprocessed/preprocessed_df.csv"

# Run the linear models by running
results/model_results/results_dict.pkl results/model_results/scores_dict.pkl results/model_results/01_baseline_results_df.png results/model_results/lr_coeffs_df.csv results/model_results/02_baseline_coeff_plt.png results/model_results/03_ridge_coeff_plt.png: src/model_building/linear_models.py data/preprocessed/feat_eng_df.csv
	python src/model_building/linear_models.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

# Run the non linear models (ensembles)
results/model_results/04_linear_nonlinear_results_df.png results/model_results/05_scores_df.png results/model_results/06_top_feat_df.png results/model_results/top_feat_df.csv results/model_results/lgbm_model.joblib: src/model_building/ensembles.py data/preprocessed/feat_eng_df.csv
    python src/model_building/ensembles.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

# Run the permutation feature importances analysis
results/model_result/07_perm_imp_plt.png: src/model_building/perm_feat_importances.py data/preprocessed/feat_eng_df.csv
    python src/model_building/perm_feat_importances.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"

# Run the SHAP analysis by running
results/model_results/08_shap_sum_plt.png: src/model_building/shap_feat_importances.py data/preprocessed/feat_eng_df.csv
    python src/model_building/shap_feat_importances.py --output_dir="results/model_results" --input_file_path="data/preprocessed/feat_eng_df.csv"


clean: 
	rm -rf data
	rm -rf results