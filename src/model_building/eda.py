# author: Mohammad Reza Nabizadeh
# date: 2023-01-05

"""
This script runs the EDA analysis over the cleaned data and the 
preprocessed data.

Usage: eda.py --output_dir=<output_dir> --clean_df_file_path=<clean_df_file_path> --preprocessed_df_file_path=<preprocessed_df_file_path>
 
Options:
--output_dir=<output_dir>
Path to folder where the EDA results will be saved, `in quotes.

--clean_df_file_path=<clean_df_file_path>
Path to folder and file name where the clean df is stored, including 
the name, `in quotes.

--preprocessed_df_file_path=<preprocessed_df_file_path>
Path to folder and file name where the preprocessed df is stored, 
including the name, `in quotes.

"""

# Import dependencies
import numpy as np
import pandas as pd
from docopt import docopt
import os
import altair as alt
from utils import save_chart
import dataframe_image as dfi
from sklearn.model_selection import train_test_split

# Handle large data sets without embedding them in the notebook
alt.data_transformers.disable_max_rows()

opt = docopt(__doc__)


def main(output_dir, clean_df_file_path, preprocessed_df_file_path):
    # Read the clean df csv file
    clean_full_df = pd.read_csv(clean_df_file_path, low_memory=False)

    # Read the preprocessed df csv file
    preprocessed_full_df = pd.read_csv(preprocessed_df_file_path, 
                                       low_memory=False)

    # A random split of the data to decrease the analysis time
    clean_df, clean_df_remain = train_test_split(clean_full_df, 
                                                 test_size=0.95, 
                                                 random_state=123
                                                 )
    preprocessed_df, preprocessed_df_remain = train_test_split(
                                                preprocessed_full_df, 
                                                test_size=0.95, 
                                                random_state=123
                                                )

    y_train_untrans = preprocessed_df['reviews_per_month']
    y_train_transformed = np.log10(y_train_untrans)

    # Produce the missing values plot using AltAir mark_rect plots.
    missing_values_plot = alt.Chart(
        clean_df.sort_values(
            'host_since',
            ignore_index=True).isna().reset_index().melt(id_vars='index')
                                    ).mark_rect().encode(
                                        alt.X('index:O', axis=None),
                                        alt.Y('variable', title=None),
                                        alt.Color('value', title='NaN'),
                                        alt.Stroke('value')
                                    ).properties(
                                        width=800,
    title="The accurence of Null values in the dataset sorted based on the \
           host joining date"
                                    ).configure_title(
                                        fontSize=20,
                                        font='Cambria',
                                        anchor='start'
                                    ).configure_axis(labelFont='Cambria'
                                    )
    
    # Produce the distribution of the numeric columns
    numeric_cols = list(preprocessed_df.select_dtypes(include='number'))
    numeric_cols.remove('id')
    
    numeric_cols_hist = alt.Chart(preprocessed_df).mark_bar().encode(
        alt.X(alt.repeat(), bin=alt.Bin(maxbins=30)),
        alt.Y('count()', axis=alt.Axis(title='Count'), stack=False)
        ).properties(
            width=250,
            height=150
        ).repeat(repeat=numeric_cols, columns = 3).properties(
        title='Histogram chart of the AirBnB dataset per each numeric feature'
        ).configure_title(
                fontSize=20,
                font='Cambria',
                anchor='start'
        ).configure_axis(labelFont='Cambria'
        )

    # Build the correlation matrix
    corr_df = preprocessed_df.drop(['id'], 
                                   axis=1).corr('spearman'
                                                ).style.background_gradient()

    # Correlation of the numeric feature with the correlation above 0.1
    numeric_corr_cols = ['host_listings_count', 'price', 'minimum_nights', 
                         'number_of_reviews', 'reviews_per_month']

    numeric_cols_corr = alt.Chart(preprocessed_df
        ).mark_circle(opacity=0.3
        ).encode(
            alt.X(alt.repeat("column"),
                  type='quantitative',
                  scale=alt.Scale(zero=False)
                  ),
            alt.Y(alt.repeat("row"),
                  type='quantitative',
                  scale=alt.Scale(zero=False)
                  )
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=numeric_corr_cols,
            column=numeric_corr_cols
        ).properties(
            title="Pair-wise correlation of numeric columns in AirBnb dataset"
        ).configure_title(
            fontSize=20,
            font='Cambria',
            anchor='start'
        )

    # Untransformed distribution of the target
    untrans_target_dist = alt.Chart(pd.DataFrame(y_train_untrans)
        ).mark_bar().encode(
        alt.X('reviews_per_month', bin=alt.Bin(maxbins=30)),
        alt.Y('count()', title='Count')
        ).properties(
            width=250,
            height=250).properties(
                title='Histogram chart of the AirBnB dataset reviews per month'
        ).configure_title(fontSize=20,
        font='Cambria', anchor='start').configure_axis(labelFont='Cambria'
        )

    # Transformed distribution of the target
    trans_target_dist = alt.Chart(pd.DataFrame(y_train_transformed)
        ).mark_bar().encode(
            alt.X('reviews_per_month', bin=alt.Bin(maxbins=30)),
            alt.Y('count()', title='Count')
        ).properties(
            width=250,
            height=250
        ).properties(
    title='Histogram chart of the AirBnB dataset transformed reviews per month'
        ).configure_title(fontSize=20,
                          font='Cambria',
                          anchor='start'
        ).configure_axis(labelFont='Cambria'
        )


    # Build the savable plots and matrixes address
    output_missing_png = output_dir + '/missing_values_plot.png'
    output_hist_png = output_dir + '/numeric_hist_plot.png'
    output_corr_matrix = output_dir + '/corr_matrix.png'
    output_corr_plot = output_dir + '/corr_plot.png'
    output_untrans_target = output_dir + '/untrans_target.png'
    output_trans_target = output_dir + '/trans_target.png'
    

    # Write the chart png file
    try:
        save_chart(missing_values_plot, output_missing_png, 2)
        save_chart(numeric_cols_hist, output_hist_png, 2)
        dfi.export(corr_df, output_corr_matrix, dpi=200)
        save_chart(numeric_cols_corr, output_corr_plot, 2)
        save_chart(untrans_target_dist, output_untrans_target, 2)
        save_chart(trans_target_dist, output_trans_target, 2)

    except:
        os.makedirs(os.path.dirname(output_missing_png))
        save_chart(missing_values_plot, output_missing_png, 2)
        save_chart(numeric_cols_hist, output_hist_png, 2)
        dfi.export(corr_df, output_corr_matrix, dpi=200)
        save_chart(numeric_cols_corr, output_corr_plot, 2)
        save_chart(untrans_target_dist, output_untrans_target, 2)
        save_chart(trans_target_dist, output_trans_target, 2)

if __name__ == "__main__":
    main(opt["--output_dir"], 
         opt["--clean_df_file_path"], 
         opt["--preprocessed_df_file_path"]
         )