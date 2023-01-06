# Import libraries
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    PolynomialFeatures
)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer


def preprocessor():
    """"
    Returns a column trnasformer object with below steps:
    1. OneHotEncoding for categorical features
    2. OrdinalEncoding for ordinal features
    3. Polynomial transformation and then standard scaler for numerical features
    4. CountVectorizer for text features like the name, description, and amenities
    5. OneHotEncoding for binary features
    
    Parameters
    ----------
    NONE

    Returns
    ----------
        A sklearn column trnasformer object
    """
    # Create feature lists
    categorical_features = ['neighbourhood_cleansed', 'room_type']
    binary_features = ['host_is_superhost', 'instant_bookable']
    ordinal_features = ['name_sent']
    sentiment_levels = ['Positive', 'Neutral', 'Negative']
    numeric_features = ['host_listings_count', 'accommodates', 'price', 'minimum_nights', 
                        'minimum_nights', 'number_of_reviews', 'time_diff', 'review_scores_rating']
    name_feature = 'name'
    description_feature = 'description'
    amenities_feature = 'amenities'
    drop_features = ['id', 'latitude', 'longitude', 'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm']
    
    # Create transformers
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    binary_trnasformer = OneHotEncoder(drop="if_binary")
    ordinal_transformer = OrdinalEncoder(categories=[sentiment_levels],
                                         dtype=int, 
                                         handle_unknown='use_encoded_value', 
                                         unknown_value=-1)
    numeric_transformer = make_pipeline(PolynomialFeatures(degree=2), StandardScaler())
    text_transformer = CountVectorizer(stop_words='english', max_features=500)
    
    # Create the preprocessor by combining the list of features and transformers
    preprocessor = make_column_transformer(
        (categorical_transformer, categorical_features),
        (binary_trnasformer, binary_features),
        (ordinal_transformer, ordinal_features),
        (numeric_transformer, numeric_features),
        (text_transformer, name_feature),
        (text_transformer, description_feature),
        (text_transformer, amenities_feature),
        ('drop', drop_features)
        )
    return preprocessor