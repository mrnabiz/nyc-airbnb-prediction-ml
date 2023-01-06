import pandas as pd
import vl_convert as vlc
from sklearn.model_selection import cross_validate
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation. Attributted to Varada Kolhatkar
    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data
    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)



def save_chart(chart, filename, scale_factor=1):
    """
    Attributed to: Joel Ostblom
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    if filename.split(".")[-1] == "svg":
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split(".")[-1] == "png":
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

# Adding features by defining functions
def diff_days(last_review_date, host_join_date):
    """
    Returns the number of days between the last review date of the property
    adn the host joining date.

    Parameters:
    ------
    last_review_date: (datetime64[ns])
    the last review date

    host_join_date: (datetime64[ns])
    the host join date

    Returns:
    -------
    the number of days between the last review date of the property
    adn the host joining date: (int)

    """
    diff = last_review_date - host_join_date
    return diff.dt.days.astype('int16')

def get_sentiment(text):
    """
    Returns the compound score of the text and reports its sentiment

    Parameters:
    ------
    test: (str)
    the input text

    Returns:
    -------
    the text sentiment Positive, Negative, or Neutral : (Str)

    """
    sent = str()
    sid = SentimentIntensityAnalyzer()
    if text is None:
        sent = 'Neutral'
    else:
        score = sid.polarity_scores(str(text))['compound']
        if score == 0:
            sent = 'Neutral'
        if score > 0:
            sent = 'Positive'
        if score < 0:
            sent = 'Negative'
    return sent