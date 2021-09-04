import pandas as pd 
import numpy as np

'''
    This function uses the split data (train, validate, test)
    and turn the overall rating to sentiment values.
    Convert the sentiment values to sentiment statements
    whether they are positive, neutral, or negative.
    
    Returns each split data with the new sentiment statements
    as a new column. 
    
    Inputs:
    train, validate, test
    
    Outputs:
    train, validate, test
'''


def sentiment_analysis(rating):
    if (rating == 5) or (rating == 4):
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    elif (rating == 2) or (rating == 1):
        return 'Negative'

def apply_sentiment(train, validate, test):    
    # Add sentiments to the data, both our train and test data
    train['sentiment'] = train['df.overall'].apply(sentiment_analysis)
    validate['sentiment'] = validate['df.overall'].apply(sentiment_analysis)
    test['sentiment'] = test['df.overall'].apply(sentiment_analysis)
    
    return train, validate, test
