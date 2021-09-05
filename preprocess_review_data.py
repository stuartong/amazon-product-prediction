import pandas as pd
import numpy as np

def clean_review_column(df):
    """
    clean_review_text_column function takes in a dataframe, dropping any reviewText duplicates
    
    Args:
        df (data_frame): review data_frame 
    Returns:
        df: cleaned version of the dataframe
    """
    
    #dropping any duplicates
    print('Dropping any duplicates in the reviewText column')
    df = df.drop_duplicates(subset=['reviewText'])
    
    #filtering any unverified reviews out
    # print('Let us filter the unverified reviews out')
    # df = df[df['verified']==True]
    return df

def clean_review(sent):
    """
    takes a string 
    - remove punctuations
    - lowercase all words
    - tokenize
    - lemmatize
    - remove stopwords
    
    and returns a list of lemmatized words that are not in the NLTK english stopwords
    
    Args:
        sent (str|list): string or list of strings
    Returns:
        list: list of lemmatized version of the original text
    """
    
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stop_words= stopwords.words('english')
    lemmatizer= WordNetLemmatizer()

    if type(sent) == list:
        text= ' '.join(sent)
        extracted_words=  re.findall(r'(\w+|\d\'\d)', string=text)
        filtered_words= [word.lower() for word in extracted_words if (word not in stop_words) & (len(word)>1)]
        lemmatized_words= [lemmatizer.lemmatize(word) for word in filtered_words]
        return lemmatized_words
    elif type(sent) == str:
        print('dtype is string')
        extracted_words=  re.findall(r"(\w+|\d\'\d)", string=sent) #FIX THIS, if want to capture 5'6" for example
        filtered_words= [word.lower() for word in extracted_words if (word not in stop_words) & (len(word)>1)]
        lemmatized_words= [lemmatizer.lemmatize(word) for word in filtered_words]
        return lemmatized_words
    
def create_full_feature_review(df):
    """
    create_full_feature_review takes the review df,
    and combines reviewText, summary columns
    into one full_review_features column and extract alphanumeric characters only
    
    Args:
        df (dataframe): target dataframe
    Returns:
        df (dataframe): updated dataframe with full_review_features column
    """
    
    df['full_review_features']= (df['reviewText']+ df['summary']).map(pd.unique)
    df.drop(columns= ['reviewText', 'summary'], inplace= True)
    return df