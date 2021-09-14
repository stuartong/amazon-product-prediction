#This file is designed to have all the functions for preprocess data in both the product df and the review data frames
#It will act as the library to import from functions needed in the preprocess_data.py
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

################################################################################################
##PRODUCT DF FUNCTIONS
################################################################################################
def clean_categories_column(df):
    """clean_categories_column takes in a dataframe and return the categories
    per item that occur 500 or more times in the full product space

    Clean_categories_column starts by generating a total category list
    of all categories tagged to every product
    then it counts the occurrance of every cat in the list and extracts only the list 
    with 500 occurrences or more.
    finally it updates the category column to only include these categories per product to
    help with feature selection/ generation.

    Args:
        df (data_frame): data_frame that includes "category" labelled column

    Returns:
        df: same data_frame with the category column updated
    """
    #extracting the most common
    print("Extracting total category list")
    total_cat_lst= [word.lower() for lst in df["category"] for word in lst]
    print("Extracting categories with at least 500 occurrences")
    cat_counter_dict= dict(Counter(total_cat_lst))
    cat_500= [cat for cat in cat_counter_dict if cat_counter_dict[cat]>=500]
    df["category"]= df["category"].apply(lambda row : [word.lower() for word in row if word in cat_500])
    print("category column successfully processed!")
    return df

def clean_text(sent):
    """takes a string and returns a list of lemmatized words that are not in the NLTK english stopwords
    Args:
        sent (str|list): string or list of strings
    Returns:
        list: list of lemmatized version of the original text
    """
    import re
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    stop_words= stopwords.words("english") 
    lemmatizer= WordNetLemmatizer()
    
    if type(sent) == list:
        # print("dtype is list of strings")
        text= " ".join(sent)
        extracted_words=  re.findall(r'(?:\\+[\']?t|\\+n|<[^>]+>|[-]{2,}|&amp|https?://[^\s]+)|(\d+[,]\d+ ?[xX]? ?\d+[,]\d+|[a-zA-Z0-9-/.]+)', string= text)
        filtered_words= [word.lower() for word in extracted_words if (word not in stop_words) & (len(word)>1)]
        lemmatized_words= [lemmatizer.lemmatize(word) for word in filtered_words]
        return lemmatized_words
    elif type(sent) == str:
        # print("dtype is string")
        extracted_words=  re.findall(r'(?:\\+[\']?t|\\+n|<[^>]+>|[-]{2,}|&amp|https?://[^\s]+)|(\d+[,]\d+ ?[xX]? ?\d+[,]\d+|[a-zA-Z0-9-/.]+)', string= sent)
        filtered_words= [word.lower() for word in extracted_words if (word not in stop_words) & (len(word)>1)]
        lemmatized_words= [lemmatizer.lemmatize(word) for word in filtered_words]
        return lemmatized_words

def quantify_features(df):
    """quantify_features returns a df with a new column 'quantified_features_array' where it grants a binary classification of availability or absence of tech1, tech2 and calculates number of images per products and words in description


    Args:
        df (dataframe): target dataframe

    Returns:
        df: dataframe with a new 'quantified_features_array' column
    """
    # creating binary columns for tech1 and tech2 where 1 is given if it contains data else 0 
    df["len_tech1"], df["len_tech2"]= [np.where(df[tech].map(len) >0, 1, 0) for tech in ["tech1", "tech2"]]
    #create column to count how many images are provided for a product
    df["num_image"]= df["imageURLHighRes"].map(len)
    #counting the number of words in description column
    df["descrption_num"]= df["description"].map(len)
    df["quantified_features_array"]= [np.array([float(len_tech1), float(len_tech2), float(num_img), float(desc_num)]) for len_tech1, len_tech2, num_img, desc_num in zip(df["len_tech1"], df["len_tech2"], df["num_image"], df["descrption_num"])]
    return df

def consolidate_text_columns(df):
    """consolidate_text_columns takes the df and combines vectorized text from category, description, brand, feature, title
    columns into one consolidated_text column and extract alphanumeric characters only

    Args:
        df (dataframe): target dataframe

    Returns:
        df (dataframe): updated dataframe with full_feature column
    """
    df["consolidated_text_column"]= (df["category"]+ df["description"]+ df["brand"].str.split()+ df["feature"] +df["title"]).map(pd.unique)
    df.drop(columns= ["category", "description", "brand", "feature", "title"], inplace= True)
    print("Consolidated vectorized text column created")
    return df


############################################################################################################################################
#REVIEW DF FUNCTIONS
############################################################################################################################################
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
        # print('dtype is string')
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
    df["reviewText"]= df["reviewText"].apply(lambda row: row if isinstance(row, list) else [])
    df["summary"]= df["summary"].apply(lambda row: row if isinstance(row, list) else [])
    df['full_review_features']= (df['reviewText']+ df['summary']).map(pd.unique)
    df.drop(columns= ['reviewText', 'summary'], inplace= True)
    return df