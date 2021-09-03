import pandas as pd
import numpy as np


def split_data(df,train_size):
    '''
    General function takes a df, shuffles and splits the data into 3 sets:
    1. Train set - based on train_size
    2. Validation Set - same size as test set
    3. Test Set - same size as validation set

    Input: Clean and tokenized dataframes, size of training set
    Output: train_df, validation_df, test_df
    '''

    # get test and validation set size
    test_size = ((1-train_size)/2)+train_size

    train, validate, test = np.split(df.sample(frac=1,
                                               random_state=42),
                                    [int(train_size*len(df)), int(test_size*len(df))]
                                    )
    return train,validate,test





