import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def create_success_metric(review_df,meta_df):
    '''
    This function uses the review_df to calculate success metrics for a product
    and appends the results to the meta_df 
    

    Inputs:
    1. review_df: per category review data
    2. meta_df: per category meta data (uses 'asin' and 'rank' column)

    Outputs:
    1. combined_df: meta_df + success_metric
    '''
    
    # get per product total stars, reviews and average stars 
    summary_df = review_df.groupby('asin').agg(
        tot_stars = ('overall','sum'),
        tot_reviews = ('overall','count'),
        avg_stars = ('overall','mean')
    )

    # get per product no of reviews by star count
    count_df = review_df.groupby(['asin','overall']).size().unstack(fill_value=0)

    # score each product
    count_df['score'] = ((count_df[1.0] * (-2)) +
                         (count_df[2.0] * (-1)) + 
                         (count_df[3.0] * (0)) +
                         (count_df[4.0] * (1)) +
                         (count_df[5.0] * (2))
                         )
    
    # updated meta_df with summary and counts
    combined_df = meta_df.join(summary_df,on='asin')
    combined_df = combined_df.join(count_df,on='asin')

    # get rank
    combined_df['new_rank'] = [' '.join(map(str,l)) for l in combined_df['rank']]
    combined_df['new_rank'] = combined_df['new_rank'].str.replace(',','')
    combined_df['new_rank'] = combined_df['new_rank'].str.extract(r'(\d+)')
    combined_df['new_rank'] = combined_df['new_rank'].astype(float)

    # fill blanks/NaNs (i.e. no reviews etc) with 0
    fill_miss = ['tot_stars','tot_reviews','avg_stars','score','new_rank']
    combined_df[fill_miss] = combined_df[fill_miss].fillna(value=0)

    # Check scores and ranks
    check_score_rank(combined_df)

    return combined_df

def check_score_rank(combined_df):
    '''
    Function to sanity check scores vs rank
    '''

    # create copy of dataframe for required fields
    scatter=combined_df[['main_cat','new_rank','avg_stars','tot_stars','score']].copy()

    # ensure no NaNs
    scatter.fillna(0,inplace=True)

    # get log
    scatter['log_score'] = np.log10(scatter['score'])
    scatter['log_tot_stars'] = np.log10(scatter['tot_stars'])
    scatter['log_new_rank'] = np.log10(scatter['new_rank'])

    # products with zero reviews/neutral rated/negative score overall get log to nan and infinity
    # replace infinity with NaNs and replace NaNs with 0
    scatter.replace(-np.inf,np.nan,inplace=True)
    scatter.fillna(0,inplace=True)

    # products with zero can skew the results
    # create seperate dataframe with no zeros for comparison
    scatter_nozero = scatter[~((scatter['log_new_rank']==0)| (scatter['log_score']==0))]

    # plot charts
    sns.set_style("darkgrid")

    # plot distribution of score vs rank
    sns.scatterplot(data=scatter, x="new_rank", y="score")
    plt.title('Distribution of Scores vs Rank')
    plt.savefig('charts/distribution.png')

    # plot log_distribution of score vs rank
    sns.lmplot(x='log_new_rank',y='log_score',data=scatter)
    plt.title('Log Distribution of Scores vs Rank')
    plt.savefig('charts/log_distribution.png')

    # plot log_distribution of score vs rank - Non-zero
    sns.lmplot(x='log_new_rank',y='log_score',data=scatter_nozero)
    plt.title('Log Distribution of Scores vs Rank - Non-zero')
    plt.savefig('charts/log_distribution(NZ).png')

    # to-do: get R^2 to see fit

    return







