# Amazon Product Success Prediction

## Overview
The purpose of this repo is to consolidate and track code related to our Milestone 2 Project.

We plan to accomplish 3 main things:
1. Predict the likelihood of success of a product (defined by a metric related to total/average number of stars or salesrank) from product metadata as features
2. Feature Engineering - Use topic modeling and NLP techniques to create new target variables that represent keywords/sentiment from review text as new target variables to see if we can predict and enable a merchant to understand how buyers on the platform would react to a newly launch product with similar features
3. Clustering - We intend to use clustering to improve product search enabling customer to search for products with similar features and potentially use methods like DBScan for outlier detection to see if we can pick up fake reviews

## Key Links
- [Full Proposal](https://docs.google.com/document/d/1qO5qy0LVd6yYzDum-VQhvQ-wNo1fDjyhly71D3Y5NtQ/edit)
- [Data Set](http://deepyeti.ucsd.edu/jianmo/amazon/index.html)


## Pipeline
!
### 1. load_data
In this first stage of our pipeline, we take the metadata and the reviews data directly from the category URL (in this case :
‘Appliances’, and we parse, clean, and convert those into Pandas dataframes. Readers may be able to load different categories
than ‘Appliances’ by simply replacing the URL in this Python file to the desired category’s URL.


### 2. preprocess_data
Once the data frame was processed thoroughly through previous functions, we designed this stage to turn the raw text data
into a machine readable vectorized feature space that can be introduced to different machine learning algorithms. We do
multiple preprocessing steps such as further cleaning, lemmatizing, removing stopwords, vectorizing every product/review
features.
We would then save this data in a pickle format as an output from this pipeline stage to be passed on to both our supervised
and unsupervised models.
Additionally there were aggregation on different features such as individual word weights using pre-trained word2vec
models, tf-idf representations, and descriptive statistics of product/review in multiple forms(length of text, number of
tables/images describing a product, and whether a review is verified or not).


### 3. Success Metric Creation
Scoring each review with a point system ranging from -2 to 2, there is also an option to remove identified fake reviews

### 4. Supervised Model 1
Train algorithms to predict a product’s success rate on combinations of word embeddings, handcrafted features, tf-idf vectors,
cluster labels.

### 5. Supervised Model 2
Train algorithms to identify potential fake reviews based on review word embeddings, tfidf, length of reviews, whether a
review is from a verified user. The aim is to test whether fake reviews detection and elimination would improve product
success predictions.

### 6. Evaluate Supervised Models
Reporting the train-val-test and dummy accuracy scores, F1 scores, correlation coefficient. Confusion matrix, ROC curve,
and Precision Recall curve is also provided.

### 7. Clustering and Evaluation
Clustering reviews data to identify trends/insights/anomalies, display graphs, compute the optimal parameter for each
clustering model, and provide silhouette/calinski harabasz/davies bouldin scores.
