# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#rclone copy onedrive:UKRI_Tweet_Data/script/Bert_script_nn2.py .

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from umap import UMAP
import gc
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import pickle
import numpy as np
#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


# processing data 
def process_data(data):
    df=pd.read_csv(data)
    dfEN = df[df.lang == 'en']
    #result = dfEN.loc[dfEN["sourcetweet_type"]=="retweeted"].copy()
    result2 = dfEN.loc[dfEN["sourcetweet_type"]!="retweeted"].copy()
    #tweets1 = result["sourcetweet_text"].to_list()
    result3 = result2[['tweet_id',"text", "like_count", "retweet_count"]].copy()
    tweets = result2["text"].to_list()
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweetlist=[]
    for i in range(0,len(tweets)):
            tweetlist.append(p.clean(tweets[i]))
    return (tweetlist, result3)

tweetslist1, tweets1 = process_data("tweetsCOP27_1.1.csv")
tweetslist2, tweets2 = process_data("tweetsCOP27_1.2.csv")
tweetslist3, tweets3 = process_data("tweetsCOP27_1.3.csv")
tweetslist4, tweets4 = process_data("tweetsCOP27_2.1.csv")
tweetslist5, tweets5 = process_data("tweetsCOP27_2.2.csv")
tweetslist6, tweets6 = process_data("tweetsCOP27_2.3.csv")
tweetslist7, tweets7 = process_data("tweetsCOP27_2.4.csv")



#join DFs of tweets and save
all = tweets1.append([tweets1, tweets2, tweets3, tweets4, tweets5, tweets6, tweets7])
all.to_csv("COP27CleanTweets.csv")

# concatenate lists if several csv files
alltweets = tweetslist1 + tweetslist2 + tweetslist3 + tweetslist4 + tweetslist5 + tweetslist6 + tweetslist7 

print(len(alltweets))

# deleting data objects no longer needed
del [[tweetslist1, tweetslist2, tweetslist3, tweetslist4, tweetslist5, tweetslist6, tweetslist7]]
del [[tweets1, tweets2, tweets3,tweets4, tweets5, tweets6, tweets7]]
gc.collect()

# preprocess and clean data
my_stopwords = frozenset(list(["rt","RT", "&", "amp", "&amp", "http","https", "http://", "https://", "fav", "FAV"]))
#new_stopwords = frozenset(list(text.ENGLISH_STOP_WORDS) + my_stopwords)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words = my_stopwords, min_df=20)

# do the BERT topic modelling

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

min_clusters = round(len(alltweets) * 0.0017)
hdbscan_model = HDBSCAN(min_cluster_size= min_clusters, metric='euclidean', cluster_selection_method='eom', prediction_data=True, min_samples=5)

sentence_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = sentence_model.encode(alltweets)

#run the model
topic_model = BERTopic(nr_topics = 'auto', umap_model=umap_model, hdbscan_model=hdbscan_model, embedding_model = sentence_model,vectorizer_model=vectorizer_model, low_memory=True, calculate_probabilities=True)

topics, probs = topic_model.fit_transform(alltweets, embeddings)

print("done Bert modelling")

# save modelling output without umap to save space
umap_model = topic_model.umap_model
topic_model.umap_model = None
topic_model.save("COP27_Bert_model", save_embedding_model=False)

#alternative method of saving the list objects
with open('COP27topics.list', 'wb') as config_list_file:
 pickle.dump(topics, config_list_file)

with open('COP27docs.list', 'wb') as doc_list_file:
 pickle.dump(alltweets, doc_list_file)  

np.save('COP27probs.npy', probs)   

index_outliers = []
for i,j in enumerate(topics):
    if j == -1:
        index_outliers.append(i)

outlier_tweets = []
for index in index_outliers:
    outlier_tweets.append(alltweets[index])

with open('COP27outliers.list', 'wb') as outlier_list_file:
 pickle.dump(outlier_tweets, outlier_list_file)   

