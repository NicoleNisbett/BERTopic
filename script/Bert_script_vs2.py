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


# processing data 
def process_data(data):
    df=pd.read_csv(data)
    dfEN = df[df.lang == 'en']
    result2 = dfEN.loc[dfEN["sourcetweet_type"]!="retweeted"].copy()
    result3 = result2[['tweet_id',"text", "like_count", "retweet_count"]].copy()
    tweets = result2["text"].to_list()
    timestamps = result2["dates"].to_list()
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweetlist=[]
    for i in range(0,len(tweets)):
            tweetlist.append(p.clean(tweets[i]))
    return (tweetlist, result3, timestamps)

tweetslist1, tweets1, timestamps1 = process_data("/nobackup/ipivs/tweetsFFFMar2019.csv")
tweetslist2, tweets2, timestamps2 = process_data("/nobackup/ipivs/tweetsFFFMay2019.csv")
tweetslist3, tweets3, timestamps3 = process_data("/nobackup/ipivs/tweetsFFF20Sept2019.csv")
tweetslist4, tweets4, timestamps4 = process_data("/nobackup/ipivs/tweetsFFF27Sept2019.csv")
tweetslist5, tweets5, timestamps5 = process_data("/nobackup/ipivs/tweetsFFF29Nov2019.csv")

#join DFs of tweets and save
all = tweets1.append([tweets2, tweets3, tweets4, tweets5])
all.to_csv("FFF2019CleanTweets.csv")

# concatenate lists if several csv files
alltweets = tweetslist1 + tweetslist2 + tweetslist3 + tweetslist4 + tweetslist5
alldates = timestamps1 + timestamps2 + timestamps3 + timestamps4 + timestamps5

print(len(alltweets))

# deleting data objects no longer needed
del [[tweetslist1, tweetslist2, tweetslist3, tweetslist4, tweetslist5]]
del [[tweets1, tweets2, tweets3, tweets4, tweets5]]
del [[timestamps1, timestamps2, timestamps3, timestamps4, timestamps5]]
gc.collect()

# preprocess and clean data
my_stopwords = frozenset(list(["rt","RT", "&", "amp", "&amp", "http","https", "http://", "https://", "fav", "FAV"]))
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
topic_model.save("FFF2019_Bert_model", save_embedding_model=False)

#alternative method of saving the list objects
with open('/nobackup/ipivs/FFF2019topics.list', 'wb') as config_list_file:
 pickle.dump(topics, config_list_file)

with open('/nobackup/ipivs/FFF2019docs.list', 'wb') as doc_list_file:
 pickle.dump(alltweets, doc_list_file)

with open('/nobackup/ipivs/FFF2019dates.list', 'wb') as doc_list_file:
 pickle.dump(alldates, doc_list_file)

np.save('/nobackup/ipivs/FFF2019probs.npy', probs)   

