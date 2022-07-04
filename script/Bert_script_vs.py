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
#import os
#os.environ["TOKENIZERS_PARALLELISM"] = "false"


# processing data 
def process_data(data):
    df=pd.read_csv(data)
    dfEN = df[df.lang == 'en']
    result = dfEN.loc[dfEN["sourcetweet_type"]=="retweeted"].copy()
    result2 = dfEN.loc[dfEN["sourcetweet_type"]!="retweeted"].copy()
    tweets1 = result["sourcetweet_text"].to_list()
    tweets2 = result2["text"].to_list()
    tweets = tweets1 + tweets2
    p.set_options(p.OPT.URL, p.OPT.RESERVED)
    tweetlist=[]
    for i in range(0,len(tweets)):
            tweetlist.append(p.clean(tweets[i]))
    return (tweetlist)

tweetslist1 = process_data("/nobackup/ipivs/tweetsCOP21_BB1.csv")
tweetslist2 = process_data("/nobackup/ipivs/tweetsCOP21_BB2.csv")
tweetslist3 = process_data("/nobackup/ipivs/tweetsCOP21_BB3.csv")

# concatenate lists if several csv files
alltweets = tweetslist1 + tweetslist2 + tweetslist3

print(len(alltweets))

# deleting data objects no longer needed
del [[tweetslist1, tweetslist2, tweetslist3]]
gc.collect()

# preprocess and clean data
my_stopwords = frozenset(list(["rt","RT", "&", "amp", "&amp", "http","https", "http://", "https://", "fav", "FAV"]))
#new_stopwords = frozenset(list(text.ENGLISH_STOP_WORDS) + my_stopwords)
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words = my_stopwords, min_df=20)

# do the BERT topic modelling

#umap_model = UMAP(n_neighbors=60, n_components=10, metric='cosine', low_memory=True)

min_clusters = round(len(alltweets) * 0.0017)
hdbscan_model = HDBSCAN(min_cluster_size= min_clusters, metric='euclidean', cluster_selection_method='eom', prediction_data=True, min_samples=5)

sentence_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = sentence_model.encode(alltweets)

#run the model
topic_model = BERTopic(nr_topics = 'auto', hdbscan_model=hdbscan_model, verbose=True, embedding_model = sentence_model,vectorizer_model=vectorizer_model, low_memory=True, calculate_probabilities=False)

topics = topic_model.fit_transform(alltweets, embeddings)

print("done Bert modelling")

# save modelling output without umap to save space
umap_model = topic_model.umap_model
topic_model.umap_model = None
topic_model.save("COP21_Bert_model", save_embedding_model=False)

#alternative method of saving the list objects
with open('/nobackup/ipivs/COP21topics.list', 'wb') as config_list_file:
 pickle.dump(topics, config_list_file)

with open('/nobackup/ipivs/COP21docs.list', 'wb') as doc_list_file:
 pickle.dump(alltweets, doc_list_file)   

index_outliers = []
for i,j in enumerate(topics):
    if j == -1:
        index_outliers.append(i)

outlier_tweets = []
for index in index_outliers:
    outlier_tweets.append(alltweets[index])

with open('/nobackup/ipivs/COP21outliers.list', 'wb') as outlier_list_file:
 pickle.dump(outlier_tweets, outlier_list_file)   


'''
# do the BERT topic modelling now only on the outliers 
min_clusters2 = round(len(outlier_tweets) * 0.0017)

hdbscan_model2 = HDBSCAN(min_cluster_size= min_clusters2, metric='euclidean', cluster_selection_method='eom', prediction_data=True, min_samples=5)

embeddings2 = sentence_model.encode(outlier_tweets)

topic_model2 = BERTopic(nr_topics = 'auto', hdbscan_model=hdbscan_model2, embedding_model = sentence_model,vectorizer_model=vectorizer_model, low_memory=True, calculate_probabilities=False)

topics2 = topic_model2.fit_transform(outlier_tweets, embeddings2)

topic_model2.save("COP21_Bert_model_outliers", save_embedding_model=False)

with open('/nobackup/ipinni/COP21topicsoutliers.list', 'wb') as config_list_file2:
 pickle.dump(topics2, config_list_file2)

with open('/nobackup/ipinni/COP21outliers.list', 'wb') as doc_list_file2:
 pickle.dump(outlier_tweets, doc_list_file2)

'''