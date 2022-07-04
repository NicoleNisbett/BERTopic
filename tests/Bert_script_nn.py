# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import gc

# processing data 
def process_data(data):
    df=pd.read_csv(data)
    dfEN = df[df.lang == 'en']
    tweets = dfEN.text
    tweetlist=tweets.to_list()
    return (tweets, tweetlist)

tweets1, tweetslist1 = process_data("/nobackup/ipinni/tweetsCOP20.csv")

from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=10)

#tweets2, tweetslist2 = process_data("/nobackup/ipivs/tweetsCOP21_BB2.csv")
#tweets3, tweetslist3 = process_data("/nobackup/ipivs/tweetsCOP22_BB3.csv")

# concatenate lists if several csv files
#alltweets = tweetlist1 + tweetlist2 + tweetslist3

# concatenate dataframes if several csv files
#alltweetsdf = pd.concat([tweets1, tweets2, tweets3])

# deleting data objects no longer needed
#del [[tweets1,tweets2, tweets3, tweetslist1, tweetslist2, tweetslist3]]
#gc.collect()

# do the BERT topic modelling
from bertopic import BERTopic
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings

glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])
topic_model_fl = BERTopic(embedding_model=document_glove_embeddings, nr_topics = 'auto', min_topic_size=50, top_n_words=10, vectorizer_model=vectorizer_model).fit(tweetslist1)

#topic_model = BERTopic(embedding_model = "distilbert-base-nli-stsb-mean-tokens", nr_topics = 'auto', min_topic_size = 50, top_n_words=10).fit(tweetslist1)

# save modelling output
topic_model_fl.save("COP20_Bert_model_flair")

# calculate highest topic probability for each data entry and save it to a csv file 
topics, probs = topic_model_fl.fit_transform(tweetslist1)
dftopics = pd.DataFrame(topics, columns=['topics'])
dfprobs = pd.DataFrame(probs, columns=['probs'])
outputdf = pd.concat([tweets1.reset_index(drop=True),dftopics.reset_index(drop=True), dfprobs.reset_index(drop=True)], axis=1)
outputdf.to_csv('COP20Output.csv', index=False)

#alternative method of saving the list objects
import pickle
with open('/nobackup/ipinni/COP20topics.list', 'wb') as config_list_file:
 pickle.dump(topics, config_list_file)

import numpy as np
np.save('/nobackup/ipinni/COP20probs.npy', probs)    
  
  
  
