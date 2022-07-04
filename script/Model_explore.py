from bertopic import BERTopic
import pandas as pd
import numpy as np
import pickle    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#use to visualise hierarchy and give understanding of where to reduce. install below packages on spyder and use .show(renderer='png')
#pip install scipy==1.5.1
#pip install plotly==5.5.0
#pip install kaleido

loadPath = "/Users/ipinni/Library/CloudStorage/OneDrive-UniversityofLeeds/UKRI_Tweet_Data/completed/COP22/"

#load model bits
probs = np.load(loadPath +'COP22probs.npy')

with open(loadPath +'COP22topics.list', 'rb') as config_list_file:   topics = pickle.load(config_list_file)

#number of topics
#max(topics_list)+1
    
with open(loadPath +'COP22docs.list', 'rb') as docs_list_file:   docs = pickle.load(docs_list_file)

model = BERTopic.load(loadPath +"COP22_Bert_model", embedding_model = "all-mpnet-base-v2")

representative_docs = model.get_representative_docs()

#explore
model.get_topic_info() 
model.get_params()
#top words in a topic
model.get_topic(1)

model.visualize_hierarchy( orientation = 'bottom').show()
model.visualize_topics().show()

#wordcloud for ALL docs 
def create_wordcloud3(docs):
    text = (" ").join(docs)
    wc = WordCloud(background_color="black", max_words=1000)
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud3(docs_list)

#wordcloud for representative docs of each topic
def create_wordcloud(topic_model, topic):
    text = {word: value for word, value in topic_model.get_representative_docs(topic)}
    wc = WordCloud(background_color="black", max_words=1000, max_font_size=40)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud(model, topic=1)

#create dictionary of all documents in each topic
topic_docs = {topic: [] for topic in set(topics)}
for topic, doc in zip(topics, docs):
    topic_docs[topic].append(doc)

topic_docs.keys()

#wordcloud for all docs in a topic
def create_wordcloud2(topic_docs, topic, avoid):
    text=(" ").join(topic_docs[topic])
    text.replace(avoid, "")
    #text = topic_docs[topic]
    wc = WordCloud(background_color="black", max_words=1000, max_font_size=40)
    wc.generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

create_wordcloud2(topic_docs,3, "COP22")

hashtags = []

for i in topic_docs.keys():   
    tags = re.findall("#([a-zA-Z0-9_]{1,50})", str(topic_docs[i]))
    hashtags.append(tags)

#get sorted list of hashtags in each topic
dict(Counter(hashtags[21]).most_common())

##Bigrams
# from https://towardsdatascience.com/text-analysis-basics-in-python-443282942ec5

my_stopwords = list(["rt","RT", "&", "amp", "&amp", "http","https", "http://", "https://", "fav", "FAV"])
new_stopwords = frozenset(list(text.ENGLISH_STOP_WORDS) + my_stopwords)
#docsDF=pd.DataFrame(docs, columns = ['text'])
c_vec = CountVectorizer(stop_words=new_stopwords, ngram_range=(2,2), min_df = 10)

#get bigrams for a particular topic
def get_bigrams(topic):
    # matrix of ngrams
    ngrams = c_vec.fit_transform(topic_docs[topic]).astype(np.int8)
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    #count_values = np.squeeze(np.asarray(ngrams)).sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_
    df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram'})
    df_ngram = df_ngram[df_ngram.frequency>1]
    df_ngram['topic'] = 'topic' + str(topic)
    return(df_ngram)

topic10 = get_bigrams(10)

topic10.nlargest(20,'frequency').plot(kind='barh', x='bigram', y='frequency').invert_yaxis() 
plt.show()

#get all bigrams in the corpus [frequency, bigram, topic]
def get_all_bigrams():
    all_bigrams = pd.DataFrame()
    for i in topic_docs.keys():
        # matrix of ngrams
        ngrams = c_vec.fit_transform(topic_docs[i]).astype(np.int8)
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram'})
        df_ngram = df_ngram[df_ngram.frequency>1]
        df_ngram['topic'] = 'topic' + str(i)
        all_bigrams = all_bigrams.append(df_ngram).reset_index(drop=True)
    return(all_bigrams)

all = get_all_bigrams()
#10 most common bigrams across all topics
all.nlargest(10, 'frequency')
#group based on topic
allgr=all.groupby(['topic']).apply(lambda x: x.sort_values(['frequency'], ascending = False)).reset_index(drop=True)
#most frequent bigram in each topic
allgr.groupby('topic').head(1)
#without 'climate change'
allgr[allgr.bigram!='climate change'].groupby('topic').head(1)
allgr.to_csv('COP22Bigrams.csv')


