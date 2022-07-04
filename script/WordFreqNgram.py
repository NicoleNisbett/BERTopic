a#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 22:15:58 2019

@author: viktoria
"""
import json 
import csv
import pandas as pd
import collections
import codecs
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.probability import FreqDist

XR1 = pd.read_json("ExtinctionRebellion_H.json", lines=True)
XR2 = pd.read_json("ExtinctionRebellionAll.json", lines=True)
XR3 = pd.read_json("ExtinctionRebellionAll2.json", lines=True)
XR4 = pd.read_json("ExtinctionRebellion3.json", lines=True)
XR5 = pd.read_json("ExtinctionRebellionAll6.json", lines=True)

XRtweets = all_data = pd.concat([XR1,XR2,XR3,XR4, XR5])

tweets_text = XRtweets['text']


# define function
def freq_words(file):  
    output = open("FreqGJ.csv", 'w')
    #tweets = pd.read_json(file)     
    default_stopwords_en = set(nltk.corpus.stopwords.words('english'))  
    default_stopwords_fr = set(nltk.corpus.stopwords.words('french'))
    # Create customized stop wordlist
    custom_stopwords = set(("rt", "don't", "i'm"))
    all_stopwords = default_stopwords_en | default_stopwords_fr | custom_stopwords       
    tt = TweetTokenizer()     
    tweets_text = tweets['text'].apply(tt.tokenize)     
    flat_list = [item for sublist in tweets_text for item in sublist] 
    words = [word for word in flat_list if len(word) > 2]     
    words = [word.lower() for word in words]     
    words = [word for word in words if word not in all_stopwords] 
    freqdist = nltk.FreqDist(words)     
    for word, frequency in freqdist.most_common(300):      
    	  output.write(u'{},{}'.format(word, frequency) + "\n")
## call function and store results in an object
#freqd = freq_words("GiletsJaunes_H.json")
freqd = freq_words(tweets)

#tweetsn = json.loads(json.dumps("ExtinctionRebellion_H.json"))

def freq_words(frames):   
    output = open("GJ_Users1.csv", 'w')
    default_stopwords_en = set(nltk.corpus.stopwords.words('english'))  
    default_stopwords_fr = set(nltk.corpus.stopwords.words('french')) 
    # Create customized stop wordlist
    custom_stopwords = set(("rt", "don't", "i'm"))
    all_stopwords = default_stopwords_en | default_stopwords_fr | custom_stopwords       
    tt = TweetTokenizer()     
    tweets_text = frames['user_description'].apply(tt.tokenize)  
    flat_list = [item for sublist in tweets_text for item in sublist] 
    words = [word for word in flat_list if len(word) > 2]     
    words = [word.lower() for word in words]     
    words = [word for word in words if word not in all_stopwords] 
    freqdist = nltk.FreqDist(words)  
    for word, frequency in freqdist.most_common(300):   
    	  output.write(u'{},{}'.format(word, frequency) + "\n")
# call function and store results in an object
freqd = freq_words(df_all)

# check
#for word, frequency in freqd.most_common(50): 	
#    print(u'{};{}'.format(word, frequency))
#
## define function 
#def saveFreqWord(freqdo, filename):     
#    output = open(filename, 'w')     
#    for word, frequency in freqdo.most_common(300):        
#        output.write(u'{};{}'.format(word, frequency) + "\n")
#        
#def saveFreqWord(freqdo, filename):     
#    output = open(filename, 'w')     
#    for word, frequency in freqdo:        
#        output.write(u'{};{}'.format(word, frequency) + "\n")
##call function
#saveFreqWord(freqd, "test")
## or to call previously defined function within call 
#saveFreqWord(freq_words("ExtinctionRebellion_H.json"), "Freq.csv")

default_stopwords_en = set(nltk.corpus.stopwords.words('english'))
default_stopwords_fr = set(nltk.corpus.stopwords.words('french'))
#custom_stopwords = set(("rt", "don't", "i'm"))
# Or get it from a file, that you or someone else prepared instead (one # stopword per line, UTF-8)
stopwords_file = './stopwords.txt'
custom_stopwords = set(codecs.open(stopwords_file, 'r', 'utf-8').read().splitlines())
all_stopwords = default_stopwords_en | default_stopwords_fr | custom_stopwords
filter_stops = lambda w: len(w)< 2 or w in all_stopwords



from flatten_json import flatten

tweets_n = []
for line in open('GiletsJaunes_H.json', 'r'):
    tweets_n.append(json.loads(line))
tweets2_n = []
for line in open('Gilets_Jaunes_H.json', 'r'):
    tweets2_n.append(json.loads(line))
tweets_flattened = [flatten(d) for d in tweets_n]
tweets2_flattened = [flatten(d) for d in tweets2_n]
df = pd.DataFrame(tweets_flattened)
df2 = pd.DataFrame(tweets2_flattened)
frames2 = [df, df2]


tweets = pd.read_json("GiletsJaunes_H.json", lines=True)

tweets2 = pd.read_json("Gilets_Jaunes_H.json", lines=True)

frames = [tweets, tweets2]

#df = pd.DataFrame(unnested)
df_all = pd.concat(frames2)


tweets_all = pd.concat(frames)
tt = TweetTokenizer()     
tweets_text = tweets['text'].apply(tt.tokenize)     
flat_list = [item for sublist in tweets_text for item in sublist] 
words = [word for word in flat_list if len(word) > 1]     
words = [word.lower() for word in words]     
words = [word for word in words if word not in all_stopwords]
# create a function for obtaining bigrams
def bicolloc(text):
    bgm = nltk.collocations.BigramAssocMeasures()
    bcf = nltk.collocations.BigramCollocationFinder.from_words(text)
    bcf.apply_word_filter(filter_stops)
    scored = bcf.score_ngrams(bgm.student_t)[:750]
    temp_dict = dict(scored)
    csvData = []
    for (col1, col2), col3 in iter(temp_dict.items()):
        csvData.append("%s, %s, %s" % (col1, col2, col3))
    f = open('BigramsFFF1.csv', 'w')
    f.write("\n".join(csvData))
    f.close()
bicolloc(words)

def tricolloc(text):
	tgm = nltk.collocations.TrigramAssocMeasures()
	tcf = nltk.collocations.TrigramCollocationFinder.from_words(text)
	tcf.apply_word_filter(filter_stops)
	scored = tcf.score_ngrams(tgm.student_t)[:200]
	return scored
tricolloc(words)

from nltk.metrics.association import QuadgramAssocMeasures
def quadcolloc(text):
	qgm = QuadgramAssocMeasures()
	qcf = nltk.collocations.QuadgramCollocationFinder.from_words(text)
	qcf.apply_word_filter(filter_stops)
	scored = qcf.score_ngrams(qgm.student_t)[:200]
	return scored
quadcolloc(words)

#looking for bigram-collocations with certain key words
#calling the function by keybigr(file, 'keyword')
def keybigr(text, word):
    bgm = nltk.collocations.BigramAssocMeasures()
    bcf = nltk.collocations.BigramCollocationFinder.from_words(text,2)
    bcf.apply_word_filter(filter_stops)
    scored = bcf.score_ngrams(bgm.student_t)[:600]
    prefix_keys = collections.defaultdict(list)
    for key, scores in scored:
        prefix_keys[key[0]].append((key[1], scores))
    for key in prefix_keys:
        prefix_keys[key].sort(key = lambda x: -x[1])
    return word, prefix_keys[word][:40]
keybigr(words, "climate")


# barcharts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import seaborn as sns
from matplotlib import cm
import colorsys
import plotly.graph_objs as go

# Make a fake dataset
height = [3499, 2262, 2169, 1863, 1754, 1582, 1427, 1340, 1317, 1236, 1194, 965, 873, 824, 822, 781, 688, 672, 
          663, 657, 616, 612, 575, 574, 544, 539, 529]
bars = ('france', 'politique', 'MAGA', 'francais', 'monde', 'droite', 'gauche', '#giletsjaunes', 'patriot', 'Trump', 'liberte', 
        'militant', 'citoyen', 'macron', 'conseiller', 'journaliste', 'nature', 'apolitique', 'animaux', 'Marine Le Pen', 
        'conservative', 'culture', 'justice', 'musique', 'dieu', 'insoumis', 'humaniste')
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars, rotation='vertical')
plt.show()
