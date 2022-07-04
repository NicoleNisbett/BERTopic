#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 11:47:55 2022

@author: ipinni
"""

from bertopic import BERTopic
import pandas as pd
import numpy as np
import pickle    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import re
from collections import Counter


#load model bits
loadedProbs = np.load('COP22probs.npy')

with open('/nobackup/ipinni/COP22topics.list', 'rb') as config_list_file:   topics_list = pickle.load(config_list_file)
    
with open('/nobackup/ipinni/COP22docs.list', 'rb') as docs_list_file:   docs_list = pickle.load(docs_list_file)

model = BERTopic.load("/nobackup/ipinni/COP22_Bert_model", embedding_model = "all-mpnet-base-v2")

#explore
model.get_topic_info() 
model.get_params()
model.get_topic(1)

model.visualize_topics().show()

#clean model and return 
def clean_model(model, probs, docs):
    #set threshold for new topic assignments
    probability_threshold = 0.01
    thresh_topics = [np.argmax(prob) if max(prob)>= probability_threshold else -1 for prob in probs]
    num_topics = max(topics_list)+1
    #assign new topics to docs ('reducing' topics to the same number)
    new_topics, new_probs = model.reduce_topics(docs, thresh_topics, probs, nr_topics=num_topics)
    #remove stopwords and add ngrams
    my_stopwords = list(["rt","RT", "&", "amp", "&amp", "http","https", "http://", "https://", "fav", "FAV"])
    new_stopwords = frozenset(list(text.ENGLISH_STOP_WORDS) + my_stopwords)
    #vectoriser model
    vectorizer_model = CountVectorizer(stop_words = new_stopwords,ngram_range=(1, 2))
    #new cleaned model
    model.update_topics(docs, new_topics, vectorizer_model=vectorizer_model)
    return(new_topics, new_probs, model)

topics, probs, modelClean = clean_model(model,loadedProbs, docs_list) #takes a while for larger data

#explore model
modelClean.get_topic_info()
modelClean.get_params()
modelClean.get_representative_docs(2)

#save cleaned model
modelClean.save("COP22_Bert_model", save_embedding_model=False)

#save new topics and probs
with open('/nobackup/ipinni/COP22topics.list', 'wb') as config_list_file:
 pickle.dump(topics, config_list_file)

np.save('/nobackup/ipinni/COP22probs.npy', probs) 

#save representative docs
representative_docs = modelClean.get_representative_docs()
data = pd.DataFrame.from_dict(representative_docs, orient='index')
data.to_csv('COP22TopDocs.csv')

#visualise
modelClean.visualize_hierarchy( orientation = 'bottom').show()
modelClean.visualize_topics().show()
'''
rclone copy COP22TopDocs.csv onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22_Bert_model onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22docs.list onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22outliers.list onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22probs.npy onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22topics.list onedrive:UKRI_Tweet_Data/completed/COP22
rclone copy COP22CleanTweets.csv onedrive:UKRI_Tweet_Data/completed/COP22
'''

'''#environment packages
conda create --name bertopic_env python=3.7
pip install bertopic
conda install -c conda-forge hdbscan
pip install tweet-preprocessor

# if running model in Jupyter
# pip install ipywidgets

# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       1_gnu    conda-forge
bertopic                  0.9.4                    pypi_0    pypi
ca-certificates           2021.10.8            ha878542_0    conda-forge
certifi                   2021.10.8        py37h89c1867_1    conda-forge
charset-normalizer        2.0.10                   pypi_0    pypi
click                     8.0.3                    pypi_0    pypi
cython                    0.29.26                  pypi_0    pypi
filelock                  3.4.2                    pypi_0    pypi
hdbscan                   0.8.27           py37h902c9e0_0    conda-forge
huggingface-hub           0.4.0                    pypi_0    pypi
idna                      3.3                      pypi_0    pypi
importlib-metadata        4.10.1                   pypi_0    pypi
joblib                    1.1.0              pyhd8ed1ab_0    conda-forge
ld_impl_linux-64          2.35.1               h7274673_9  
libblas                   3.9.0                8_openblas    conda-forge
libcblas                  3.9.0                8_openblas    conda-forge
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0              h1d223b6_12    conda-forge
libgfortran-ng            7.5.0               h14aa051_19    conda-forge
libgfortran4              7.5.0               h14aa051_19    conda-forge
libgomp                   11.2.0              h1d223b6_12    conda-forge
liblapack                 3.9.0                8_openblas    conda-forge
libopenblas               0.3.12          pthreads_hb3c22a3_1    conda-forge
libstdcxx-ng              9.1.0                hdf63c60_0  
llvmlite                  0.38.0                   pypi_0    pypi
ncurses                   6.3                  h7f8727e_2  
nltk                      3.6.7                    pypi_0    pypi
numba                     0.55.1                   pypi_0    pypi
numpy                     1.21.5                   pypi_0    pypi
openssl                   1.1.1l               h7f98852_0    conda-forge
packaging                 21.3                     pypi_0    pypi
pandas                    1.3.5                    pypi_0    pypi
pillow                    9.0.0                    pypi_0    pypi
pip                       21.2.2           py37h06a4308_0  
plotly                    5.5.0                    pypi_0    pypi
pynndescent               0.5.6                    pypi_0    pypi
pyparsing                 3.0.7                    pypi_0    pypi
python                    3.7.11               h12debd9_0  
python-dateutil           2.8.2                    pypi_0    pypi
python_abi                3.7                     2_cp37m    conda-forge
pytz                      2021.3                   pypi_0    pypi
pyyaml                    5.4.1                    pypi_0    pypi
readline                  8.1.2                h7f8727e_1  
regex                     2022.1.18                pypi_0    pypi
requests                  2.27.1                   pypi_0    pypi
sacremoses                0.0.47                   pypi_0    pypi
scikit-learn              1.0.2                    pypi_0    pypi
scipy                     1.7.3                    pypi_0    pypi
sentence-transformers     2.1.0                    pypi_0    pypi
sentencepiece             0.1.96                   pypi_0    pypi
setuptools                58.0.4           py37h06a4308_0  
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.37.0               hc218d9a_0  
tenacity                  8.0.1                    pypi_0    pypi
threadpoolctl             3.0.0              pyh8a188c0_0    conda-forge
tk                        8.6.11               h1ccaba5_0  
tokenizers                0.11.4                   pypi_0    pypi
torch                     1.10.2                   pypi_0    pypi
torchvision               0.11.3                   pypi_0    pypi
tqdm                      4.62.3                   pypi_0    pypi
transformers              4.16.1                   pypi_0    pypi
tweet-preprocessor        0.6.0                    pypi_0    pypi
typing-extensions         4.0.1                    pypi_0    pypi
umap-learn                0.5.2                    pypi_0    pypi
urllib3                   1.26.8                   pypi_0    pypi
wheel                     0.37.1             pyhd3eb1b0_0  
xz                        5.2.5                h7b6447c_0  
zipp                      3.7.0                    pypi_0    pypi
zlib                      1.2.11               h7f8727e_4  

'''