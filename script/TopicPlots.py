import numpy as np
import pandas as pd
from umap import UMAP
from bertopic import BERTopic
import pickle

import matplotlib
import matplotlib.pyplot as plt

# load topic model
path = "/Users/ipinni/Library/CloudStorage/OneDrive-UniversityofLeeds/UKRI_Tweet_Data/completed/"
def get_data(version):
    with open(path + version + "/" + version + "docs.list", 'rb') as docs_list_file:
        docs = pickle.load(docs_list_file)

    with open(path + version + "/" + version + "topics.list", 'rb') as topics_list_file:
        topics = pickle.load(topics_list_file)
    
    model = BERTopic.load(path + version + "/" + version + "_Bert_model" , embedding_model = "all-mpnet-base-v2")

    embeddings = model._extract_embeddings(docs, method = "document")

    return(model, embeddings, topics)   

COP20model, COP20embs, COP20topics = get_data("COP20")

COP25model, COP25embs, COP25topics = get_data("COP25")

#create df to show embeddings and distance measures

COP20_frames = [3,4,7,14,22,35,40,44]
COP20_delay = [6,15,17,31,45]

def get_plot(embs, topics, model, frames, delay):
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit(embs)
    df = pd.DataFrame(umap_model.embedding_, columns=["x", "y"])
    df["topic"] = topics

    # Plot parameters
    top_n = 65

    # Slice data to only colour frame and delay topics
    to_plot = df.copy()
    to_plot[df.topic >= top_n] = -1
    outliers = to_plot.loc[to_plot.topic <= 0]
    sc_frames = df.loc[df.topic.isin(frames)]
    sc_delay = df.loc[df.topic.isin(delay)]

    # Visualize outliers + inliers 
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter_outliers = ax.scatter(outliers['x'], outliers['y'], c="#E0E0E0", s=1, alpha=.3)
    scatter_frames = ax.scatter(sc_frames['x'], sc_frames['y'], c='red', s=1, alpha=.3)
    scatter_delay = ax.scatter(sc_delay['x'], sc_delay['y'], c='blue', s=1, alpha=.3)

    # Add topic names to clusters
    centroids = to_plot.groupby("topic").mean().reset_index().iloc[1:]
    for row in centroids.iterrows():
        topic = int(row[1].topic)
        if topic in frames + delay:
            text = f"{topic}: " + "_".join([x[0] 
            for x in model.get_topic(topic)[:2]])
            ax.text(row[1].x, row[1].y*1.01, text, fontsize=9, horizontalalignment='center')

    ax.text(0.99, 0.01, f"BERTopic - Framing and delay topics", transform=ax.transAxes, horizontalalignment="right", color="black")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()


get_plot(COP20embs, COP20topics, COP20model, COP20_frames, COP20_delay)






#plots all the topics
def get_plot(embs, topics, model):
    umap_model = UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit(embs)
    df = pd.DataFrame(umap_model.embedding_, columns=["x", "y"])
    df["topic"] = topics

    # Plot parameters
    top_n = 65

    # Slice data to only colour largest topics (apart from 0 and -1)
    to_plot = df.copy()
    to_plot[df.topic >= top_n] = -1
    outliers = to_plot.loc[to_plot.topic <= 0]
    non_outliers = to_plot.loc[to_plot.topic > 0]

    # Visualize topics
    cmap = matplotlib.colors.ListedColormap(['#FF5722', '#03A9F4',
    '#4CAF50',
    '#80CBC4',
    '#673AB7', 
    '#795548', 
    '#E91E63', 
    '#212121',
    '#00BCD4', 
    '#CDDC39', 
    '#AED581', 
    '#FFE082', 
    '#BCAAA4', 
    '#B39DDB', 
    '#F48FB1', 
    ])

    # Visualize outliers + inliers
    fig, ax = plt.subplots(figsize=(15, 15))
    scatter_outliers = ax.scatter(outliers['x'], outliers['y'], c="#E0E0E0", s=1, alpha=.3)
    scatter = ax.scatter(non_outliers['x'], non_outliers['y'], c=non_outliers['topic'], s=1, alpha=.3, cmap="rainbow")

    # Add topic names to clusters
    centroids = to_plot.groupby("topic").mean().reset_index().iloc[1:]
    for row in centroids.iterrows():
        topic = int(row[1].topic)
        text = f"{topic}: " + "_".join([x[0] for x in model.get_topic(topic)[:1]])
        ax.text(row[1].x, row[1].y*1.01, text, fontsize=9, horizontalalignment='center')

    ax.text(0.99, 0.01, f"BERTopic - All topics", transform=ax.transAxes, horizontalalignment="right", color="black")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.show()
