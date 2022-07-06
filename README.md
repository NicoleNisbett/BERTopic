# Analysis tracker

This outlines the steps we have taken to analyse COP and FFF tweets using NLP and social network analysis

## General 
1. Scrape tweets using the academic twitter API
2. Use the `Bert_script.py` code to take in a csv of tweets and output a bert model based on transformer models
3. Use the `model_eval.ipynb` notebook to clean the model, getting rid of stopwords and re-assigning topics based on a threshold
4. Use the `get_distributions_clean.ipynb` notebook to load cleaned models and associated files, extract word frequencies, and create word distribution tables and frequency disrtibution graphs across datasets
    - Also computes the Jensen-Shannon Divergence scores across all datasets based on the normalised word frequency distributions in each.

## Bigrams

5. Run the `Bigrams.ipynb` notebook to create the bigram network for frames, delay, and fringe topics. 
5. Can use the csv outputs into Gephi to analyse the semantic network. 
    - Use the Force Atlas layout with the following:
    ```
    repulsion strength  = 50,000
    auto stabilize funtion = T
    attraction distribution = T
    adjust by sizes = T
    ```

    Node size by betweeness centrality, node colour by modularity category


### Network Similarity (Graphlet Correlation Distance)
7. Use the `Get_leda.ipynb` notebook to transform the bigram networks into a leda format (.gw). 
    - These should be saved in a ./BigramAnalysis/Graphlets folder.
6. In the terminal (wd should be ./BigramAnalysis/Graphlets), run 
    ```
    python count.py 'Ledanetworkname'.gw
    ```
    to produce the Graphlet Degree Vector (GDV) signatures in `.ndump2` format. *Make sure to download ORCA first - See Notes*

7. Still using the terminal, `cd ..` so you're now in ./BigramAnalysis/ folder. 
    - Run 

    ```
    conda activate bertopic_env2
    python NetworkComparisonScript.py ./Graphlets 'gcd11' 5
    ```
    to compute the GCD across all _n_ networks in the ./Graphlets folder. 
    - Outputs an _n x n_ txt file or csv.

10. Create a new folder ./BigramAnalysis/.Graphlet<'Frames/Delay/Fringe'> and copy all `.gw, .ndump2`, and the `gcd11.txt` file into it.
8. Go back to the `Get_leda.ipynb` notebook and run `get_gcd()` function to explore the GCD matrix.


# Notes
- Graphlet Correlation Distance code adapted from [here](http://www0.cs.ucl.ac.uk/staff/natasa/GCD/)
- In order to run the GDV and GCD code first need to:
    1. Download [orca.zip file](https://file.biolab.si/biolab/supp/orca/) 
    2. In the terminal, run: 
        ```
        echo 'export PATH=/Users/ipinni/Downloads/orca1:$PATH; export LD_LIBRARY_PATH=/Users/ipinni/Downloads/orca1:$LD_LIBRARY_PATH'  >> ~/.bash_profile  
        
        source ~/.bash_profile
        ```
         The actual path should be wherever you downloaded the orca file. See [here](https://www.orcasoftware.de/tutorials_orca/first_steps/trouble_install.html#linux-and-mac) for more info

# Key papers
## Topic Models
**BERTopic**

Grootendorst, M. (2022) BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint
[link](https://arxiv.org/abs/2203.05794)

**Embedding model**

* MPNET - Song, K., Tan, X., Qin, T., Lu, J. and Liu, T.Y., 2020. Mpnet: Masked and permuted pre-training for language understanding. Advances in Neural Information Processing Systems, 33, pp.16857-16867. - [link](https://arxiv.org/pdf/2004.09297.pdf)

* MPNET V2 - HuggingFace, all-mpnet-base-v2 [link](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

**UMAP dimension reduction**

McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018 - [link](https://arxiv.org/pdf/1802.03426.pdf)

**HDBSCAN clustering**

Campello R.J.G.B., Moulavi D., Sander J. (2013) Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei J., Tseng V.S., Cao L., Motoda H., Xu G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science, vol 7819. Springer, Berlin, Heidelberg - [link](https://doi.org/10.1007/978-3-642-37456-2_14)

## Network Analysis

### Different types of network comparison methods
* Tantardini, M., Ieva, F., Tajoli, L. et al. Comparing methods for comparing networks. Sci Rep 9, 17557 (2019).  - [link](https://doi.org/10.1038/s41598-019-53708-y)

**Graphlet-based measures**

* Pržulj, D. G. Corneil, I. Jurisica, Modeling interactome: scale-free or geometric?, Bioinformatics, Volume 20, Issue 18, 12 December 2004, Pages 3508–3515. - [link](https://doi.org/10.1093/bioinformatics/bth436)

**Alignment-free networks**

* Ömer Nebil Yaveroğlu, Tijana Milenković, Nataša Pržulj, Proper evaluation of alignment-free network comparison methods, Bioinformatics, Volume 31, Issue 16, 15 August 2015, Pages 2697–2704 - [link](https://doi.org/10.1093/bioinformatics/btv170)

### Graphlet Correlation Distance (GCD)
* Yaveroğlu, Ö., Malod-Dognin, N., Davis, D. et al. Revealing the Hidden Language of Complex Networks. Sci Rep 4, 4547 (2014) - [link](https://www.nature.com/articles/srep04547)

* Tomaž Hočevar, Janez Demšar, A combinatorial approach to graphlet counting, Bioinformatics, Volume 30, Issue 4, 15 February 2014, Pages 559–565 - [link](https://academic.oup.com/bioinformatics/article/30/4/559/205331)

* Rossi, R.A., Zhou, R. and Ahmed, N.K., 2017. Estimation of graphlet statistics. arXiv preprint arXiv:1701.01772. [link](https://arxiv.org/abs/1701.01772)

* Raphaël Charbey, Christophe Prieur. Graphlet-based characterization of many ego networks. 2018.
ffhal-01764253v2f

* Finotelli P, Piccardi C, Miglio E and Dulio P (2021) A Graphlet-Based Topological Characterization of the Resting-State Network in Healthy People. Front. Neurosci. 15:665544. doi: 10.3389/fnins.2021.665544 - [link](https://www.frontiersin.org/article/10.3389/fnins.2021.665544)


**Netdis (alternative to GDC)**

Waqar Ali, Tiago Rito, Gesine Reinert, Fengzhu Sun, Charlotte M. Deane, Alignment-free protein interaction network comparison, Bioinformatics, Volume 30, Issue 17, 1 September 2014, Pages i430–i437 - [link](https://doi.org/10.1093/bioinformatics/btu447)