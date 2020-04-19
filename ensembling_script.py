from scipy.stats import entropy
import numpy as np
import csv
import re
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import pearsonr


def compute_thresh_from_stopwords(lang):
    w2v_file = "w2v_results/" + lang + "_stop.tsv"
    w2v_df = pd.read_csv(w2v_file, sep="\t", header=0, names=['word','w2v_dist']).dropna()
    count, division = np.histogram(w2v_df['w2v_dist'], bins=10)
        threshold = 0
        for (c, d) in zip(count[1:], division[1:]):
            if lang == 'english' or lang == 'german' or lang == 'swedish':
                if c < 20:
                    threshold = d
                    break
            elif lang == 'latin':
                if c < 100:
                    threshold = d
                    break
        return threshold


languages = ['english', 'german', 'latin', 'swedish']
method_name1 = 'aff_prop'
method_name2 = 'w2v_dist'


for lang in languages:
    # open w2v cosine dist file (this only for ensembling with word2vec, comment out if not needed)
    w2v_file = "w2v_results/" + lang + "_OP.tsv"
    w2v_df = pd.read_csv(w2v_file, sep="\t", names=['word','w2v_dist'])

    # open BERT clustering file
    clustering_file = "semeval_results/results_" + lang + "_fine_tuned_concat.csv"
    clustering_df = pd.read_csv(clustering_file, sep="\t")
    # only do this when ensembling with word2vec
    if lang == 'english':
         clustering_df['word'] = clustering_df['word'].apply(lambda x: x.split("_")[0])

    # merge dataframes results if needed
    df_merged = pd.merge(w2v_df, clustering_df, on='word')

    # let's see how words are ranked by the two methods
    df_merged = df_merged[['word', method_name1, method_name2]]
    df_merged = df_merged.assign(method1_rank=df_merged[method_name1].rank())
    df_merged = df_merged.assign(method2_rank=df_merged[method_name2].rank())

    # normalize distances for method1 such that the mean is at 0
    method1_arr = np.array(df_merged[method_name1])
    method1_norm = method1_arr - np.mean(method1_arr)

    # do the same for method2
    method2_arr = np.array(df_merged[method_name2])
    method2_norm = method2_arr - np.mean(method2_arr)

    # get the mean of the two normalized distances
    ensemble_mean = np.mean([method1_norm, method2_norm], axis=0)

    df_merged = df_merged.assign(ensemble_mean=ensemble_mean)

    #csv_file = "semeval_results/ensembling_" + method_name1 + "_" + method_name2 + "_" + lang+".csv"
    #df_merged.to_csv(csv_file, sep='\t', encoding='utf-8', index=False)

    print("\n===== Results for", lang.upper(),"=====")
    print("Correlation between", method_name1, "and", method_name2)
    # compute Spearman and Pearson correlation between methods
    print(method_name1, "and", method_name2)
    spearman_corr = spearmanr(df_merged[method_name1].astype('float64'), df_merged[method_name2].astype('float64'))
    print("Spearman correlation:", spearman_corr[0])

    # compute Spearman and Pearson correlation between method1 and the ensembled method
    print("Ensemble vs", method_name1)
    spearman_corr = spearmanr(df_merged[method_name1].astype('float64'), df_merged['ensemble_mean'].astype('float64'))
    print("Spearman correlation:", spearman_corr[0])

    # compute Spearman and Pearson correlation between method2 and the ensembled method
    print("Ensemble vs", method_name2)
    spearman_corr = spearmanr(df_merged[method_name2].astype('float64'), df_merged['ensemble_mean'].astype('float64'))
    print("Spearman correlation:", spearman_corr[0])


