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


languages = ['english', 'german', 'latin', 'swedish']
thresholds = {'english': 0.3,
              'german': 0.43,
              'latin': 0.4,
              'swedish':0.4
              }
method_name = 'aff_prop'
for lang in languages:
    print("Language:", lang.upper())

    clustering_file = "semeval_results/results_" + lang + "_fine_tuned_concat.csv"
    clustering_df = pd.read_csv(clustering_file, sep="\t")
    target_file = "semeval_targets/"+lang+".txt"
    target_words = open(target_file,'r').readlines()
    target_words = [t.strip() for t in target_words]

    thresh = thresholds[lang]
    outfilename1 = "answer/task1/" + lang  + ".txt"
    outfile1 = open(outfilename1, 'w', encoding='utf-8')
    for i,word in enumerate(target_words):
        val = float(clustering_df[clustering_df['word'] == word][method_name])
        classif = 0 if val < thresh else 1
        line = word + "\t" + str(classif)
        outfile1.write(line)
        if i < len(target_words):
            outfile1.write("\n")
    print("Done writing", outfilename1,"!")

    outfilename2 = "answer/task2/" + lang  + ".txt"
    outfile2 = open(outfilename2, 'w', encoding='utf-8')
    for i,word in enumerate(target_words):
        val = float(clustering_df[clustering_df['word'] == word][method_name])
        line = word + "\t" + str(val)
        outfile2.write(line)
        if i < len(target_words):
            outfile2.write("\n")

    print("Done writing", outfilename2,"!")




