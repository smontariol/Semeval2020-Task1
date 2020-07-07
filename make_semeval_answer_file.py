import pandas as pd
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin', 'swedish', 'german'])
    parser.add_argument("--results_file", default="semeval_results/results_english.csv", type=str,
                        help="Path to csv file with results")
    parser.add_argument("--method", default='aff_prop', type=str,
                        help="CLustering method for which to write results-should match a column in the results file")
    parser.add_argument("--target_path", default='data/english/targets.txt', type=str,
                        help="Path to target files")

    args = parser.parse_args()
    thresholds = {'english': 0.3,
                  'german': 0.43,
                  'latin': 0.4,
                  'swedish':0.4
                  }
    method_name = args.method
    lang = args.language
    languages = ['english', 'latin', 'swedish', 'german']
    methods = ['aff_prop', 'kmeans_5', 'kmeans_7', 'averaging']
    if method_name not in methods:
        print("Method not valid, valid choices are: ", ", ".join(methods))
        sys.exit()
    if lang not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()
    print("Language:", lang.upper())

    clustering_file = args.results_file
    clustering_df = pd.read_csv(clustering_file, sep="\t")
    target_file = args.target_path
    target_words = open(target_file,'r').readlines()
    target_words = [t.strip() for t in target_words]

    thresh = thresholds[lang]
    if not os.path.exists("answer/task1/"):
        os.makedirs("answer/task1/")
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

    if not os.path.exists("answer/task2/"):
        os.makedirs("answer/task2/")
    outfilename2 = "answer/task2/" + lang  + ".txt"
    outfile2 = open(outfilename2, 'w', encoding='utf-8')
    for i,word in enumerate(target_words):
        val = float(clustering_df[clustering_df['word'] == word][method_name])
        line = word + "\t" + str(val)
        outfile2.write(line)
        if i < len(target_words):
            outfile2.write("\n")

    print("Done writing", outfilename2,"!")




