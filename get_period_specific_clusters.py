import sys
import pickle
import os
import numpy as np
import argparse

def write_to_file(lang, changed, target_path):
    target_words = open(target_path, 'r').readlines()
    target_words = [t.strip() for t in target_words]

    if not os.path.exists("answer/task1/"):
        os.makedirs("answer/task1/")
    outfilename1 = "answer/task1/" + lang + ".txt"
    outfile1 = open(outfilename1, 'w', encoding='utf-8')

    for i, word in enumerate(target_words):
        classif = 1 if word in changed else 0
        line = word + "\t" + str(classif)
        outfile1.write(line)
        if i < len(target_words) - 1:
            outfile1.write("\n")

    print("Done writing", outfilename1, "!")


def get_targets(input_path, lang):
    targets_dict = {}
    with open(input_path, 'r', encoding='utf8') as f:
        for line in f:
            target = line.strip()
            if lang=='english':
                 target_no_pos = target[:-3]
                 targets_dict[target_no_pos] = target
            else:
                targets_dict[target] = target
    return targets_dict

def classify(targets, lang, label_file, dynamic, treshold=2):

    cluster_data = pickle.load(open(label_file, 'rb'))
    changed_l = []
    unchanged_l = []

    for word in targets:
        print(word)

        labels_t0 = cluster_data[word]['t1']
        labels_t1 = cluster_data[word]['t2']
        df_clusters = {}

        for label in labels_t0:
            if label not in df_clusters:
                df_clusters[label] = [1,0]
            else:
                df_clusters[label][0] += 1

        for label in labels_t1:
            if label not in df_clusters:
                df_clusters[label] = [0,1]
            else:
                df_clusters[label][1] += 1

        # average number of words per cluster
        if dynamic == True:
            size_clusters = [n1 + n2 for [n1, n2] in list(df_clusters.values())]
            dynamic_threshold = 2 * np.mean(size_clusters)
        else:
            dynamic_threshold = 10

        changed = False

        for k, v in df_clusters.items():
            t1_count, t2_count = v
            if t1_count + t2_count >= dynamic_threshold:
                if t1_count < treshold or t2_count < treshold:
                    changed = True
                    print("Change: ", k,v)

        print('---------------------------------')
        print()

        if changed:
            changed_l.append(word)
        else:
            unchanged_l.append(word)

    print('changed: ', changed_l)
    print('unchanged: ', unchanged_l)
    return changed_l


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin', 'swedish', 'german'])
    parser.add_argument("--target_path", default='data/english/targets.txt', type=str,
                        help="Path to target file")
    parser.add_argument("--results_file", default="semeval_results/kmeans_7_labels_english.pkl", type=str,
                        help="Path to file with cluster labels")
    parser.add_argument("--dynamic_treshold", action="store_true", help="If true, dynamic treshold (appropriate for affinity propagation cluster) will be used")
    parser.add_argument("--treshold", default=2, type=int, help="Cluster should contain less or equal than treshold instances from a specific time period to be considered period specific.")
    args = parser.parse_args()

    languages = ['english', 'latin', 'swedish', 'german']
    if args.language not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()

    target_dict = get_targets(args.target_path, args.language)
    targets = target_dict.values()

    changed = classify(targets, args.language, args.results_file, args.dynamic_treshold, args.treshold)
    write_to_file(args.language, changed, args.target_path)
