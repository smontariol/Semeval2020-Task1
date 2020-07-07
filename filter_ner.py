import os, sys
from collections import defaultdict
import pickle
import re
import argparse

# latin
from cltk.tag import ner

#german, english
import spacy
from polyglot.text import Text

de_nlp = spacy.load("de_core_news_sm")
en_nlp = spacy.load('en_core_web_sm')

#swedish

from calculate_semantic_change import compute_divergence_from_cluster_labels


def count_names_latin(target, sents):
    total_name_count = 0
    target_name_count = 0
    for s in sents:
        ner_tags = ner.tag_ner('latin', input_text=re.sub(r'#\d', "", s))
        names = [tag[0].lower() for tag in ner_tags if len(tag)>1 and tag[1] == 'Entity']
        total_name_count += len(names)
        if target.lower() in names:
            target_name_count += 1
    return total_name_count, target_name_count


def count_names_german(target, sents):
    total_name_count = 0
    target_name_count = 0
    for s in sents:
        words = de_nlp(s)
        names = [w.text.lower() for w in words if w.tag_ == 'NE' and w.text != 'd'] # d is probably article in the lemmatized text, always returns NE
        total_name_count += len(names)
        if target.lower() in names:
            target_name_count += 1
    return total_name_count, target_name_count


def count_names_english(target, sents):
    total_name_count = 0
    target_name_count = 0
    for s in sents:
        words = en_nlp(s)
        names = [w.text.lower() for w in words if w.tag_ == 'NNP']
        total_name_count += len(names)
        if target.replace("_nn", "").replace("_vb", "").lower() in names:
            target_name_count += 1
    return total_name_count, target_name_count


def count_names_swedish(target, sents):
    total_name_count = 0
    target_name_count = 0
    for s in sents:
        text = Text(s, hint_language_code='sv')
        names = [tag[0].lower() for tag in text.pos_tags if tag[1] == 'PROPN']
        total_name_count += len(names)
        if target.lower() in names:
            target_name_count += 1
    return total_name_count, target_name_count
            
def filter_name_clusters(cluster_to_sentence, target, l, target_name_threshold, total_name_threshold):
    keep_clusters = []
    radical_keep_clusters = []
    for cluster, sents in cluster_to_sentence.items():
        cluster_size = len(sents)


        if cluster_size <= 2:
            continue

        if l == "latin":
            total_name_count, target_name_count =count_names_latin(target, sents)
        elif l == "german":
            total_name_count, target_name_count =count_names_german(target, sents)
        elif l == "english":
            total_name_count, target_name_count =count_names_english(target, sents)
        elif l == "swedish":
            total_name_count, target_name_count =count_names_swedish(target, sents)

        if target_name_count / cluster_size < target_name_threshold:
            keep_clusters.append(cluster)
        if total_name_count < cluster_size * total_name_threshold:
            radical_keep_clusters.append(cluster)
    return  keep_clusters, radical_keep_clusters

            

def filter(label_file, sent_file, filter_folder, l, target_name_threshold, total_name_threshold):
    labels = pickle.load(open(label_file, 'rb'))
    sentences = pickle.load(open(sent_file, 'rb'))

    print()
    print("==============================")
    print(label_file)
    print(sent_file)
    print(l)
    print()

    targets = list(labels.keys())

    filtered_jsd = {}
    radical_filtered_jsd = {}
    keep_info = {}
    
    for target in targets:
        cluster_to_sentence = defaultdict(list)
        for t in ['t1', 't2']:    
            for label, sent in zip(labels[target][t], sentences[target][t]):
                cluster_to_sentence[label].append(sent)
        keep_clusters, radical_keep_clusters = filter_name_clusters(cluster_to_sentence, target, l, target_name_threshold, total_name_threshold)

        keep_info[target] = keep_clusters, radical_keep_clusters
        
        keep_labels = [[l for l in labels[target][t] if l in keep_clusters] for t in ['t1', 't2']]
        radical_keep_labels = [[l for l in labels[target][t] if l in radical_keep_clusters] for t in ['t1', 't2']]

        filtered_jsd[target] = compute_divergence_from_cluster_labels(keep_labels[0], keep_labels[1])
        
        radical_filtered_jsd[target] = compute_divergence_from_cluster_labels(radical_keep_labels[0], radical_keep_labels[1])
        print(target, filtered_jsd[target], radical_filtered_jsd[target])
        
    file_name_base = label_file.split('/')[-1].replace('.pkl', '')

    info_file = os.path.join(filter_folder, file_name_base + "_keep_info.pkl")
    pickle.dump(keep_info, open(info_file, 'wb'))
    
    res_file = os.path.join(filter_folder, file_name_base + "_result.csv")

    with open(res_file, 'w') as out:
        print('\t'.join(['word', 'filtered_jsd', 'clusters', 'radically_filtered_jsd', 'clusters']), file=out)
        for t in filtered_jsd.keys():
            print('\t'.join([str(x) for x in [t, filtered_jsd[t], len(keep_info[t][0]), radical_filtered_jsd[t], len(keep_info[t][1])]]), file=out)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='english', const='all', nargs='?',
                        help="Choose a language", choices=['english', 'latin', 'swedish', 'german'])
    parser.add_argument("--input_sent_file", default='semeval_results/sents_english.pkl', type=str,
                        help="Path to sentence file generated by caluclate_semantic_change_script")
    parser.add_argument("--input_label_file", default='semeval_results/aff_prop_labels_english.pkl', type=str,
                        help="Path to sentence file generated by caluclate_semantic_change_script")
    parser.add_argument("--output_dir_path", default='semeval_results_filtered', type=str,
                        help="Path to file with cluster labels")
    parser.add_argument("--target_name_threshold", default=0.8, type=float,
                        help="Filter cluster if target_name_threshold * 100% of target words inside are named entities.")
    parser.add_argument("--total_name_threshold", default=5, type=int,
                        help="Filter cluster if the number ofproper nouns is total_name_threshold times larger than the number of sentences.")
    args = parser.parse_args()

    languages = ['english', 'latin', 'swedish', 'german']
    if args.language not in languages:
        print("Language not valid, valid choices are: ", ", ".join(languages))
        sys.exit()

    filter_folder = args.output_dir_path
    if not os.path.exists(filter_folder):
        os.makedirs(filter_folder)
    filter(args.input_label_file, args.input_sent_file, filter_folder, args.language, args.target_name_threshold, args.total_name_threshold)
