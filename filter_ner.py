import os, sys
from collections import defaultdict
import pickle
import re

# latin
from cltk.tag import ner

#german, english
import spacy
de_nlp = spacy.load("de_core_news_sm")
en_nlp = spacy.load('en_core_web_sm')

#swedish
from polyglot.text import Text

from get_all_results import compute_divergence_from_cluster_labels

TARGET_NAME_THRESHOLD = 0.8
TOTAL_NAME_THRESHOLD = 5

FILTER_SHORT = True
FILTER_NER = True

languages = ["latin", "german", "english", "swedish"]
emb = ["concat", "averaged"]
input_folder = sys.argv[1]

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
            
def filter_name_clusters(cluster_to_sentence, target, l):
    keep_clusters = []
    radical_keep_clusters = []
    for cluster, sents in cluster_to_sentence.items():
        cluster_size = len(sents)

        if FILTER_SHORT:
            if cluster_size <= 2:
                continue

        if FILTER_NER:
            
	        if l == "latin":
	            total_name_count, target_name_count =count_names_latin(target, sents)
	        elif l == "german":
	            total_name_count, target_name_count =count_names_german(target, sents)
	        elif l == "english":
	            total_name_count, target_name_count =count_names_english(target, sents)
	        elif l == "swedish":
	            total_name_count, target_name_count =count_names_swedish(target, sents)
	
	        if target_name_count / cluster_size < TARGET_NAME_THRESHOLD:
	            keep_clusters.append(cluster)
	        if total_name_count < cluster_size * TOTAL_NAME_THRESHOLD:
	            radical_keep_clusters.append(cluster)

        else:
            keep_clusters.append(cluster)
            radical_keep_clusters.append(cluster)

    return  keep_clusters, radical_keep_clusters

            

def filter(label_file, sent_file, l):
    labels = pickle.load(open(os.path.join(input_folder, label_file), 'rb'))
    sentences = pickle.load(open(os.path.join(input_folder, sent_file), 'rb'))

    print()
    print("==============================")
    print(label_file)
    print(sent_file)
    print(l)
    print()
    
    new_labels = {}
        
    targets = list(labels.keys())

    filtered_jsd = {}
    radical_filtered_jsd = {}
    keep_info = {}
    
    for target in targets:
        cluster_to_sentence = defaultdict(list)
        for t in ['t1', 't2']:    
            for label, sent in zip(labels[target][t], sentences[target][t]):
                cluster_to_sentence[label].append(sent)
        keep_clusters, radical_keep_clusters = filter_name_clusters(cluster_to_sentence, target, l)

        keep_info[target] = keep_clusters, radical_keep_clusters
        
        keep_labels = [[l for l in labels[target][t] if l in keep_clusters] for t in ['t1', 't2']]
        radical_keep_labels = [[l for l in labels[target][t] if l in radical_keep_clusters] for t in ['t1', 't2']]

        filtered_jsd[target] = compute_divergence_from_cluster_labels(keep_labels[0], keep_labels[1])
        
        radical_filtered_jsd[target] = compute_divergence_from_cluster_labels(radical_keep_labels[0], radical_keep_labels[1])
        print(target, filtered_jsd[target], radical_filtered_jsd[target])
        
    file_name_base = label_file.replace('.pkl', '')

    info_file = os.path.join(filter_folder, file_name_base + "_keep_info.pkl")
    pickle.dump(keep_info, open(info_file, 'wb'))
    
    res_file = os.path.join(filter_folder, file_name_base + "_result.csv")

    with open(res_file, 'w') as out:
        print('\t'.join(['word', 'filtered_jsd', 'clusters', 'radically_filtered_jsd', 'clusters']), file=out)
        for t in filtered_jsd.keys():
            print('\t'.join([str(x) for x in [t, filtered_jsd[t], len(keep_info[t][0]), radical_filtered_jsd[t], len(keep_info[t][1])]]), file=out)

def main():
    pickles = [f for f in os.listdir(input_folder) if f.endswith(".pkl")]

    sents = [p for p in pickles if p.startswith("sents")]
    labels = [p for p in pickles if p.startswith("aff_prop_labels")] # doesn't make sense for other clustering

    for label_file in labels:
        for l in languages:
            if l in label_file:
                break
        for e in emb:
            if e in label_file:
                break
        for sent_file in sents:
            if l in sent_file and e in sent_file:
                filter(label_file, sent_file, l)
                break


if __name__ == "__main__":
    # input folder should contain results from 'get_all_results_clustering_drifts.py'
    input_folder = sys.argv[1]
    filter_folder = input_folder.strip("/")+"_filtered"
    if not os.path.exists(filter_folder):
        os.makedirs(filter_folder)
    main()
