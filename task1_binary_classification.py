
import pickle
import os
import collections

langs = ['english', 'latin', 'swedish', 'german']


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



def classify(targets, lang, results_folder, ending, clustering, treshold=2, dynamic = False):



    clustering_file = os.path.join(results_folder, clustering + "_labels_" + lang + ending + '.pkl')
    cluster_data = pickle.load(open(clustering_file, 'rb'))

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

        changed = False
	# average number of words per cluster
	if dynamic == true:
            size_clusters = [n1 + n2 for [n1, n2] in list(df_clusters.values())]
            dynamic_threshold = 2 * np.mean(size_clusters)
	else:
	    dynamic_threshold = 10

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





def extract_threshold(lang, stop_data):
    ''' stop_data is a csv with the JDS measure of the stopwords in the language "lang"
    '''
    thresholds=collections.defaultdict(lambda : collections.defaultdict(dict))
    print(lang)

    # extract threshold from the data
    # stop_data['aff_prop'].hist()
    # stop_data['averaging'].hist()
    threshold1 = len(stop_data) / 10
    n_bins = 10
    threshold_low = 0
    print('threshold1:', threshold1)
    for method in ['aff_prop', 'kmeans_7', 'averaging']:
        if method == 'averaging':
            n_bins = 20
        count, division = np.histogram(data_clean[method], bins=n_bins)
        for i, (c, d) in enumerate(zip(count[1:], division[1:])):
            if c < threshold1:
                threshold_low = d
                index_stop = i
                #print("############ THRESHOLD:", d, "##############")
                break
        threshold_high=division[i+2]

        thresholds[lang][method]['low'] = round(threshold_low, 3)
        thresholds[lang][method]['high'] = round(threshold_high, 3)

    return thresholds



data_folder = 'data/semeval_data/'
lang = 'english'
target_dict = get_targets(data_folder + lang + '/targets.txt', lang)
targets = target_dict.values()
results_folder = 'semeval_results/fine-tuned'
ending = '_fine_tuned_concat'
clustering = 'kmeans_7'


classify(targets, lang, results_folder, ending, clustering)
