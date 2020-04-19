import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pickle
import gc
import re
import pandas as pd
import json
from collections import defaultdict
from tokenizers import (BertWordPieceTokenizer)



def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


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


def tokens_to_batches(ds, tokenizer, batch_size, max_length, target_words, lang):

    batches = []
    batch = []
    batch_counter = 0

    frequencies = defaultdict(int)

    print('Dataset: ', ds)
    counter = 0
    with open(ds, 'r', encoding='utf8') as f:

        for line in f:
            counter += 1

            if counter % 50000 == 0:
                print('Num articles: ', counter)

            text = line.strip()
            contains = False

            for w in target_words:
                if w.strip() in set(text.split()):
                    frequencies[w] += text.count(w)
                    contains = True

            if contains:

                if lang=='swedish':
                    tokenized_text = tokenizer.encode(text)
                else:
                    marked_text =  "[CLS] " + text + " [SEP]"
                    tokenized_text = tokenizer.tokenize(marked_text)

                for i in range(0, len(tokenized_text), max_length):

                    batch_counter += 1
                    if lang=='swedish':
                        input_sequence = tokenized_text.tokens[i:i + max_length]
                        indexed_tokens = tokenized_text.ids[i:i + max_length]

                    else:
                        input_sequence = tokenized_text[i:i + max_length]
                        indexed_tokens = tokenizer.convert_tokens_to_ids(input_sequence)

                    batch.append((indexed_tokens, input_sequence))

                    if batch_counter % batch_size == 0:
                        batches.append(batch)
                        batch = []

    print()
    print('Tokenization done!')
    print('len batches: ', len(batches))
    print('Word frequencies:')

    for w, freq in frequencies.items():
        print(w + ': ', str(freq))

    return batches


def get_token_embeddings(batches, model, batch_size):

    input_token_embeddings = []
    encoder_token_embeddings = []
    tokenized_text = []
    counter = 0

    for batch in batches:
        counter += 1
        if counter % 1000 == 0:
            print('Generating embedding for batch: ', counter)
        lens = [len(x[0]) for x in batch]
        max_len = max(lens)
        tokens_tensor = torch.zeros(batch_size, max_len, dtype=torch.long).cuda()
        segments_tensors = torch.ones(batch_size, max_len, dtype=torch.long).cuda()
        batch_idx = [x[0] for x in batch]
        batch_tokens = [x[1] for x in batch]

        for i in range(batch_size):
            length = len(batch_idx[i])
            for j in range(max_len):
                if j < length:
                    tokens_tensor[i][j] = batch_idx[i][j]

        #print("Input shape: ", tokens_tensor.shape)
        #print(tokens_tensor)

        # Predict hidden states features for each layer
        with torch.no_grad():
            model_output = model(tokens_tensor, segments_tensors)
            encoded_layers = model_output[-1][1:]  #last four layers of the encoder
            input_embeddings = model_output[-1][0]

        for batch_i in range(batch_size):
            encoder_token_embeddings_example = []
            input_token_embeddings_example = []
            tokenized_text_example = []


            # For each token in the sentence...
            for token_i in range(len(batch_tokens[batch_i])):

                # Holds 12 layers of hidden states for each token
                hidden_layers = []

                # For each of the 12 layers...
                for layer_i in range(len(encoded_layers)):
                    # Lookup the vector for `token_i` in `layer_i`
                    vec = encoded_layers[layer_i][batch_i][token_i]

                    hidden_layers.append(vec)

                hidden_layers = torch.sum(torch.stack(hidden_layers)[-4:], 0).reshape(1, -1).detach().cpu().numpy()

                encoder_token_embeddings_example.append(hidden_layers)
                input_token_embeddings_example.append((input_embeddings[batch_i][token_i]).reshape(1, -1).detach().cpu().numpy())
                tokenized_text_example.append(batch_tokens[batch_i][token_i])

            encoder_token_embeddings.append(encoder_token_embeddings_example)
            input_token_embeddings.append(input_token_embeddings_example)
            tokenized_text.append(tokenized_text_example)


        # Sanity check the dimensions:
        #print("Number of tokens in sequence:", len(token_embeddings))
        #print("Number of layers per token:", len(token_embeddings[0]))

    return input_token_embeddings, encoder_token_embeddings, tokenized_text


def get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict, concat=False):
    targets = list(target_dict.keys())
    vocab_vectors = {}

    for ds in datasets:

        period = ds[-5:-4]

        all_batches = tokens_to_batches(ds, tokenizer, batch_size, max_length, targets, lang)
        targets = set(targets)
        #print(targets)
        chunked_batches = chunks(all_batches, 1000)
        num_chunk = 0

        for batches in chunked_batches:
            num_chunk += 1
            print('Chunk ', num_chunk)

            #get list of embeddings and list of bpe tokens
            input_token_embeddings, encoder_token_embeddings, tokenized_text = get_token_embeddings(batches, model, batch_size)

            splitted_tokens = []
            if not concat:
                input_splitted_array = np.zeros((1, 768))
                encoder_splitted_array = np.zeros((1, 768))
            else:
                input_splitted_array = []
                encoder_splitted_array = []
            prev_token = ""
            input_prev_array = np.zeros((1, 768))
            encoder_prev_array = np.zeros((1, 768))

            #go through text token by token
            for example_idx, example in enumerate(tokenized_text):
                for i, token_i in enumerate(example):

                    if lang != 'swedish':
                        context = tokenizer.convert_tokens_to_string(example)
                    else:
                        context = " ".join(example).replace(" ##", "")

                    input_array = input_token_embeddings[example_idx][i]
                    encoder_array = encoder_token_embeddings[example_idx][i]

                    #word is split into parts
                    if token_i.startswith('##'):

                        #add words prefix (not starting with ##) to the list
                        if prev_token:
                            splitted_tokens.append(prev_token)
                            prev_token = ""
                            if not concat:
                                input_splitted_array = input_prev_array
                                encoder_splitted_array = encoder_prev_array
                            else:
                                input_splitted_array.append(input_prev_array)
                                encoder_splitted_array.append(encoder_prev_array)


                        #add word to splitted tokens array and add its embedding to splitted_array
                        splitted_tokens.append(token_i)
                        if not concat:
                            input_splitted_array += input_array
                            encoder_splitted_array += encoder_array
                        else:
                            input_splitted_array.append(input_array)
                            encoder_splitted_array.append(encoder_array)

                    #word is not split into parts
                    else:
                        if token_i in targets:
                            #print(token_i)
                            if i == len(example) - 1 or not example[i + 1].startswith('##'):
                                if target_dict[token_i] in vocab_vectors:
                                    #print("In vocab: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    if 't' + period in vocab_vectors[target_dict[token_i]]:
                                        vocab_vectors[target_dict[token_i]]['t' + period + '_input'].append(input_array.squeeze())
                                        vocab_vectors[target_dict[token_i]]['t' + period].append(encoder_array.squeeze())
                                        vocab_vectors[target_dict[token_i]]['t' + period + '_text'].append(context)
                                    else:
                                        vocab_vectors[target_dict[token_i]]['t' + period] = [encoder_array.squeeze()]
                                        vocab_vectors[target_dict[token_i]]['t' + period + '_input'] = [input_array.squeeze()]
                                        vocab_vectors[target_dict[token_i]]['t' + period + '_text'] = [context]
                                else:
                                    #print("Not in vocab yet: ", token_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[target_dict[token_i]] = {'t' + period + '_input':[input_array.squeeze()], 't' + period:[encoder_array.squeeze()], 't' + period + '_text':[context]}
                                    #vocab_vectors[target_dict[token_i]] = {'t' + period + '_input': [input_array.squeeze()], 't' + period: [encoder_array.squeeze()]}

                        #check if there are words in splitted tokens array, calculate average embedding and add the word to the vocabulary
                        if splitted_tokens:
                            if not concat:
                                input_sarray = input_splitted_array / len(splitted_tokens)
                                encoder_sarray = encoder_splitted_array / len(splitted_tokens)
                            else:
                                input_sarray = np.concatenate(input_splitted_array, axis=1)
                                encoder_sarray = np.concatenate(encoder_splitted_array, axis=1)
                            stoken_i = "".join(splitted_tokens).replace('##', '')

                            if stoken_i in targets:
                                if target_dict[stoken_i] in vocab_vectors:
                                    #print("S In vocab: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    if 't' + period in vocab_vectors[target_dict[stoken_i]]:
                                        vocab_vectors[target_dict[stoken_i]]['t' + period + '_input'].append(input_sarray.squeeze())
                                        vocab_vectors[target_dict[stoken_i]]['t' + period].append(encoder_sarray.squeeze())
                                        vocab_vectors[target_dict[stoken_i]]['t' + period + '_text'].append(context)
                                    else:
                                        vocab_vectors[target_dict[stoken_i]]['t' + period] = [encoder_sarray.squeeze()]
                                        vocab_vectors[target_dict[stoken_i]]['t' + period + '_input'] = [input_sarray.squeeze()]
                                        vocab_vectors[target_dict[stoken_i]]['t' + period + '_text'] = [context]

                                    #vocab_vectors[target_dict[stoken_i]]['t' + period + '_text'].append(context)
                                else:
                                    #print("S Not in vocab yet: ", stoken_i + '_' + period, list(vocab_vectors.keys()))
                                    vocab_vectors[target_dict[stoken_i]] = {'t' + period + '_input': [input_sarray.squeeze()], 't' + period: [encoder_sarray.squeeze()], 't' + period + '_text': [context]}
                                    #vocab_vectors[target_dict[stoken_i]] = {'t' + period + '_input': [input_sarray.squeeze()], 't' + period: [encoder_sarray.squeeze()]}

                            splitted_tokens = []
                            if not concat:
                                input_splitted_array = np.zeros((1, 768))
                                encoder_splitted_array = np.zeros((1, 768))
                            else:
                                input_splitted_array = []
                                encoder_splitted_array = []

                        input_prev_array = input_array
                        encoder_prev_array = encoder_array
                        prev_token = token_i

            del input_token_embeddings
            del encoder_token_embeddings
            del tokenized_text
            del batches
            gc.collect()

            '''for k, v in vocab_vectors.items():
                print(k)
                input = v[0]
                encoder = v[1]
                context = v[2]
                print(len(input))
                print(len(encoder))
                print(len(context))
                print(context[0])'''

        print('Sentence embeddings generated.')

    print("Length of vocab after training: ", len(vocab_vectors.items()))

    with open(embeddings_path.split('.')[0] + '.pickle', 'wb') as handle:
        pickle.dump(vocab_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    gc.collect()




if __name__ == '__main__':
    batch_size = 8
    max_length = 256

    data_folder = 'data/semeval_data/'

    langs = ['english', 'latin', 'german', 'swedish']
    concats = [True, False]
    fine_tuned = True

    for lang in langs:
        for concat in concats:
            if lang in ['latin', 'english']:
                datasets = [data_folder + lang + '/' + lang + '_clean_1.txt',
                            data_folder + lang + '/' + lang + '_clean_2.txt',]
            else:
                datasets = [data_folder + lang + '/' + lang + '_1.txt',
                            data_folder + lang + '/' + lang + '_2.txt', ]


            if lang == 'swedish':
                tokenizer = BertWordPieceTokenizer("data/semeval_data/swedish/vocab_swebert.txt", lowercase=True, strip_accents=False)
                if fine_tuned:
                    state_dict = torch.load("models/model_swedish/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('af-ai-center/bert-base-swedish-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/swedish_5_epochs_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/swedish_5_epochs.pickle'
                else:
                    model = BertModel.from_pretrained('af-ai-center/bert-base-swedish-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/swedish_pretrained_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/swedish_pretrained.pickle'
            elif lang == 'german':
                tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
                if fine_tuned:
                    state_dict = torch.load("models/model_german_cased/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-german-cased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/german_5_epochs_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/german_5_epochs.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-german-cased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/german_pretrained_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/german_pretrained.pickle'

            elif lang == 'english':
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
                if fine_tuned:
                    state_dict = torch.load("models/model_english_epoch_8/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/english_5_epochs_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/english_5_epochs.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/english_pretrained_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/english_pretrained.pickle'

            elif lang == 'latin':
                tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
                if fine_tuned:
                    state_dict = torch.load("models/model_latin/epoch_5/pytorch_model.bin")
                    model = BertModel.from_pretrained('bert-base-multilingual-uncased', state_dict=state_dict, output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/latin_5_epochs_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/latin_5_epochs.pickle'
                else:
                    model = BertModel.from_pretrained('bert-base-multilingual-uncased', output_hidden_states=True)
                    if concat:
                        embeddings_path = 'embeddings/latin_pretrained_concat.pickle'
                    else:
                        embeddings_path = 'embeddings/latin_pretrained.pickle'


            model.cuda()
            model.eval()

            target_dict = get_targets(data_folder + lang + '/targets.txt', lang)


            #print(target_dict)

            get_time_embeddings(embeddings_path, datasets, tokenizer, model, batch_size, max_length, lang, target_dict=target_dict, concat=concat)


