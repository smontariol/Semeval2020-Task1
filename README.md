# Semeval_2020_Task_1

Official repository for Semeval 2020 task 1 - Discovery Team system. Published results were produced in Python 3 programming environment on Linux Mint 18 Cinnamon operating system. Instructions for installation assume the usage of PyPI package manager.<br/>


## Installation, documentation ##

Install dependencies if needed: pip install -r requirements.txt

### To reproduce the results published in the paper run the code in the command line using following commands: ###

#### Fine-tune language model on the SemEval corpus:<br/>

Generate language model train and test sets:<br/>
```
python build_lm_train_test.py  --corpus_paths pathToCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --lm_train_test_folder pathToOutputFolder
```
Fine-tune BERT model:<br/>

```
python fine-tune_BERT.py --train_data_file pathToTrainSet --output_dir pathToOutputModelDir --eval_data_file pathToTestSet --model_name_or_path modelForSpecificLanguage --line_by_line --mlm --do_train --do_eval --evaluate_during_training
```

For '--model_name_or_path' parameter, see the paper for info about which models we use for each language.

#### Extract BERT embeddings:<br/>

Preprocess corpus:<br/>

```
python preprocess_semeval_corpora.py  --corpus_paths pathToCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --lm_train_test_folder pathToOutputFolder
```

Extract embeddings from the preprocessed corpus in .txt format:<br/>

```
python extract_embeddings.py --corpus_paths pathToPreprocessedCorpusSlicesSeparatedBy';' --target_path pathToSemEvalTargetFile --language language --path_to_fine_tuned_model pathToFineTunedModel --embeddings_path pathToOutputEmbeddingFile --concat 
```

This creates a pickled file containing all contextual embeddings for all target words.<br/>

#### Get results:<br/>

Conduct clustering and measure semantic shift with JSD:<br/>

```
python calculate_semantic_change.py --language language --embeddings_path pathToInputEmbeddingFile --semeval_results pathToOutputResultsDir
```

This script takes the pickled embedding file as an input and creates several files, a csv file containing JSD scores for all clustering methods for each target word, files containing cluster labels for each embedding , files containing cluster centroids and a file containing context (sentence) mapped to each embedding.<br/>

Generate SemEval submission files for task 1 (binary classification using stopword tresholding method) and task 2 (ranking) using a specific clustering method:<br/>

```
python make_semeval_answer_file.py --language language --results_file pathToInputResultsCSVFile --method clusteringMethod --target_path pathToSemEvalTargetFile
```

This script takes the CSV file generated in the previous step as an input and creates SemEval submission files for a specific clustering method (options are 'aff_prop', 'kmeans_5', 'kmeans_7', 'averaging') and language.<br/>

Generate SemEval submission files for task 1 (binary classification) using time period specific cluster method:<br/>

```
python get_period_specific_clusters.py --language language --results_file pathToInputClusterLabelFile --target_path pathToSemEvalTargetFile
```
This script takes one of the cluster labels files generated with the calculate_semantic_change.py script as an input. Use the "--dynamic_treshold" flag if your input labels are for affinity propagation clustering.<br/>

#### Extra:<br/>

Filter Named entities from clusters:<br/>

```
python filter_ner.py --language language --input_sent_file pathToFileWithSentences --input_label_file pathToInputClusterLabelFile --output_dir_path pathToOutputResultsDir
```

This script takes one of the cluster labels files and a sentence file generated with the calculate_semantic_change.py script as an input. It is only appropriate for filtering of affinity propagation clusters.<br/>

Script for ensembling of static word2vec and contextual embeddings:<br/>

```
python ensembling_script.py --language language --method_1 clusteringMethodName --input_file_method_1 pathToInputResultsCSVFile --method_2 word2vecMethodName --input_file_method_2 pathToInputWord2VecFile --output_file_path OutputCSVResultsFile
```

#### If something is unclear, check the default arguments for each script. If you still can't make it work, feel free to contact us :).

