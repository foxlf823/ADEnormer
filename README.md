# ADEnormer
This repository contains the code of the system that we used to paticipate the [FDA Adverse Drug Event Evaluation Challenge](https://sites.mitre.org/adeeval/). 
The system contains two parts, a NER model to identify adverse drug events and a normalization model to map them to the Medical Dictionary for Regulatory Activities ([MedDRA](https://www.meddra.org)).
For more details, please refer to XXX.
If you have any question, please leave a issue or contact foxlf823@gmail.com

We refer to several repositories. Highly appreciate!
* [NCRF++](https://github.com/jiesutd/NCRFpp)
* [Sieve-Based Entity Linking](https://github.com/jennydsuza9/disorder-normalizer)
* [Multinomial Adversarial Nets](https://github.com/ccsasuke/man)
* [ELMoForManyLangs](https://github.com/HIT-SCIR/ELMoForManyLangs)

## Setup
* Install python 3.6, pytorch 1.0.1, nltk and other necessary packages.
* Compile "PorterStemmer.java".
* Download all the support files in [dropbox](https://www.dropbox.com/sh/x7sv3tjy960j21o/AABJI838euVffEvKGNzoDdVra?dl=0) and put them in the same directory.
* Install ELMoForManyLangs.

## Quick Start
Train a NER model.

`
python main_fda.py -train_file ./sample -dev_file ./sample -test_file ./sample -iter 1 -ner_number_normalized -schema BIOHD_1234 -types OSE_Labeled_AE -word_emb_file ./fasttext_lower_0norm_200d.vec -hidden_dim 200 -elmo -char_hidden_dim 50 -nbest 10 -config ./config.txt -output output_ner -random_seed 0 -dropout 0.5
`

Train a similarity-based normalization model.

`
python main_fda.py -train_file ./sample -dev_file ./sample -test_file ./sample -norm_vsm -types OSE_Labeled_AE -schema BIOHD_1234 -whattodo 2 -config ./config.txt -iter 1 -output output_vsm -random_seed 0
`

Train a classification-based normalization model.

`
python main_fda.py -train_file ./sample -dev_file ./sample -test_file ./sample -norm_neural -types OSE_Labeled_AE -schema BIOHD_1234 -whattodo 2 -iter 1 -config ./config.txt -output output_neural -random_seed 0
`

Test on new unannotated data.

`
python main_fda.py -norm_rule -norm_vsm -norm_neural -ensemble vote -test_file ./sample -ner_number_normalized -schema BIOHD_1234 -types OSE_Labeled_AE -whattodo 3 -train_file ./sample -dev_file "" -hidden_dim 200 -elmo -char_hidden_dim 50 -nbest 10 -output output -predict predict -config ./config.txt
`

## Reproduce the Results in the Competition
Assuming "ose_xml_training_20181101" is the training set and "UnannotatedTestCorpus" is the test set. 
To reproduce our results using pre-trained models, run

`
python main_fda.py -norm_rule -norm_vsm -norm_neural -ensemble vote -test_file ./UnannotatedTestCorpus -ner_number_normalized -schema BIOHD_1234 -types OSE_Labeled_AE -whattodo 3 -train_file ./ose_xml_training_20181101 -dev_file "" -hidden_dim 200 -elmo -char_hidden_dim 50 -nbest 10 -output pretrained_model -predict predict -config ./config_fda_challenge.txt -gpu 2
`

If you want to train the system from scratch, do Step 1-3.

Step 1

`
python main_fda.py -train_file ./ose_xml_training_20181101 -dev_file "" -test_file ./UnannotatedTestCorpus -iter 50 -ner_number_normalized -schema BIOHD_1234 -types OSE_Labeled_AE -gpu 2 -word_emb_file ../fasttext_lower_0norm_200d.vec -hidden_dim 200 -elmo -char_hidden_dim 50 -nbest 10 -config ./config_fda_challenge.txt -output output_ner -random_seed 0 -dropout 0.5
`

Step 2

`
python main_fda.py -train_file ./ose_xml_training_20181101 -dev_file "" -test_file ./UnannotatedTestCorpus -norm_vsm -types OSE_Labeled_AE -schema BIOHD_1234 -whattodo 2 -config ./config_fda_challenge.txt -iter 45 -output output_vsm -gpu 2 -random_seed 0
`

Step 3

`
python main_fda.py -train_file ./ose_xml_training_20181101 -dev_file "" -test_file ./UnannotatedTestCorpus -norm_neural -types OSE_Labeled_AE -schema BIOHD_1234 -whattodo 2 -iter 5 -gpu 2 -config ./config_fda_challenge.txt -output output_neural -random_seed 0
`

Then copy the models in "output_ner", "output_vsm" and "output_neural" to a new directory "output". Do Step 4.

Step 4

`
python main_fda.py -norm_rule -norm_vsm -norm_neural -ensemble vote -test_file ./UnannotatedTestCorpus -ner_number_normalized -schema BIOHD_1234 -types OSE_Labeled_AE -whattodo 3 -train_file ./ose_xml_training_20181101 -dev_file "" -hidden_dim 200 -elmo -char_hidden_dim 50 -nbest 10 -output output -predict predict -config ./config_fda_challenge.txt -gpu 2
`
