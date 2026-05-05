# mariNER: Multi-cased Augmentation for Robustness to Idiosyncrasies in Named Entity Recognition

- ```load_data.py```: This file contains the ```AugmentedDataLoader``` class, which loads datasets from HuggingFace based on user input to ```run.py``` and performs relevant preprocessing for data augmentation.
  - Datasets loaded from HF: CoNLL-2003 ('eriktks/conll2003'); CoNLL-2002 Spanish + Dutch, GermEval 2014, TurkuNLP, MasakhaNER 2.0 Zulu, Tweebank NER, WNUT17 (all from 'bltlab/open-ner-core-types')
- ```models.py```: The ```NERModel``` class used for all model training, testing, and grid search is located in this file. Each instantiated model contains an ```AugmentedDataLoader``` instance with the appropriate dataset. Grid search hyperparameters are pre-defined in the relevant function.
- ```run.py```: This file instantiates ```AugmentedDataLoader```s and ```NERModel```s based on user input and runs experiments accordingly. It accepts arguments from the command line for selecting datasets, model types (base/augmented), and whether to perform grid search or simply train/evaluate. The optimal learning rate, decay rate, and training batch size for each model/dataset combination (as used in evaluation for the final paper) are hardcoded into the main method.
- ```logs_final```: This directory contains the log output from running experiments on the cluster for final test set results.
  - ```all_langs_46217.log```: This file contains original/lowercase/uppercase test set results for baseline and augmented models trained and tested on CoNLL03, CoNLL02, GermEval 2014, TurkuNLP, and MasakhaNER 2.0 Zulu.
  - ```socmed_46221.log```: This file contains original/lowercase/uppercase test set results for baseline and augmented models trained on CoNLL03 and tested on Tweebank and WNUT17.
- ```scripts```: This directory contains the bash scripts used to run all experiments on the Brandeis GPU cluster.
  - ```all.sh```: Runs all sixteen model/dataset combinations.
  - ```socmed.sh```: Runs baseline + augmented models that train on CoNLL03 and test on Tweebank/WNUT17.
  - ```[language prefix].sh```: Runs baseline + augmented models that train/test on dataset in target language.
  - ```[language prefix]_gs.sh```: Runs grid search for baseline + augmented models in target language.