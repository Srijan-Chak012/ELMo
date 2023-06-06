# Assignment 4 - ELMO

## README

The code in this repository trains and evaluates an ELMo model on two widely-used natural language processing datasets: multinli and sst.

1. The multinli dataset is composed of sentence pairs from various genres that have been manually labeled for entailment, contradiction, and neutral relationships. This dataset is intended to test models' natural language inference abilities across different writing styles and domains. It has around 400,000 sentence pairs for training and 10,000 for testing.

2. The sst dataset includes movie reviews with sentiment labels ranging from very negative to very positive, along with a parse tree for each sentence, providing structural information for the models. The dataset contains approximately 11,000 movie reviews for training and 2,500 for testing.

These datasets are considered standard benchmarks for evaluating NLP models, and many top-performing models, including ELMo, have been trained and tested on them.

## File and Usage

Both the SST and the NLI datasets have been trained and used in the same file (elmo.py). The file can be run with the following command:

`python elmo.py`

However, to test the model on the NLI dataset, there are comments in the code that must be uncommented. Currently, the dataset runs on the SST dataset.

If you want to utilize pretrained models, you simply need to have them located in the same directory as the scripts. In this case, the scripts will automatically load the models for use. If the pretrained models are not present in the directory, the scripts will proceed to train new models from scratch.

To use the files, go to the following link and download the files:
![Link to files](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/srijan_chakraborty_research_iiit_ac_in/EvncEZjfdWBFjK88xxFkWlgBVTjBO0_3uCg8WXA2higwqg?e=ksHcyX)

It has 4 models:
1. Elmo pretrained on SST dataset
2. Elmo finetuned on SST dataset
3. Elmo pretrained on NLI dataset
4. Elmo finetuned on NLI dataset

There is a file `visualise.py` which is there purely to generate the Confusion Matrix visualisations. It is not required for the code to run.