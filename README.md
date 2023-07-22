# Source Code Summarization via Encoder-Decoder with Attention
Implementation of proposed source code summarization method for CS 585 class project using an encoder-decoder method with attention for translating source code into descriptive comments. The base neural architecture for this model is adapted from the [dynet example](https://dynet.readthedocs.io/en/latest/examples.html). This architecture is tested both with and without pretrained word embeddings that were created using the word2vec algorithm.


## Description of the Repo
- All utilities, including the pickle files that contain the parallel corpus can be found in the `utils/` directory
- All data including the train/dev/test split dataset, model weights, and a sample of generated output from dev and test can be found in the `data/` directory.
- The sample output is named {embed | non_embed}_{dev | test}.nl and consists of one line per source data instance. These lines correspond to the lines in {dev | test}.nl that are the original source comment.
- All information that was used to create the word embeddings using word2vec is available in the `vector-maker/` directory. The top-level script used for this process is `w2v.sh`

## Procedure to run the testing code portion
Run the following two commands using docker installed on a machine running Ubuntu 16.04.01

1. `docker build -t hein-cs585-hw4 .`
2. `docker run hein-cs585-hw4`

This will display the BLEU score output for the Dev and Test portions of both the trained model with pretrained embeddings and the trained model without pretrained embeddings.

## Training details
- Training was done using the `main.py` script. Portions of this code are commented out depending on whether we are training the model with or without pretrained embeddings.
- The validation set output is also generated during the training process in order to assist in debugging the model.
- Training was done using the UA HPC Ocelote computer. Training sessions utilized one Nvidia Tesla P100 per session to train for a total of 40 epochs over approximately 6 hours
- To train on Ocelote the script `run-comm-gen.pbs` was used
