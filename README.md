# TCTS-iM6A-2S: a two-stage deep learning framework for predicting RNA N6-methyladenosine sites across tissues and physiological systems

## Introduction

We introduce TCTS-iM6A-2S, a novel two-stage deep learning framework to identify tissue-conserved m6A sites and decipher their specificity at the physiological system level. The first stage distinguishes tissue-conserved from tissue-specific m6A sites; the second stage predicts m6A sites across five physiological systems (circulatory, digestive, nervous, respiratory, and urinary). TCTS-iM6A-2S integrates contextual embeddings from a pre-trained RNA language model and distributional features derived from Word2Vec. The fused features are then fed into a bidirectional long short-term memory network and fully connected layers for final classification. A Bahdanau attention module is incorporated within the first-stage classifier to capture and compress critical sequence information.


## Setup

### Requirements

Bio==1.8.5

biopython==1.83

gensim==4.4.0

matplotlib==3.11.2

networkx==3.2.1

numpy==2.5.5

scikit_learn==1.9.1

torch==2.7.1+cu118

tqdm==4.67.3

transformers==4.57.6


### Create an environment

We highly recommend using a virtual environment for the installation of TCTS-iM6A-2S and its dependencies. A virtual environment can be created and (de)activated as follows by using [conda](https://conda.io/docs/):

```sh
# create
conda create -n TCTS-iM6A-2S python=3.9
# activate
conda activate TCTS-iM6A-2S
```

### Install TCTS-iM6A-2S

After creating and activating the environment, download and install TCTS-iM6A-2S (**latest version**) from github:

```sh
git clone https://github.com/mengzhanggggg/TCTS-iM6A-2S.git
cd TCTS-iM6A-2S
pip install -r requirements.txt
```

### Quick Start

After cloning or entering the project directory, use `train.py` for training and `predict_first.py` / `predict_system.py` for prediction. Test is split by system type (circulatory, digestive, nervous, respiratory, and urinary) for separate models and checkpoints.


### Training
In the `utils/config.py` file, you can adjust the hyperparameters. 

>In the `utils/config.py`, the meaning of the variables is explained as follows:
>
>> ***seed*** is the seed for model initialization.
>> 
>> ***batchSize*** is the batchsize of training.
>>
>> ***numEpochs*** is the largest number of training epochs.
>> 
>> ***earlyStop*** is the parameter corresponding to the early stop method.
>> 
>> ***lr*** is the learning rate of training.
>> 
>> ***savePath*** is the folder where the model is saved.
>> 
>> ***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.

If you don’t modify the file, the default parameters will be used for training. To start the training, just run the following command:

```bash
python train.py
```
After training, you will find the trained model saved in the `checkpoints` folder. You can use this model for making predictions.

### Prediction
For prediction,if your file to be tested is in FASTA format. You can place the file in the working directory and run the following command to get the prediction results:
```bash
python predict.py
```

In addition, you can test your sequence as follows:


## Citation

Xiuyan Gao†, Jian Liu†, Anyan Yao, Xiaomin Ren, Hongxia Zheng, Yuna Zhang, Lewei Dong, Xiaofeng Song, Yifei Gao, Leyi Wei, Meng Zhang*, "TCTS-iM6A-2S: a two-stage deep learning framework for predicting RNA N6-methyladenosine sites across tissues and physiological systems".
