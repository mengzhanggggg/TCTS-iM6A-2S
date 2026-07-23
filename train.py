from cv_train import *
from getfeatures import *
from utils.config import *
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

features = SpliceBERT_Word2Vec(fasta_path="data/501/501_traindata.fasta")
cv_models = cv_train()
cv_models.train(features)
