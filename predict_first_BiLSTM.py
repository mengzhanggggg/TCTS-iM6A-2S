from model_first import *
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
from utils.config import *
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
from transformers import logging
import matplotlib.pyplot as plt
logging.set_verbosity_error()

def caculate_metric(pred_y, labels, pred_prob):

    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index in range(test_num):
        if int(labels[index]) == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    ACC = float(tp + tn) / test_num

    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)

    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)
    AUC = roc_auc_score(labels, pred_prob)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AP = average_precision_score(labels, pred_prob, average='macro', pos_label=1, sample_weight=None)

    metric = torch.tensor([Sensitivity, Specificity, ACC, F1, Precision, Recall, AUC, AP, MCC])

    # ROC(fpr, tpr, AUC)
    # PRC(recall, precision, AP)
    roc_data = [fpr, tpr, AUC]
    prc_data = [recall, precision, AP]
    return metric, roc_data, prc_data

def sequence_to_matrix(sequence, model):
    return torch.tensor(np.array([model.wv[kmer] for kmer in sequence if kmer in model.wv]))

SPLICEBERT_PATH = "models_folder/SpliceBERT.1024nt"


def data_process(seq):
    k_RNA = [seq[j:j + 3] for j in range(len(seq) - 3 + 1)]
    sp_fea = []
    tokenizer = AutoTokenizer.from_pretrained(SPLICEBERT_PATH)
    sp_model = AutoModel.from_pretrained(SPLICEBERT_PATH)
    sp_model.eval() 
    seq = ' '.join(list(seq.upper().replace("U", "T")))
    input_ids = tokenizer.encode(seq, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():  
        output = sp_model(input_ids)
    last_hidden_state = output.last_hidden_state
    sp_fea.append(last_hidden_state.cpu())
    sp_fea = torch.cat(sp_fea, dim=0)
    w2v_model = Word2Vec.load('model/first/w2v_model.model')
    with torch.no_grad():
        w2v_fea = sequence_to_matrix(k_RNA, w2v_model).unsqueeze(0)
    return (w2v_fea, sp_fea)

def predict(seq, model_list):
    input = data_process(seq)
    res = 0.0
    with torch.no_grad():
        for _, model in enumerate(model_list):
            y, _, _ = model(input)
            res += y.detach()
    res /= 5
    return res




fasta_path="data/501/501_testdata.fasta"

# Loading model
model1 = TCTS_iM6A_2L()
model1.load_state_dict(torch.load('model/first/cv1_0.909_10.181.pkl'))
model1.eval()

model2 = TCTS_iM6A_2L()
model2.load_state_dict(torch.load('model/first/cv2_0.918_10.058.pkl'))
model2.eval()

model3 = TCTS_iM6A_2L()
model3.load_state_dict(torch.load('model/first/cv3_0.919_10.039.pkl'))
model3.eval()

model4 = TCTS_iM6A_2L()
model4.load_state_dict(torch.load('model/first/cv4_0.926_10.060.pkl'))
model4.eval()

model5 = TCTS_iM6A_2L()
model5.load_state_dict(torch.load('model/first/cv5_0.918_10.420.pkl'))
model5.eval()

model_list = [model1, model2, model3, model4, model5]

records = list(SeqIO.parse(fasta_path, 'fasta'))

fasta_seqs = [str(record.seq) for record in records]
fasta_seq_names = [record.id for record in records]
label_real = [1 if name.startswith('P') else 0 if name.startswith('N') else None for name in fasta_seq_names]
pred_prob, label_pred = [], []
for i, seq in enumerate(fasta_seqs):
    res = predict(seq, model_list)
    res_numpy = res.cpu().numpy()
    pred_prob.append(res_numpy)
print(pred_prob)
print(label_real)
for prob in pred_prob:
    if prob >= 0.5:
        label_pred.append(1)
    else:
        label_pred.append(0)
print(label_pred)

metric, roc_data, prc_data = caculate_metric(label_pred, label_real, pred_prob)
print(metric)
