import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.qat import Embedding
from utils.config import *
from torch.nn.utils.weight_norm import weight_norm

params = config()

class BahdanauAttention(nn.Module):
    """
    input: from RNN module h_1, ... , h_n (batch_size, seq_len, units*num_directions),
                                    h_n: (num_directions, batch_size, units)
    return: (batch_size, num_task, units)
    """
    def __init__(self,in_features, hidden_units,num_task):
        super(BahdanauAttention,self).__init__()
        self.W1 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.W2 = nn.Linear(in_features=in_features,out_features=hidden_units)
        self.V = nn.Linear(in_features=hidden_units, out_features=num_task)

    def forward(self, hidden_states, values):
        hidden_with_time_axis = torch.unsqueeze(hidden_states,dim=1)
        # print("values device:", values.device)
        # print("hidden_with_time_axis device:", hidden_with_time_axis.device)
        # print("W1.weight device:", self.W1.weight.device)
        # print("W2.weight device:", self.W2.weight.device)
        # print("V.weight device:", self.V.weight.device)
        score  = self.V(nn.Tanh()(self.W1(values)+self.W2(hidden_with_time_axis)))
        attention_weights = nn.Softmax(dim=1)(score)
        values = torch.transpose(values,1,2)   # transpose to make it suitable for matrix multiplication
        #print(attention_weights.shape,values.shape)
        context_vector = torch.matmul(values,attention_weights)
        context_vector = torch.transpose(context_vector,1,2)
        return context_vector, attention_weights


class TCTS_iM6A_2L(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super(TCTS_iM6A_2L, self).__init__(*args, **kwargs)

        self.bidirectional = nn.LSTM(
            input_size=640,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(256, 128),  # 41-5248 101-12928 301-38528 601-76928 501-64128 201-25728 401-51328
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.dense2 = nn.Sequential(
            nn.Linear(128, 20),
            nn.Dropout(0.2),
            nn.ReLU(),
        )

        self.main_output = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        self.Attention = BahdanauAttention(in_features=256, hidden_units=100, num_task=1)
        self.to(params.device)

    def forward(self, input):
        w2v, embedding = input
        w2v = w2v.to(params.device)
        embedding = embedding.to(params.device)

        w2v = w2v.permute(0, 2, 1)
        w2v = F.interpolate(w2v, size=501, mode='linear', align_corners=False)
        w2v = w2v.permute(0, 2, 1)

        embedding = embedding.permute(0, 2, 1)
        embedding = F.interpolate(embedding, size=501, mode='linear', align_corners=False)
        embedding = embedding.permute(0, 2, 1)

        concat = torch.cat((w2v, embedding), dim=2)
        output, (h, c) = self.bidirectional(concat)
        h = torch.cat([h[0], h[1]], dim=-1)  # [B, 512]
        context_vector, attention_weights = self.Attention(h, output)
        context_vector = context_vector.squeeze(1)


        dense1_out = self.dense1(context_vector)
        dense2_out = self.dense2(dense1_out)
        logits = self.main_output(dense2_out).squeeze(1)
        preds = self.sigmoid(logits)
        return preds, concat, output#, dense1_out, dense2_out



