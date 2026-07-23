import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn.qat import Embedding
from utils.config import *
from torch.nn.utils.weight_norm import weight_norm

params = config()

class TCTS_iM6A_2L(nn.Module):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super(TCTS_iM6A_2L, self).__init__(*args, **kwargs)

        self.bidirectional = nn.LSTM(
            input_size=640,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.flatten = nn.Flatten()

        self.dense1 = nn.Sequential(
            nn.Linear(64128, 128),  # 41-5248 101-12928 301-38528 601-76928 501-64128 201-25728 401-51328
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
        lstm, (h, c) = self.bidirectional(concat)
        dense1_out = self.dense1(self.flatten(lstm))
        dense2_out = self.dense2(dense1_out)
        logits = self.main_output(dense2_out).squeeze(1)
        preds = self.sigmoid(logits)
        return preds, concat, lstm#, dense1_out, dense2_out



