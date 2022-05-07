import time
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from workspace.util.Encoding import user_label_encoder
from workspace.util.User import get_user_list, combine
from workspace.util.combine_dataset import encode_df

from sklearn.metrics import r2_score

dataset_path = "../../dataset"
model_path = "./models"


class life_label_Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :].clone().detach()
        y = self.y[idx].clone().detach()
        return X, y


class VanillaRnn(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_size, num_layers, num_classes, device):
        super(VanillaRnn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        # (batch_size, sequence_size, input_size)
        # => (batch_size, hidden_size)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        # (batch_size, hidden_size)
        # => (batch_size, num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initialize first hidden state
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)

        # drop last hidden state
        out, _ = self.rnn(x, h0)

        out = self.fc(out)

        # get last sequence
        out = out[:, -1, :]

        return out


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_size, num_layers, num_classes, device):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.ln = nn.LayerNorm(input_size, device=device)

        # (batch_size, sequence_size, input_size)
        # => (batch_size, hidden_size)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        # (batch_size, hidden_size)
        # => (batch_size, num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # initialize first hidden state
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)

        x = self.ln(x)
        # drop last hidden state
        out, _ = self.GRU(x, h0)

        out = self.fc(out)

        # get last sequence
        out = out[:, -1, :]

        return out


class GRU_bi(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_size, num_layers, num_classes, device):
        super(GRU_bi, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_size = sequence_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device

        self.ln = nn.LayerNorm(input_size, device=device)

        # (batch_size, sequence_size, input_size)
        # => (batch_size, hidden_size)
        self.GRU = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # (batch_size, hidden_size)
        # => (batch_size, num_classes)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        # initialize first hidden state
        h0 = torch.zeros(self.num_layers*2, x.size()[0], self.hidden_size).to(self.device)

        x = self.ln(x)
        # drop last hidden state
        out, _ = self.GRU(x, h0)

        out = self.fc(out)

        # get last sequence
        out = out[:, -1, :]

        return out