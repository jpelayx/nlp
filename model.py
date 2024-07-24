import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Sequential
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv
from transformers import BertModel

import numpy as np

import os.path

class Model(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, encoder=None) -> None:
        super().__init__()

        if encoder is None:
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.encoder.parameters():
            param.requires_grad = False
        self._default_encoding_path = 'data/input_encoding.pt'

        self.num_layers = num_layers
        self.conv_layers = ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.out = Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim))

    def encode_inputs(self, tokens:torch.Tensor, batch_size=1024, encoding_dim=768, verbose=False):
        token_loader = DataLoader(tokens, batch_size)
        num_batches = len(token_loader)
        num_sentences = tokens.shape[0]
        encodings = torch.empty((num_sentences, encoding_dim), dtype=torch.float)
        for batch_idx, token_batch in enumerate(token_loader):
            if verbose:
                print(f'Batch {batch_idx}/{num_batches}')
            batch_encoding = self.encoder(token_batch[:,0,:],
                                          token_batch[:,1,:],
                                          token_batch[:,2,:]).last_hidden_state[:,0,:]
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx+1)*batch_size, num_sentences)
            encodings[batch_start:batch_end,:] = batch_encoding
        return encodings
    
    def save_input_encodings(self, encodings, path=None):
        if path is None:
            path = self._default_encoding_path
        torch.save(encodings, path)

    def load_input_encodings(self, path=None):
        if path is None:
            path = self._default_encoding_path
        assert os.path.exists(path)

        encodings = torch.load('data/input_encodings.pt')
        return encodings

    def forward(self, x:torch.Tensor, edge_index:torch.Tensor, encode_tokens=False) -> torch.Tensor:
        if encode_tokens:
            assert x.dim() == 3
            assert x.shape[1] == 3

            x = self.encode_inputs(x)
            self.save_input_encodings(x)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        return self.out(x)

class LinkPredictor(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.linear_layers = ModuleList()
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.linear_layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for linear in self.linear_layers[:-1]:
            x = linear(x)
            x = F.relu(x)
        x = self.linear_layers[-1](x)
        return torch.sigmoid(x)