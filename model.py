import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Sequential
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm
from transformers import BertModel

import numpy as np

import os.path

class NodeEmbedder(Module):
    def __init__(self,
                 input_dim,
                 hidden_dim, 
                 output_dim, 
                 num_layers, 
                 num_heads, 
                 dropout=None,
                 use_precomputed_encodings=True,
                 encoder=None
    ) -> None:
        super().__init__()

        if num_layers == 0:
            self._fake = True
            return 
        else:
            self._fake = False

        if not use_precomputed_encodings:
            if encoder is None:
                self.encoder = BertModel.from_pretrained('bert-base-uncased')
            for param in self.encoder.parameters():
                param.requires_grad = False
        self._default_encoding_path = 'data/input_encoding.pt'

        self.dropout = dropout
        self.num_layers = num_layers
        self.norm_layers = ModuleList()
        self.norm_layers.append(BatchNorm(input_dim))
        self.conv_layers = ModuleList()
        self.conv_layers.append(GATv2Conv(input_dim, 
                                          hidden_dim, 
                                          num_heads,
                                          edge_dim=3,
                                          add_self_loops=False))
        self.lin_layers = ModuleList()
        self.lin_layers.append(nn.Linear(hidden_dim*num_heads, 
                                         hidden_dim*num_heads))
        for _ in range(num_layers-1):
            self.norm_layers.append(BatchNorm(hidden_dim*num_heads))
            self.conv_layers.append(GATv2Conv(hidden_dim*num_heads,
                                              hidden_dim,
                                              num_heads,
                                              edge_dim=3,
                                              add_self_loops=False))
            self.lin_layers.append(nn.Linear(hidden_dim*num_heads, 
                                             hidden_dim*num_heads))


        self.out = Sequential(
            nn.Linear(hidden_dim*num_heads, hidden_dim),
            nn.Linear(hidden_dim, output_dim))

    def encode_inputs(self, tokens:torch.Tensor, batch_size=1024, encoding_dim=768, verbose=False):
        token_loader = DataLoader(tokens, batch_size)
        num_batches = len(token_loader)
        encodings = None
        for batch_idx, token_batch in enumerate(token_loader):
            if verbose and (batch_idx % 10*batch_size == 0):
                print(f'Batch {batch_idx}/{num_batches}')
            batch_encoding = self.encoder(token_batch[:,0,:],
                                          token_batch[:,1,:],
                                          token_batch[:,2,:]).last_hidden_state[:,0,:]
            if encodings is None: 
                encodings = batch_encoding
            else:
                encodings = torch.concat([encodings, batch_encoding])
        return encodings
    
    def save_input_encodings(self, encodings, path=None):
        if path is None:
            path = self._default_encoding_path
        torch.save(encodings, path)

    def load_input_encodings(self, path=None):
        if path is None:
            path = self._default_encoding_path
        assert os.path.exists(path)

        encodings = torch.load(path)
        return encodings

    def forward(self, 
                x:torch.Tensor, 
                edge_index:torch.Tensor, 
                edge_attr:torch.Tensor,
                encode_tokens=False
    ) -> torch.Tensor:
        if encode_tokens:
            assert x.dim() == 3
            assert x.shape[1] == 3

            x = self.encode_inputs(x)
            self.save_input_encodings(x)
        if self._fake:
            return x
        
        for l in range(self.num_layers):
            x = self.norm_layers[l](x)
            x = self.conv_layers[l](x, edge_index, edge_attr)
            x = self.lin_layers[l](x)
            x = F.relu(x)
            if not self.dropout is None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.out(x)

class LinkPredictor(Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers, 
                 dropout=None
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        try:
            _ = (hd for hd in hidden_dim)
        except:
            hidden_dim = torch.tensor([hidden_dim]).repeat(num_layers-1)
        self.normalization_layers = ModuleList()
        self.normalization_layers.append(BatchNorm(input_dim))
        self.linear_layers = ModuleList()
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim[0]))
        for l in range(num_layers - 2):
            self.normalization_layers.append(BatchNorm(hidden_dim[l]))
            self.linear_layers.append(nn.Linear(hidden_dim[l], hidden_dim[l+1]))
        self.linear_layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.normalization_layers.append(BatchNorm(hidden_dim[-1]))

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for l in range(self.num_layers-1):
            x = self.normalization_layers[l](x)
            x = self.linear_layers[l](x)
            x = F.relu(x)
            if not self.dropout is None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.normalization_layers[-1](x)
        x = self.linear_layers[-1](x)
        # return torch.sigmoid(x)
        return x

class Model(Module):
    def __init__(self, node_embedder, link_predictor) -> None:
        super().__init__()
        self.node_embedder = node_embedder
        self.link_predictor = link_predictor
    def forward(self, x, edge_index, edge_attr, target_links):
        emb = self.node_embedder(x, edge_index, edge_attr)
        preds = self.link_predictor(emb[target_links[0]],
                                    emb[target_links[1]])
        return preds
