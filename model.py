import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Sequential
from torch_geometric.nn import GCNConv
from transformers import BertModel

class Model(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, encoder=None) -> None:
        super().__init__()

        if encoder is None:
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.num_layers = num_layers
        self.conv_layers = ModuleList()
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers-1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))

        self.out = Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, tokens:dict, edge_index:torch.Tensor) -> torch.Tensor:
        x = self.encoder(tokens['token_ids'], tokens['token_mask'], tokens['token_type_ids'])
        x = x.last_hidden_state.flatten(start_dim=1)

        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu()
        
        x = self.out(x)

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
