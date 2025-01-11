import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, ModuleList, Sequential, Conv1d
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, BatchNorm
from transformers import BertModel

import numpy as np

import os.path


class NodeEmbedder(Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        edge_dim,
        num_layers,
        num_heads,
        dropout=None,
    ) -> None:
        super().__init__()

        self._fake = num_layers == 0
        if self._fake:
            return

        self.dropout = dropout
        self.num_layers = num_layers

        self.norm_layers = ModuleList()
        self.norm_layers.append(BatchNorm(input_dim))

        self.conv_layers = ModuleList()
        self.conv_layers.append(
            GATv2Conv(
                input_dim,
                hidden_dim,
                num_heads,
                edge_dim=edge_dim,
                add_self_loops=False,
            )
        )

        self.lin_layers = ModuleList()
        self.lin_layers.append(
            nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads)
        )

        for _ in range(num_layers - 1):
            self.norm_layers.append(BatchNorm(hidden_dim * num_heads))
            self.conv_layers.append(
                GATv2Conv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.lin_layers.append(
                nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads)
            )

        self.out = Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
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
    def __init__(
        self, input_dim, hidden_dim, output_dim, edge_dim, num_layers, dropout=None
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        try:
            _ = (hd for hd in hidden_dim)
            assert len(hidden_dim) == num_layers
        except AssertionError:
            raise ValueError("hidden_dim must be an iterable of length num_layers")
        except TypeError:
            hidden_dim = torch.tensor([hidden_dim]).repeat(num_layers)

        self.r_projection = nn.Linear(edge_dim, input_dim)

        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=2)

        self.normalization_layers = ModuleList()
        self.normalization_layers.append(BatchNorm(input_dim * 2))

        self.linear_layers = ModuleList()
        self.linear_layers.append(nn.Linear(input_dim * 2, hidden_dim[1]))

        for l in range(1, num_layers - 1):
            self.normalization_layers.append(BatchNorm(hidden_dim[l - 1]))
            self.linear_layers.append(nn.Linear(hidden_dim[l - 1], hidden_dim[l]))

        self.normalization_layers.append(BatchNorm(hidden_dim[-1]))
        self.linear_layers.append(nn.Linear(hidden_dim[-1], output_dim))

    def forward(self, xs, xr, xo):
        xr = self.r_projection(xr)
        xr = F.relu(xr)

        x = self.conv(torch.stack([xs, xo], dim=-1)).squeeze(-1)
        x = F.relu(x)

        x = torch.cat([x, xo], dim=1)

        for l in range(self.num_layers - 1):
            x = self.normalization_layers[l](x)
            x = self.linear_layers[l](x)
            x = F.relu(x)
            if not self.dropout is None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.normalization_layers[-1](x)
        x = self.linear_layers[-1](x)

        return x


class Model(Module):
    def __init__(self, node_embedder, link_predictor) -> None:
        super().__init__()
        self.node_embedder = node_embedder
        self.link_predictor = link_predictor

    def forward(self, x, edge_index, edge_attr, target_edges, target_edge_attrs):
        emb = self.node_embedder(x, edge_index, edge_attr)

        xs = emb[target_edges[0]]
        xo = emb[target_edges[1]]
        xr = target_edge_attrs
        preds = self.link_predictor(xs, xr, xo)

        return preds
