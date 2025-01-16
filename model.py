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

        try:
            assert len(hidden_dim) == num_layers
        except AssertionError:
            raise ValueError("hidden_dim must be an iterable of length num_layers")

        self.dropout = dropout
        self.num_layers = num_layers

        self.norm_layers = ModuleList()
        self.norm_layers.append(BatchNorm(input_dim))

        self.x_projection = nn.Linear(input_dim, hidden_dim[0])

        self.conv_layers = ModuleList()
        self.lin_layers = ModuleList()
        hidden_dim.append(output_dim)
        for l in range(num_layers):
            self.conv_layers.append(
                GATv2Conv(
                    hidden_dim[l],
                    hidden_dim[l],
                    num_heads[l],
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )
            self.lin_layers.append(
                nn.Linear(hidden_dim[l] * num_heads[l], hidden_dim[l + 1])
            )
        self.norm_layers.append(BatchNorm(hidden_dim[-1]))
        self.out = Sequential(
            nn.Linear(hidden_dim[-1], output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if self._fake:
            return x, edge_attr

        x = self.norm_layers[0](x)
        x = self.x_projection(x)
        for l in range(self.num_layers):
            x = self.conv_layers[l](x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.lin_layers[l](x)
            x = F.relu(x)
            if not self.dropout is None:
                x = F.dropout(x, p=self.dropout, training=self.training)
        self.norm_layers[-1](x)
        return self.out(x)


class RelationEmbedder(Module):
    def __init__(self, edge_dim, output_dim):
        super().__init__()

        self.r_project = nn.Linear(edge_dim, output_dim)
        self.conv = nn.Conv1d(output_dim, output_dim, 3)

    def forward(self, x, edge_index, edge_attr):
        xh = x[edge_index[0]]
        xt = x[edge_index[1]]
        xr = F.relu(self.r_project(edge_attr))

        xr = self.conv(torch.stack([xh, xt, xr], dim=-1)).squeeze(-1)
        xr = F.relu(xr)
        return xr


class TransE(Module):
    def __init__(self, p_norm=1.0, margin=1.0):
        super().__init__()
        self.p_norm = p_norm
        self.margin = margin

    def forward(self, xh, xr, xt):
        xh = F.normalize(xh, p=self.p_norm, dim=-1)
        xt = F.normalize(xt, p=self.p_norm, dim=-1)

        return -((xh + xr) - xt).norm(p=self.p_norm, dim=-1)

    def loss(self, golden_score, corrupted_score):
        return F.margin_ranking_loss(
            golden_score,
            corrupted_score,
            torch.ones_like(golden_score),
            margin=self.margin,
        )


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

        self.conv = nn.Conv1d(input_dim, input_dim, kernel_size=3)

        self.normalization_layers = ModuleList()
        self.normalization_layers.append(BatchNorm(input_dim))

        self.linear_layers = ModuleList()
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim[0]))

        for l in range(1, num_layers):
            self.linear_layers.append(nn.Linear(hidden_dim[l - 1], hidden_dim[l]))

        self.normalization_layers.append(BatchNorm(hidden_dim[-1]))
        self.linear_layers.append(nn.Linear(hidden_dim[num_layers - 1], output_dim))

    def forward(self, xs, xr, xo):
        xr = self.r_projection(xr)
        xr = F.relu(xr)

        x = self.conv(torch.stack([xs, xr, xo], dim=-1)).squeeze(-1)
        x = F.relu(x)

        x = self.normalization_layers[0](x)

        for l in range(self.num_layers):
            x = self.linear_layers[l](x)
            x = F.relu(x)
            if not self.dropout is None:
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.normalization_layers[-1](x)
        x = self.linear_layers[-1](x)
        x = F.sigmoid(x)

        return x


class Model(Module):
    def __init__(self, node_embedder, relation_embedder, link_predictor) -> None:
        super().__init__()
        self.node_embedder = node_embedder
        self.relation_embedder = relation_embedder
        self.link_predictor = link_predictor

    def forward(self, x, edge_index, edge_attr, target_edge_index, target_edge_attr):
        x = self.node_embedder(x, edge_index, edge_attr)
        xr = self.relation_embedder(x, target_edge_index, target_edge_attr)

        xs = x[target_edge_index[0]]
        xo = x[target_edge_index[1]]
        preds = self.link_predictor(xs, xr, xo)

        return preds

    def loss_function(self):
        return self.link_predictor.loss
