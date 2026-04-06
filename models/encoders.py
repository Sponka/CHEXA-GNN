import torch
import torch.nn as nn


class EdgeMLP(nn.Module):
    def __init__(self, edge_dim, node_dim, edge_hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_hidden_dim, node_dim),
            nn.LayerNorm(node_dim)
        )

    def forward(self, edge_attr):
        return self.edge_mlp(edge_attr)


class GlobalEncoder(nn.Module):
    def __init__(self, global_dim, global_hidden_dim, global_output_dim):
        super().__init__()
        self.global_mlp = nn.Sequential(
            nn.Linear(global_dim, global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(global_hidden_dim, global_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(global_hidden_dim, global_output_dim)
        )

    def forward(self, u):
        if u.dim() == 1:
            u = u.unsqueeze(0)
        return self.global_mlp(u)


class NodeUpdateMLP(nn.Module):
    def __init__(self, node_dim, global_encoding_dim, node_hidden_dim, dropout_p=0.0):
        super().__init__()

        input_dim = node_dim + node_dim + global_encoding_dim

        layers = []

        layers.append(nn.Linear(input_dim, node_hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(node_hidden_dim, node_hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(node_hidden_dim, node_hidden_dim))
        layers.append(nn.LeakyReLU(0.2))
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        layers.append(nn.Linear(node_hidden_dim, node_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, h_i, m_i, u_i):
        h_cat = torch.cat([h_i, m_i, u_i], dim=-1)
        update_val = self.model(h_cat)
        return h_i + update_val
