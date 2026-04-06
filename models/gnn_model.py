import torch.nn as nn

from .encoders import GlobalEncoder
from .layers import CustomGNNLayer


class GNNModel(nn.Module):
    def __init__(self, node_dim, edge_dim, global_dim,
                 global_hidden_dim,
                 global_encoding_dim,
                 edge_hidden_dim,
                 node_hidden_dim,
                 decoder_hidden_dim,
                 num_layers=3, dropout_p=0.0):
        super().__init__()

        self.layers = nn.ModuleList()
        self.global_encoders = nn.ModuleList()

        for _ in range(num_layers):
            self.global_encoders.append(
                GlobalEncoder(global_dim,
                              global_hidden_dim,
                              global_encoding_dim)
            )

            self.layers.append(
                CustomGNNLayer(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    global_encoding_dim=global_encoding_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    node_hidden_dim=node_hidden_dim,
                    aggr='add',
                    dropout_p=dropout_p)
            )

        self.decoder = nn.Sequential(
            nn.Linear(node_dim, decoder_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(decoder_hidden_dim, node_dim)
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        for layer, global_encoder in zip(self.layers, self.global_encoders):
            u_encoded = global_encoder(u)
            x = layer(x, edge_index, edge_attr, u_encoded, batch)
        return self.decoder(x)
