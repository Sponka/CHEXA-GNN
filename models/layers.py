import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, LayerNorm

from .encoders import EdgeMLP, NodeUpdateMLP


class CustomGNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, global_encoding_dim,
                 edge_hidden_dim, node_hidden_dim,
                 aggr='add', dropout_p=0.0):
        super().__init__(aggr=aggr)

        self.edge_encoder = EdgeMLP(edge_dim, node_dim, edge_hidden_dim)
        self.node_update = NodeUpdateMLP(node_dim, global_encoding_dim,
                                         node_hidden_dim, dropout_p)
        self.norm = LayerNorm(node_dim, mode="node")

    def forward(self, x, edge_index, edge_attr, u_encoded, batch):
        self._cached_u = u_encoded[batch]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        W_e_normed = self.edge_encoder(edge_attr)
        msg = W_e_normed * x_j
        return msg

    def update(self, aggr_out, x):
        aggr_out = self.norm(aggr_out)
        aggr_out = F.leaky_relu(aggr_out, negative_slope=0.2)
        return self.node_update(x, aggr_out, self._cached_u)
