import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from .conv_layers import MYACRConv
from .mlp import MLP

# base code taken from https://github.com/juanpablos/GNN-logic

class MYACRGnnNode(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, mlp_layers=0, final_read="add", num_classes=2, fwd_dp=0.1, lin_dp=0.5, mlp_dp=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.fwd_dp = fwd_dp
        self.layers = torch.nn.ModuleList()
        self.mlp_layers = mlp_layers

        if final_read == "add":
            self.final_readout = global_add_pool
        elif final_read == "mean":
            self.final_readout = global_mean_pool
        else:
            print("Final readout {0} not implemented!".format(final_read))

        if type(hidden_dim) == list:
            for l in range(num_layers):
                if l == 0:
                    self.layers.append(MYACRConv(input_dim, hidden_dim[l], mlp_layers=self.mlp_layers, mlp_dp=mlp_dp))
                else:
                    self.layers.append(MYACRConv(hidden_dim[l-1], hidden_dim[l], mlp_layers=self.mlp_layers, mlp_dp=mlp_dp))
            self.linear = torch.nn.Linear(hidden_dim[-1], num_classes)
        else:
            print("Hidden dim {0} not implemented! Must be a list".format(hidden_dim))

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer.forward(x, edge_index, batch)
            x = F.dropout(x, p=self.fwd_dp, training=self.training)
        x = self.linear(x)
        return x
