import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from .mlp import MLP
class MYACRConv(MessagePassing):
    def __init__(self, in_channels, out_channels, mlp_layers=0, mlp_dp=0):
        super().__init__(aggr='add', flow="source_to_target")
        self.ic = in_channels
        self.oc = out_channels
        self.ACR_bias = True
        self.mlp_layers = mlp_layers
        self.A = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)
        self.R = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)
        self.C = torch.nn.Linear(in_channels, out_channels, bias=self.ACR_bias)
        #self.b = torch.nn.Parameter(torch.zeros(self.oc))
        #nn.init.xavier_uniform_(self.b)

        if self.mlp_layers > 0:
            self.mlp = MLP(
                num_layers=mlp_layers,
                input_dim=out_channels,
                hidden_dim=out_channels,
                output_dim=out_channels,
                mlp_dp=mlp_dp,
                batch_norm=False,
                activation='sigmoid')

    def forward(self, x, edge_index, batch):
        # first readout
        r = global_add_pool(x, batch)

        # reshapes to (num nodes x features) matrix (with identical rows):
        r = r[batch]
        out = self.propagate(edge_index, h=x, readout=r)
        return out

    def message(self, h_j):
        return h_j

    def update(self, aggr, h, readout):
        updated = self.A(aggr) + self.C(h) + self.R(readout)# + self.b
        if self.mlp_layers > 0:
            updated = self.mlp(updated)
        return torch.sigmoid(updated)