import torch
import torch.nn as nn
import torch.functional as functional
from typing import List, Any
from dataclasses import dataclass
from torch_geometric.nn import GCNConv, GraphConv, GATConv, HypergraphConv, DenseGCNConv


@dataclass
class GeoGNNState:
    state_dict: Any
    sizes: List[int]


class GeoGNN(nn.Module):

    @staticmethod
    def from_model_state(state: GeoGNNState) -> "GeoGNN":
        model = GeoGNN(state.sizes)
        model.load_state_dict(state.state_dict)
        return model

    def __init__(self, sizes: List[int]):
        """
        :param sizes: the sizes of the input and output of sizes.
        """

        nn.Module.__init__(self)

        self._sizes = sizes
        self._activation = nn.ReLU()

        self._concats = nn.ModuleList(
            [GraphConv(in_, out, aggr="max") for (in_, out) in zip(sizes[:-1], sizes[1:])]
        )

        self.linear1 = nn.Linear(sizes[-1], sizes[-1])

    def get_model_state(self):
        return GeoGNNState(
            self.state_dict(),
            self._sizes,
        )

    def forward(self, x: torch.Tensor, edge_index):
        """
        :param x: tensor of shape [batch_size, n_intersections, n_features]
        :return:
        """
        for cc_layer in self._concats:
            x = cc_layer(x, edge_index)
            x = self._activation(x)

        self.linear1(x)
        self._activation(x)

        return x
