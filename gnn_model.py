import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.functional as functional
from typing import List, Any
from dataclasses import dataclass

from torch_geometric.data import Data


@dataclass
class IntersectionGNNState:
    state_dict: Any
    sizes: List[int]
    adj_list: List[List[int]]

class IntersectionGNN(nn.Module):

    @staticmethod
    def _mean_aggregate(*hs) -> torch.Tensor:
        if len(hs) == 0:
            raise ValueError()

        result = hs[0]
        for h in hs[1:]:
            result = result + h

        return result / len(hs)

    @staticmethod
    def _max_aggregate(*hs):
        if len(hs) == 0:
            raise ValueError()
        # h: [batch, feats]
        hs = torch.stack(hs, dim=1)
        # hs: [batch, agents, feats]
        _, result = torch.max(hs, dim=1)
        return result

    @staticmethod
    def from_model_state(state: IntersectionGNNState) -> "IntersectionGNN":
        model = IntersectionGNN(state.sizes, state.adj_list)
        model.load_state_dict(state.state_dict)
        return model

    def __init__(self, sizes: List[int], adj_list: List[List[int]], aggr="mean"):
        """

        :param n_features:
        :param adj_list:
        :param depth: ignored if sizes is set. n_features will be used as first size
        :param sizes:
        """

        nn.Module.__init__(self)

        self._sizes = sizes
        self._adj_list = adj_list

        self._agg = IntersectionGNN._mean_aggregate
        # self._activation = nn.Softplus()
        self._activation = nn.GELU()

        self._layers = nn.ModuleList(
            [gnn.GraphConv(in_, out, aggr=aggr) for (in_, out) in zip(sizes[:-1], sizes[1:])]
        )

        self._concats = nn.ModuleList(
            [nn.Linear(2 * in_, out) for (in_, out) in zip(sizes[:-1], sizes[1:])]
        )

        self._lin_layers = nn.ModuleList(
            [nn.Linear(in_, out) for (in_, out) in zip(sizes[:-1], sizes[1:])]
        )

    def get_model_state(self):
        return IntersectionGNNState(
            self.state_dict(),
            self._sizes,
            self._adj_list
        )


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """

        :param edge_index:
        :param x: tensor of shape [batch_size, n_intersections, n_features]
        :return:
        """

        # for layer in self._lin_layers:
        #     x = layer(x)
        #     x = self._activation(x)
        #
        # return x

        for i, layer in enumerate(self._layers):
            x = layer(x, edge_index)

            x = self._activation(x)

        return x

        # for cc_layer in self._concats:
        #     new_x = []
        #
        #     for i_node, neighbors in enumerate(self._adj_list):
        #         aggregated = self._agg(
        #             *(x[..., i_nb, :] for i_nb in neighbors)
        #         )
        #
        #         h_i_node = x[..., i_node, :]
        #         # h_i_node ~= aggregated: [batch_size, n_features]
        #
        #         concatted = torch.cat((aggregated, h_i_node), dim=-1)
        #
        #         h_i_node_new = self._activation(cc_layer(concatted))
        #
        #         new_x.append(h_i_node_new)
        #
        #     x = torch.stack(new_x, dim=-2)
        #
        # return x
