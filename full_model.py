from typing import List

import torch
from torch import nn
from torch_geometric.data import Batch

from geometric_gnn_model import GeoGNN
from gnn_model import IntersectionGNN
from vae_net import VAENet, VariationalEncoderLayer, VariationalLayer, LOGNORMAL_DISTR
from dataclasses import dataclass
from typing import Any, Optional
from torch.distributions.normal import Normal


@dataclass
class GNNVAEModelState:
    state_dict: Any
    n_features: int
    adj_list: List[List[int]]
    n_hidden: int


class GNNVAEModel(nn.Module):

    @staticmethod
    def from_model_state(state: GNNVAEModelState) -> "GNNVAEModel":
        model = GNNVAEModel(state.n_features, state.adj_list, n_hidden=state.n_hidden)
        model.load_state_dict(state.state_dict)
        return model

    def __init__(self, n_features: int, adj_list: List[List[int]], n_hidden: Optional[int] = None):
        """

        :param n_features:
        :param adj_list:
        :param n_hidden: will be automatically chosen if unspecified
        """

        nn.Module.__init__(self)

        n_intersetions = len(adj_list)

        sizes = [n_features, int(n_features * (5 / 6)), int(n_features * (2 / 3)), int(n_features * (1 / 2))]

        if n_hidden is None:
            n_hidden = sizes[-1]

        self._n_hidden = n_hidden
        self._n_features = n_features
        self._adj_list = adj_list

        self._gnn_encoder = GeoGNN(sizes)
        self._variational_encoder = VariationalEncoderLayer(sizes[-1], n_hidden)
        self._gnn_decoder = GeoGNN(list(reversed(sizes)))
        self._variational_decoder = VariationalLayer(n_features, n_features, distr_cfg=LOGNORMAL_DISTR)

    def get_model_state(self) -> GNNVAEModelState:
        return GNNVAEModelState(
            self.state_dict(),
            self._n_features,
            self._adj_list,
            self._n_hidden
        )

    def sample(self, edge_index):

        x = self._variational_encoder.random_output([len(self._adj_list), self._n_features])

        x = self._gnn_decoder(x, edge_index)
        x = self._variational_decoder(x)

        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, calc_kl_div=False):
        """

        :param x:
        :param calc_kl_div: Calculate kl div loss and return it if desired
        :return:
        """
        #assert x.dim() == 3

        x = self._gnn_encoder(x, edge_index)

        if calc_kl_div:
            x, kl_loss = self._variational_encoder(x, calc_kl_div=True)
        else:
            x = self._variational_encoder(x)
            kl_loss = None

        x = self._gnn_decoder(x, edge_index)
        x = self._variational_decoder(x)

        if calc_kl_div:
            return x, kl_loss

        return x
