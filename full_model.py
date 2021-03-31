from typing import List

import torch
from torch import nn
from gnn_model import IntersectionGNN
from vae_net import VAENet
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

    def __init__(self, n_features: int, adj_list: List[List[int]], n_hidden: Optional[int]=None):
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



        self._gnn_encoder = IntersectionGNN(sizes, adj_list)
        self._vae_net = VAENet(sizes[-1], n_hidden=n_hidden)
        self._gnn_decoder = IntersectionGNN(list(reversed(sizes)), adj_list)
        self._to_out = nn.Linear(n_features, n_features)



    def get_model_state(self) -> GNNVAEModelState:
        return GNNVAEModelState(
            self.state_dict(),
            self._n_features,
            self._adj_list,
            self._n_hidden
        )

    def sample(self):

       x = self._vae_net.random_output([len(self._adj_list), int(self._n_features * (1/2))])

       x = self._gnn_decoder(x)
       x = self._to_out(x)

       return x

    def forward(self, x: torch.Tensor, calc_kl_div=False):
        """

        :param x:
        :param calc_kl_div: Calculate kl div loss and return it if desired
        :return:
        """
        assert x.dim() == 3

        x = self._gnn_encoder(x)

        if calc_kl_div:
            x, kl_loss = self._vae_net(x, calc_kl_div=True)
        else:
            x = self._vae_net(x)
            kl_loss = None

        x = self._gnn_decoder(x)
        x = self._to_out(x)

        if calc_kl_div:
            return x, kl_loss

        return x
