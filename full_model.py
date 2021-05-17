from typing import List, Union

import torch
from torch import nn, Tensor
from gnn_model import IntersectionGNN, GNNEncoder, GNNDecoder
from vae_net import VariationalEncoderLayer, VariationalLayer, VAEEncoderForwardResult, VAECategoricalDistr, \
    VAELogNormalDistr, VAEDecoderForwardResult, VAEDistr
from dataclasses import dataclass
from typing import Any, Optional
from torch.distributions.normal import Normal
from torch_geometric.data import Batch, Data
import torch_geometric.nn as geomnn


@dataclass
class GNNVAEModelState:
    state_dict: Any
    n_features: int
    adj_list: List[List[int]]
    n_out: int
    n_hidden: int
    decoder_distr: VAEDistr


@dataclass
class GNNVAEForwardResult:
    x: Tensor
    kl_div: Tensor
    params_encoder: List[Tensor]
    params_decoder: List[Tensor]


class GNNVAEModel(nn.Module):

    @staticmethod
    def from_model_state(state: GNNVAEModelState) -> "GNNVAEModel":
        model = GNNVAEModel(state.n_features, state.adj_list, n_hidden=state.n_hidden, n_out=state.n_out,
                            decoder_distr=state.decoder_distr)
        model.load_state_dict(state.state_dict)
        return model

    def __init__(self, n_features: int, adj_list: List[List[int]], n_out: int = None, n_hidden: Optional[int] = None,
                 decoder_distr: Optional[VAEDistr] = None):
        """

        :param n_features:
        :param adj_list:
        :param n_out: in case output is not the same as input (should be roughly same size though)
        :param n_hidden: will be automatically chosen if unspecified
        :param decoder_distr: default is lognormal distribution
        """

        nn.Module.__init__(self)

        n_intersetions = len(adj_list)

        edges = []
        for i_from, tos in enumerate(adj_list):
            for i_to in tos:
                edges.append([i_from, i_to])

        self._edges = torch.tensor(edges).transpose(0, 1)

        if n_out is None:
            n_out = n_features

        # sizes = [n_features, int(n_features * (5 / 6)), int(n_features * (2 / 3)), int(n_features * (1 / 2))]
        sizes = [n_features, n_features, n_features]

        if n_hidden is None:
            n_hidden = sizes[-1]

        if decoder_distr is None:
            decoder_distr = VAECategoricalDistr(30)

        self._encoder = IntersectionGNN(sizes, adj_list)

        self._layers = nn.ModuleList(
            [nn.Linear(in_, out) for (in_, out) in zip(sizes[:-1], sizes[1:])]
        )

        self._n_hidden = n_hidden
        self._n_out = n_out
        self._n_features = n_features
        self._adj_list = adj_list

        self._gnn_encoder = GNNEncoder(sizes, adj_list)
        self._gnn_decoder = GNNDecoder(sizes, adj_list) # IntersectionGNN(list(reversed(sizes)), adj_list)
        self._VAE = geomnn.VGAE(self._gnn_encoder, self._gnn_decoder)

        self._activation = nn.ReLU()

        # Change to not sampling
        # self._variational_decoder = VariationalLayer(n_features, n_out, distr=decoder_distr)

        # self._variational_decoder = VariationalLayer(n_features, n_out, distr=VAELogNormalDistr())

    def get_model_state(self) -> GNNVAEModelState:
        return GNNVAEModelState(
            self.state_dict(),
            self._n_features,
            self._adj_list,
            self._n_out,
            self._n_hidden,
            self._variational_decoder.distr
        )

    def distr(self) -> VAEDistr:
        return self._variational_decoder.distr

    def sample(self):

        x = self._variational_encoder.random_output([len(self._adj_list), self._n_hidden])

        x = self._gnn_decoder(x, self._edges)

        x: VAEDecoderForwardResult = self._variational_decoder(x)

        return VAEEncoderForwardResult(x.x, torch.tensor(0.0), x.params)

    def forward(self, x: Tensor):
        """

        :param x:
        :param calc_kl_div: Calculate kl div loss and return it if desired
        :return:
        """
        assert x.dim() == 3

        # if isinstance(x, Tensor):
        #     batch_x = self._tensor_to_batch(x)
        # else:
        #     batch_x = x

        #for layer in self._layers:
        #    x = layer(x) #, edge_index)
        #    x = self._activation(x)

        #x = self._encoder(x, self._edges)

        #return GNNVAEForwardResult(x, 0, None, None)

        x = self._VAE.encode(x, edge_index=self._edges)
        kl_div = self._VAE.kl_loss()
        x = self._VAE.decode(x, edge_index=self._edges)

        return GNNVAEForwardResult(x, kl_div, None, None)
