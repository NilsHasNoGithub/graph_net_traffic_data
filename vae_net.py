from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as funct
from torch.tensor import Tensor
from torch.distributions import Distribution, Categorical, OneHotCategoricalStraightThrough
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
import numpy as np
from typing import Tuple

def rsample_normal_distr(loc, scale) -> Tensor:
    distr = Normal(loc, scale)
    return distr.rsample()

def rsample_log_normal_distr(loc, scale) -> Tensor:
    distr = LogNormal(loc, scale)
    return distr.rsample()



@dataclass
class DistributionCfg:
    n_params: int
    constructor: Callable[[Tensor, ...], Distribution]
    param_transfms: Callable[[Tensor, ...], List[Tensor]]


    def try_rsample(self, *params) -> Tensor:
        distr = self.constructor(*params)
        if not distr.has_rsample:
            return distr.sample()
        return distr.rsample()

    def extract_params(self, x: Tensor) -> List[Tensor]:
        """
                    # Parameters:
                    - `x` with input shape `[a, b*2]`
                    """
        n_dims = len(x.size())
        *dims, distr_params = x.size()
        assert distr_params % self.n_params == 0, "Last dimension must be a multiple of the amount of parameters"

        # view mean and variance seperately [batch_size, 2    , n_hidden]
        x = x.view(*dims, self.n_params, distr_params // self.n_params)

        params = [x[..., i, :] for i in range(self.n_params)]
        params = self.param_transfms(*params)

        return params

    def extract_params_and_sample(self, x: Tensor, return_params=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
            # Parameters:
            - `x` with input shape `[a, b*2]`
            """
        params = self.extract_params(x)

        samples = self.try_rsample(*params)

        if return_params:
            return samples, params
        return samples



def _normal_param_transform(loc, scale):
    scale = funct.softplus(scale)
    scale = torch.add(scale, 0.00000001)

    return [loc, scale]

def _categorical_param_transform(*params):
    # params: [batch_size, n_neighbors, n_features] * n_categories -> [batch_size, n_neighbors, n_features, n_categories]
    params = torch.stack(params, dim=-1)
    params = funct.softmax(params, dim=-1)
    return [params]


NORMAL_DISTR = DistributionCfg(2, Normal, _normal_param_transform)
LOGNORMAL_DISTR = DistributionCfg(2, LogNormal, _normal_param_transform)

def categorical_distr(n_categories: int) -> DistributionCfg:
    return DistributionCfg(n_categories, Categorical, _categorical_param_transform)

def one_hot_categorical_distr(n_categories: int) -> DistributionCfg:
    return DistributionCfg(n_categories, OneHotCategoricalStraightThrough, _categorical_param_transform)

@dataclass
class VAENetData:
    n_features: int
    n_hidden: int
    distr_cfg: DistributionCfg
    state_dict: Any

@dataclass
class VAEEncoderForwardResult:
    x: Tensor
    kl_div: Tensor
    params: List[Tensor]

@dataclass
class VAEDecoderForwardResult:
    x: Tensor
    params: List[Tensor]

class VAENet(nn.Module):

    def __init__(
        self,
         n_features,
         n_hidden=300,
         distr_cfg: Optional[DistributionCfg] = None
    ) -> None:
        nn.Module.__init__(self)


        if distr_cfg is None:
            distr_cfg = NORMAL_DISTR

        self.distr_cfg = distr_cfg

        # mean and variance param
        self.n_hidden = n_hidden
        self.n_features = n_features

        self.to_hidden = nn.Linear(n_features, 2*n_hidden)
        self.from_hidden = nn.Linear(n_hidden, 2*n_features)

        # self.encoder = VariationalLayer(n_features, n_hidden, distr_cfg=NORMAL_DISTR)
        # self.decoder = VariationalLayer(n_hidden, n_features, distr_cfg=distr_cfg)

    @staticmethod
    def from_model_data(data: VAENetData) -> 'VAENet':
        result = VAENet(data.n_features, n_hidden=data.n_hidden, distr_cfg=data.distr_cfg)
        result.load_state_dict(data.state_dict)
        return result

    def get_model_data(self) -> VAENetData:
        return VAENetData(
            self.n_features,
            self.n_hidden,
            self.distr_cfg,
            self.state_dict()
        )

    def forward(self, x: Tensor) -> VAEEncoderForwardResult:
        """

        :param x:
        :param calc_kl_div: if true, the kl diversion will be computed and returned
        :return:
        """
        
        batch_size, *original_size = x.size()

        x = self.to_hidden(x)

        x, (enc_loc, enc_scale) = NORMAL_DISTR.extract_params_and_sample(x, return_params=True)

        x = self.from_hidden(x)

        x, params = self.distr_cfg.extract_params_and_sample(x, return_params=True)

        x = x.view(batch_size, *original_size)

        kl_div = torch.log(1.0 / enc_scale) + (enc_scale ** 2.0 + enc_loc ** 2.0) / (2.0 * 1.0) - 0.5
        kl_div = torch.mean(kl_div)

        return VAEEncoderForwardResult(x, kl_div, params)


    def random_output(self, output_shape=None):
        # assert self.n_features == np.product(output_shape)

        loc = torch.zeros(*output_shape)
        scale = torch.ones(*output_shape)

        x = rsample_normal_distr(loc, scale)
        # x = x.view(1, -1)

        x = self.distr_cfg.extract_params_and_sample(x)

        if output_shape is None:
            return x

        return x.view(1, *output_shape)



class VariationalLayer(nn.Module):

    def __init__(self, n_in: int, n_out: int, distr_cfg: Optional[DistributionCfg] = None):
        nn.Module.__init__(self)

        if distr_cfg is None:
            distr_cfg = NORMAL_DISTR

        self.distr_cfg = distr_cfg

        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in,  distr_cfg.n_params*n_out)

    def forward(self, x: Tensor):

        x = self.linear(x)

        x, params = self.distr_cfg.extract_params_and_sample(x, return_params=True)

        return VAEDecoderForwardResult(x, params)


class VariationalEncoderLayer(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        nn.Module.__init__(self)

        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, NORMAL_DISTR.n_params * n_out)

    def forward(self, x: Tensor):

        x = self.linear(x)

        x, (loc, scale) = NORMAL_DISTR.extract_params_and_sample(x, return_params=True)

        kl_loss = torch.log(1.0 / scale) + (scale ** 2.0 + loc ** 2.0) / (2.0 * 1.0) - 0.5
        kl_loss = torch.mean(kl_loss)

        return VAEEncoderForwardResult(x, kl_loss, [loc, scale])

    def random_output(self, output_shape, batch_size=1):

        loc = torch.zeros(batch_size, *output_shape)
        scale = torch.ones(batch_size, *output_shape)

        x = rsample_normal_distr(loc, scale)

        x = NORMAL_DISTR.extract_params_and_sample(x)

        return x