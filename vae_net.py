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
from abc import ABC, abstractmethod


class VAEDistr(ABC):

    def __init__(self, n_params: int):
        self.n_params = n_params


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
        return params

    @abstractmethod
    def rsample_from_params(self, *params) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError()

    def rsample(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """

        :param x:
        :return: The output, and the parameters which were used to sample the output
        """
        params = self.extract_params(x)
        x, params = self.rsample_from_params(*params)
        return x, params


class VAENormalDistr(VAEDistr):

    def __init__(self):
        VAEDistr.__init__(self, 2)

    def rsample_from_params(self, *params) -> Tuple[Tensor, List[Tensor]]:
        loc, scale = params

        scale = funct.softplus(scale)
        scale = torch.add(scale, 0.00000001)

        distr = Normal(loc, scale)
        return distr.rsample(), [loc, scale]

class VAELogNormalDistr(VAEDistr):

    def __init__(self):
        VAEDistr.__init__(self, 2)

    def rsample_from_params(self, *params) -> Tuple[Tensor, List[Tensor]]:
        loc, scale = params

        scale = funct.softplus(scale)
        scale = torch.add(scale, 0.00000001)

        distr = LogNormal(loc, scale)
        return distr.rsample(), [loc, scale]

class VAECategoricalDistr(VAEDistr):

    def __init__(self, n_categories):
        VAEDistr.__init__(self, n_categories)

    def rsample_from_params(self, *params) -> Tuple[Tensor, List[Tensor]]:
        params = torch.stack(params, dim=-1)
        params = funct.softmax(params, dim=-1)

        distr = OneHotCategoricalStraightThrough(probs=params)
        result = distr.rsample()

        categories = torch.arange(0, self.n_params, step=1, dtype=torch.float32).to(params.device)
        result = result * categories
        result = torch.sum(result, dim=-1)

        return result, [params]


@dataclass
class VAENetData:
    n_features: int
    n_hidden: int
    distr: VAEDistr
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


class VariationalLayer(nn.Module):

    def __init__(self, n_in: int, n_out: int, distr: Optional[VAEDistr] = None):
        nn.Module.__init__(self)

        if distr is None:
            distr = VAENormalDistr()

        self.distr = distr

        self.n_in = n_in
        self.n_out = n_out

        self.linear = nn.Linear(n_in, distr.n_params * n_out)

    def forward(self, x: Tensor):

        x = self.linear(x)

        x, params = self.distr.rsample(x)

        return VAEDecoderForwardResult(x, params)


class VariationalEncoderLayer(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        nn.Module.__init__(self)

        self.n_in = n_in
        self.n_out = n_out

        self._distr = VAENormalDistr()

        self.linear = nn.Linear(n_in, self._distr.n_params * n_out)

    def forward(self, x: Tensor):

        x = self.linear(x)

        x, (loc, scale) = self._distr.rsample(x)

        kl_loss = torch.log(1.0 / scale) + (scale ** 2.0 + loc ** 2.0) / (2.0 * 1.0) - 0.5
        kl_loss = torch.mean(kl_loss)

        return VAEEncoderForwardResult(x, kl_loss, [loc, scale])

    def random_output(self, output_shape, batch_size=1):

        loc = torch.zeros(batch_size, *output_shape)
        scale = torch.ones(batch_size, *output_shape)

        x, _ = self._distr.rsample_from_params(loc, scale)

        x, _ = self._distr.rsample(x)

        return x