from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Union
import torch
from torch import nn
import torch.nn.functional as funct
from torch.tensor import Tensor
from torch.distributions import Distribution, Categorical
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
    def sample_from_params(self, *params, n=1) -> Tuple[Tensor, List[Tensor]]:
        raise NotImplementedError()

    @abstractmethod
    def log_prob(self, params: List[Tensor], value: Tensor):
        raise NotImplementedError()

    @abstractmethod
    def torch_distr(self, *params) -> Distribution:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *processed_params):
        raise NotImplementedError()

    def rsample(self, x: Tensor, n=1) -> Tuple[Tensor, List[Tensor]]:
        """

        :param x:
        :return: The output, and the parameters which were used to sample the output
        """
        params = self.extract_params(x)
        x, params = self.sample_from_params(*params, n=n)

        return x, params




class VAENormalDistr(VAEDistr):

    def __init__(self):
        VAEDistr.__init__(self, 2)

    def _transform_params(self, *params) -> List[Tensor]:
        loc, scale = params

        scale = funct.softplus(scale)
        scale = torch.add(scale, 0.00000001)

        return [loc, scale]

    def log_prob(self, params: List[Tensor], value: Tensor) -> float:
        loc, scale = params
        distr = Normal(loc, scale)
        return distr.log_prob(value)

    def sample_from_params(self, *params, n=1) -> Tuple[Tensor, List[Tensor]]:
        loc, scale = self._transform_params(*params)
        distr = Normal(loc, scale)

        # eps = torch.rand_like(scale)
        samples = distr.rsample([n])
        samples = samples.transpose(0, 1)

        return samples, [loc, scale]

    def torch_distr(self, *params) -> Distribution:
        return Normal(*params)

    def predict(self, *processed_params):
        loc, _ = processed_params
        return loc

class VAELogNormalDistr(VAENormalDistr):

    def __init__(self):
        VAENormalDistr.__init__(self)

    def log_prob(self, params: List[Tensor], value: Tensor) -> float:
        loc, scale = params
        distr = LogNormal(loc, scale)
        return distr.log_prob(value + 0.000001)

    def sample_from_params(self, *params, n=1) -> Tuple[Tensor, List[Tensor]]:
        loc, scale = self._transform_params(*params)
        distr = LogNormal(loc, scale)

        samples = distr.rsample([n])
        samples = samples.transpose(0, 1)

        return samples, [loc, scale]

    def torch_distr(self, *params) -> Distribution:
        return LogNormal(*params)

    def predict(self, *processed_params):
        loc, scale = processed_params
        return torch.exp(loc - scale ** 2)


class VAECategoricalDistr(VAEDistr):

    def __init__(self, n_categories):
        VAEDistr.__init__(self, n_categories)

    def _transform_params(self, *params) -> Tensor:
        params = torch.stack(params, dim=-1)
        params = funct.softmax(params, dim=-1)

        return params

    def sample_from_params(self, *params, n=1) -> Tuple[Tensor, List[Tensor]]:
        params = self._transform_params(*params)

        distr = Categorical(probs=params)

        samples = distr.sample([n])
        samples = samples.transpose(0, 1)

        return samples, [params]

    def log_prob(self, params: List[Tensor], value: Tensor) -> float:
        params = params[0]
        distr = Categorical(probs=params)

        return distr.log_prob(value)

    def torch_distr(self, *params) -> Distribution:
        params = self._transform_params(*params)
        return Categorical(probs=params)

    def predict(self, *processed_params):
        probs = processed_params[0]
        return torch.argmax(probs, dim=-1)


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

        for i in range(len(params)):
            params[i] = torch.mean(params[i], dim=1)

        x = self.distr.predict(*params)

        return VAEDecoderForwardResult(x, params)


class VariationalEncoderLayer(nn.Module):

    def __init__(self, n_in: int, n_out: int):
        nn.Module.__init__(self)

        self.n_in = n_in
        self.n_out = n_out

        self._distr = VAENormalDistr()

        self.linear = nn.Linear(n_in, self._distr.n_params * n_out)

    def forward(self, x: Tensor, n=1):

        x = self.linear(x)

        x, (loc, scale) = self._distr.rsample(x, n=n)


        # kl_loss = torch.log(1.0 / scale) + (scale ** 2.0 + loc ** 2.0) / 2.0 - 0.5
        kl_loss = torch.distributions.kl.kl_divergence(Normal(loc, scale), Normal(0, 1))
        kl_loss = torch.sum(kl_loss, -1)
        # kl_loss = torch.sum(kl_loss, -1)
        kl_loss = torch.mean(kl_loss)



        return VAEEncoderForwardResult(x, kl_loss, [loc, scale])

    def random_output(self, output_shape, batch_size=1):

        loc = torch.zeros(batch_size, *output_shape)
        scale = torch.ones(batch_size, *output_shape)

        x, _ = self._distr.sample_from_params(loc, scale)

        # x, _ = self._distr.rsample(x)

        return x