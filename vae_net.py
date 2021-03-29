from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as funct
from torch.tensor import Tensor
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
import numpy as np
from typing import Tuple

def rsample_normal_distr(loc, scale) -> Tensor:
    distr = Normal(loc, scale)
    return distr.rsample()

def rsample_log_normal_distr(loc, scale) -> Tensor:
    distr = LogNormal(loc, scale)
    return distr.sample()

@dataclass
class VAENetData:
    n_features: int
    n_hidden: int
    state_dict: Any

class VAENet(nn.Module):

    def __init__(self, n_features, n_hidden=300) -> None:
        nn.Module.__init__(self)

        # mean and variance param
        self.n_hidden = n_hidden
        self.n_features = n_features

        self.encoder = VAEEncoder(n_features, n_hidden)
        self.decoder = VAEDecoder(n_features, n_hidden)

    @staticmethod
    def from_model_data(data: VAENetData) -> 'VAENet':
        result = VAENet(data.n_features, n_hidden=data.n_hidden)
        result.load_state_dict(data.state_dict)
        return result

    def get_model_data(self) -> VAENetData:
        return VAENetData(
            self.n_features,
            self.n_hidden,
            self.state_dict()
        )

    def forward(self, x: Tensor, calc_kl_div=False):
        """

        :param x:
        :param calc_kl_div: if true, the kl diversion will be computed and returned
        :return:
        """
        
        # batch_size, *original_size = x.size()

        enc_loc, enc_scale = self.encoder(x)

        x = rsample_normal_distr(enc_loc, enc_scale)
        
        loc, scale = self.decoder(x)

        x = rsample_log_normal_distr(loc, scale)

        # x = x.view(batch_size, *original_size)

        if calc_kl_div:
            kl_loss = torch.log(1.0 / enc_scale) + (enc_scale ** 2.0 + enc_loc ** 2.0) / (2.0 * 1.0) - 0.5
            kl_loss = torch.mean(kl_loss)
            return x, kl_loss

        return x

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        # Returns:
        - means
        - stds
        """
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        ## Returns:

        - means
        - stds
        """
        return self.decoder(x)

    def random_output(self, output_shape=None):
        assert self.n_features == np.product(output_shape)

        loc = torch.zeros(self.n_hidden)
        scale = torch.ones(self.n_hidden)

        x = rsample_normal_distr(loc, scale)
        x = x.view(1, -1)

        loc, scale = self.decode(x)

        x = rsample_normal_distr(loc, scale)

        if output_shape is None:
            return x

        return x.view(1, *output_shape)



def extract_distr_params(x: Tensor):
    """
    # Parameters:
    - `x` with input shape `[a, b*2]`
    """
    n_dims = len(x.size())
    *dims, distr_params = x.size()
    assert distr_params % 2 == 0, "There must be a mean and variance parameter for each feature"

    # view mean and variance seperately [batch_size, 2    , n_hidden]
    x = x.view(*dims, 2, distr_params // 2)

    distr_param_idx = n_dims - 1

    loc = x.index_select(distr_param_idx, torch.tensor([0]).to(x.device)).squeeze()
    scale = x.index_select(distr_param_idx, torch.tensor([1]).to(x.device)).squeeze()

    scale = funct.softplus(scale)
    scale = torch.add(scale, 0.00000001)

    return loc, scale

VAE_N_HIDDEN = 200

class VAEEncoder(nn.Module):

    def __init__(self, n_features, n_hidden_states) -> None:
        nn.Module.__init__(self)

        self.n_states = n_hidden_states
        self.n_features = n_features
        self.to_encoded = nn.Linear(n_features, n_hidden_states*2)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        x = self.to_encoded(x)

        loc, scale = extract_distr_params(x)

        return loc, scale

    

class VAEDecoder(nn.Module):

    def __init__(self, n_output_features, n_hidden_states) -> None:
        """

        ## Parameters:
        - `output_shape`, shape in which the original features were. `forward` will return a tensor of [batch_size, *output_shape]
        """
        nn.Module.__init__(self)

        self.n_hidden = n_hidden_states
        self.n_features = n_output_features

        self.decoder = nn.Linear(n_hidden_states, 2*self.n_features)

    def forward(self, x: Tensor):
        # batch_size, *_ = x.size()
        #
        # x = x.view(batch_size, -1)

        x = self.decoder(x)

        loc, scale = extract_distr_params(x)

        return loc, scale

