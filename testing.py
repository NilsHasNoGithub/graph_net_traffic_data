import multiprocessing
from dataclasses import dataclass
from typing import AnyStr, Optional, List, Callable
import argparse
import os
import torch
from torch.distributions import Categorical
from torch.optim import Optimizer

from load_data import LaneVehicleCountDataset, LaneVehicleCountDatasetMissing, RandData
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional

from gnn_model import IntersectionGNN
from full_model import GNNVAEModel, GNNVAEForwardResult
import matplotlib.pyplot as plt
import time
import random

from plotter import gen_data_visualization, gen_input_output_error_random_vizualization, gen_uncertainty_vizualization

from utils import DEVICE

@dataclass
class Args:
    roadnet_file: str
    data_file: str
    model_file: str
    result_dir: str
    p_missing: float

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet-file", "-r", type=str, required=True, help="roadnet data file")
    parser.add_argument("--data-file", "-d", type=str, required=True, help="data file containing the vehicles on each lane at each time point")
    parser.add_argument("--model-file", "-f", type=str, required=True)
    parser.add_argument("--result-dir", "-R", type=str, required=True)
    parser.add_argument("--p-missing", "-p", type=float, default=0.3)

    parsed = parser.parse_args()

    return Args(
        parsed.roadnet_file,
        parsed.data_file,
        parsed.model_file,
        parsed.result_dir,
        parsed.p_missing
    )


def main():
    args = parse_args()

    def o_file(name):
        return os.path.join(args.result_dir, name)

    _, dataset = LaneVehicleCountDatasetMissing.train_test_from_files(args.roadnet_file, args.data_file, p_missing=args.p_missing, shuffle=False, scale_by_road_len=False)
    # _, dataset = RandData(args.roadnet_file, p_missing=args.p_missing), RandData(args.roadnet_file, p_missing=args.p_missing)

    t = random.randint(0, len(dataset)-1)
    state = torch.load(args.model_file)
    model = GNNVAEModel.from_model_state(state)

    loss_fn = nn.MSELoss()

    sample, target, hidden_intersections = dataset.get_item(t, return_hidden_intersections=True)
    input_shape = dataset.input_shape()
    output_shape = dataset.output_shape()

    output: GNNVAEForwardResult = model(sample.view(1, *input_shape))

    params = output.params_decoder
    # y = y.get_output()
    y = output.x
    y = y.view(*output_shape)

    # random_y = model.sample()
    # # random_y = random_y.get_output()
    # random_y = random_y.x
    # random_y = random_y.view(*output_shape)

    #
    torch.set_printoptions(edgeitems=100000)

    # print(f"MSE loss: {loss_fn(y, target)}")


    # print(sample[22,:])
    # print(torch.round(y[22,:]))

    # pars = model.parameters()

    print(f"num parameters: {sum(p.numel() for p in model.parameters())}")

    distr = model.distr().torch_distr(*params)

    if isinstance(distr, Categorical):
        probs = params[0]
        vars = - 1.0 * (probs * torch.log2(probs+ 0.0000000001)).sum(-1)
        # vars = distr.entropy()
    else:
        vars = distr.variance

        # xs = []
        # for _ in range(1000):
        #     output = model(sample.view(1, *input_shape))
        #     xs.append(output.x)
        #
        # xs = torch.stack(xs, dim=-1)
        # vars = torch.var(xs, dim=-1)

    var_result = gen_uncertainty_vizualization(dataset, vars, no_data_intersections=hidden_intersections)
    var_result.write_to_png(o_file("variances.png"))

    hidden_intersections_idxs = sample[:, 0].long() == 1

    hidden_out = y[hidden_intersections_idxs]
    hidden_target = target[hidden_intersections_idxs]

    hidden_mse = functional.mse_loss(hidden_out, hidden_target)

    obs_out = y[~hidden_intersections_idxs]
    obs_target = target[~hidden_intersections_idxs]

    obs_mse = functional.mse_loss(obs_out, obs_target)

    print(f"observed mse: {obs_mse},  hidden mse: {hidden_mse}")

    squared_errors = (y - target) ** 2

    ior_result = gen_input_output_error_random_vizualization(dataset, target, y, squared_errors, torch.ones(target.shape), no_data_intersections=hidden_intersections, scale_data_by_road_len=True)

    ior_result.input.write_to_png(o_file("input.png"))
    ior_result.output.write_to_png(o_file("output.png"))
    ior_result.error.write_to_png(o_file("errors.png"))
    ior_result.random.write_to_png(o_file("random.png"))

    enc_loc, enc_scale = output.params_encoder

    enc_loc = enc_loc.squeeze()
    enc_scale = enc_scale.squeeze()

    enc_loc_hid, enc_loc_obs = enc_loc[hidden_intersections_idxs], enc_loc[~hidden_intersections_idxs]
    enc_scale_hid, enc_scale_obs = enc_scale[hidden_intersections_idxs], enc_scale[~hidden_intersections_idxs]

    enc_loc_hid, enc_loc_obs, enc_scale_hid, enc_scale_obs = (x.flatten().detach().numpy() for x in (enc_loc_hid, enc_loc_obs, enc_scale_hid, enc_scale_obs))

    plt.clf()
    plt.title("Mean and standard deviation of parameters in latent space")
    plt.scatter(enc_loc_hid, enc_scale_hid, label="unobserved", alpha=0.5)
    plt.scatter(enc_loc_obs, enc_scale_obs, label="observed", alpha=0.5)
    plt.xlabel("Mean")
    plt.ylabel("Standard deviation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(o_file("encoder_params.png"))


    # print(torch.round(model.sample()))

if __name__ == '__main__':
    main()


