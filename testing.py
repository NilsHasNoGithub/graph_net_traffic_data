import multiprocessing
from dataclasses import dataclass
from typing import AnyStr, Optional, List, Callable
import argparse
import os
import torch
from torch.optim import Optimizer

from load_data import LaneVehicleCountDataset, LaneVehicleCountDatasetMissing
from torch.utils.data import DataLoader
from torch import nn

from gnn_model import IntersectionGNN
from full_model import GNNVAEModel, GNNVAEForwardResult
import matplotlib.pyplot as plt
import time

from plotter import gen_data_visualization, gen_input_output_random_vizualization, gen_uncertainty_vizualization

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

    t = 2000

    dataset, _= LaneVehicleCountDatasetMissing.train_test_from_files(args.roadnet_file, args.data_file, p_missing=args.p_missing, shuffle=False, scale_by_road_len=False)

    state = torch.load(args.model_file)
    model = GNNVAEModel.from_model_state(state)

    loss_fn = nn.MSELoss()

    sample, target, hidden_intersections = dataset.get_item(t, return_hidden_intersections=True)
    input_shape = dataset.input_shape()
    output_shape = dataset.output_shape()

    output = model(sample.view(1, *input_shape))

    params = output.params_decoder
    # y = y.get_output()
    y = output.x
    y = y.view(*output_shape)

    random_y = model.sample()
    # random_y = random_y.get_output()
    random_y = random_y.x
    random_y = random_y.view(*output_shape)

    #
    torch.set_printoptions(edgeitems=100000)

    print(f"MSE loss: {loss_fn(y, target)}")


    # print(sample[22,:])
    # print(torch.round(y[22,:]))

    # pars = model.parameters()

    print(f"num parameters: {sum(p.numel() for p in model.parameters())}")

    if len(params) == 2:
        scale = params[1]

        var_result = gen_uncertainty_vizualization(dataset, scale, no_data_intersections=hidden_intersections)
        var_result.write_to_png("results/variances.png")

    ior_result = gen_input_output_random_vizualization(dataset, target, y, random_y, no_data_intersections=hidden_intersections, scale_data_by_road_len=True)

    ior_result.input.write_to_png("results/input.png")
    ior_result.output.write_to_png("results/output.png")
    ior_result.random.write_to_png("results/random.png")

    # input_names = ['Original']
    # output_names = ['Reconstructed']
    # torch.onnx.export(model, sample.view(1, *shape), 'results/model.onnx', input_names=input_names, output_names=output_names)


    # print(torch.round(model.sample()))

if __name__ == '__main__':
    main()


