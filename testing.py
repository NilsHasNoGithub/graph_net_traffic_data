import multiprocessing
from dataclasses import dataclass
from typing import AnyStr, Optional, List, Callable
import argparse
import os
import torch
from torch.optim import Optimizer
from torch_geometric.data import Data

from load_data import LaneVehicleCountDataset
from torch.utils.data import DataLoader
from torch import nn

from gnn_model import IntersectionGNN
from full_model import GNNVAEModel
import matplotlib.pyplot as plt
import time

from plotter import gen_data_visualization

from utils import DEVICE

@dataclass
class Args:
    roadnet_file: str
    data_file: str
    model_file: str
    result_dir: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet-file", "-r", type=str, required=True, help="roadnet data file")
    parser.add_argument("--data-file", "-d", type=str, required=True, help="data file containing the vehicles on each lane at each time point")
    parser.add_argument("--model-file", "-f", type=str, required=True)
    parser.add_argument("--result-dir", "-R", type=str, required=True)

    parsed = parser.parse_args()

    return Args(
        parsed.roadnet_file,
        parsed.data_file,
        parsed.model_file,
        parsed.result_dir,
    )


def main():
    args = parse_args()

    data_train, data_val = LaneVehicleCountDataset.train_test_from_files(args.roadnet_file, args.data_file)

    state = torch.load(args.model_file)
    model = GNNVAEModel.from_model_state(state)

    loss_fn = nn.MSELoss()

    sample = data_val[500]
    sample_geo = data_val.get_geometric_datasets()
    sample_edge_index = sample_geo[500].edge_index
    shape = data_val.sample_shape()

    y = model(sample.view(1, *shape), sample_edge_index)
    y = y.view(*shape)

    random_y = model.sample(sample_edge_index).view(*shape)

    #
    torch.set_printoptions(edgeitems=100000)

    print(f"MSE loss: {loss_fn(y, sample)}")


    # print(sample[22,:])
    # print(torch.round(y[22,:]))

    # pars = model.parameters()

    print(f"num parameters: {sum(p.numel() for p in model.parameters())}")

    input_data_im = gen_data_visualization(data_val, sample)
    output_data_im = gen_data_visualization(data_val, y)
    random_data_im = gen_data_visualization(data_val, random_y)

    input_data_im.write_to_png("results/input.png")
    output_data_im.write_to_png("results/output.png")
    random_data_im.write_to_png("results/random.png")

    # input_names = ['Original']
    # output_names = ['Reconstructed']
    # torch.onnx.export(model, sample.view(1, *shape), 'results/model.onnx', input_names=input_names, output_names=output_names)


    # print(torch.round(model.sample()))

if __name__ == '__main__':
    main()


