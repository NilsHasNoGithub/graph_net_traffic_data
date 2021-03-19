import multiprocessing
from dataclasses import dataclass
from typing import AnyStr, Optional, List, Callable
import argparse
import os
import torch
from torch.optim import Optimizer

from load_data import LaneVehicleCountDataset
from torch.utils.data import DataLoader
from torch import nn

from gnn_model import IntersectionGNN
from full_model import GNNVAEModel
import matplotlib.pyplot as plt
import time

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

    sample = data_train[500]
    shape = data_train.sample_shape()

    # y = model(sample.view(1, *shape))
    # y = y.view(*shape)
    #
    torch.set_printoptions(edgeitems=100000)
    # print(sample[22,:])
    # print(torch.round(y[22,:]))

    # pars = model.parameters()

    print(*[p.numel() for p in model.parameters()])

    # print(torch.round(model.sample()))

if __name__ == '__main__':
    main()


