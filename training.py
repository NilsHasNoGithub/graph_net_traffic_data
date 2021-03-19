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
    model_file: Optional[str]
    result_dir: Optional[str]
    n_epochs: int
    batch_size: int

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet-file", "-r", type=str, required=True, help="roadnet data file")
    parser.add_argument("--data-file", "-d", type=str, required=True, help="data file containing the vehicles on each lane at each time point")
    parser.add_argument("--model-file", "-f", type=str)
    parser.add_argument("--result-dir", "-R", type=str)
    parser.add_argument("--n-epochs", "-n", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=50)

    parsed = parser.parse_args()

    return Args(
        parsed.roadnet_file,
        parsed.data_file,
        parsed.model_file,
        parsed.result_dir,
        parsed.n_epochs,
        parsed.batch_size,
    )

@dataclass
class TrainResults:
    train_losses: List[float]
    val_losses: List[float]

def train(
        model: GNNVAEModel,
        optimizer: Optimizer,
        loss_fn: Callable,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device=None,
        n_epochs: int = 1000,
        loss_fn_weight: float = 10.0,
        model_file=None
    ):
    """

    :param model:
    :param optimizer:
    :param loss_fn:
    :param train_dl:
    :param val_dl:
    :param n_epochs:
    :param loss_fn_weight: weight to give to the loss function, relative to KL Loss
    :return:
    """
    train_losses = []
    val_losses = []

    if device is None:
        device = DEVICE
    model.to(device)

    for i_epoch in range(n_epochs):

        cur_train_loss = 0.0
        cur_val_loss = 0.0

        t = time.time()

        for i, inputs in enumerate(train_dl):

            inputs = inputs.to(device)
            targets = inputs.clone().detach()

            optimizer.zero_grad()

            predicted, kl_div_loss = model(inputs, calc_kl_div=True)
            loss = loss_fn_weight * loss_fn(predicted, targets) + kl_div_loss
            loss.backward()

            optimizer.step()

            cur_train_loss += loss.item()

            t1 = time.time()
            print(
                f"\repoch {i_epoch + 1}/{n_epochs}, batch {i + 1}/{len(train_dl)}, train_loss: {loss.item()} , took {t1 - t}",
                end="")
            t = t1

        with torch.no_grad():
            for i, inputs in enumerate(val_dl):
                inputs = inputs.to(device)
                targets = inputs.clone().detach()

                predicted, kl_div_loss = model(inputs, calc_kl_div=True)
                loss = loss_fn_weight * loss_fn(predicted, targets) + kl_div_loss

                cur_val_loss += loss.item()

                t1 = time.time()
                print(
                    f"\repoch {i_epoch + 1}/{n_epochs}, batch {i + 1}/{len(val_dl)}, val_loss: {loss.item()} , took {t1 - t}",
                    end=""
                )
                t = t1

        train_losses.append(cur_train_loss / len(train_dl))
        val_losses.append(cur_val_loss / len(val_dl))

        print(f"\repoch {i_epoch + 1}/{n_epochs}: train_loss: {train_losses[-1]}, val_loss: {val_losses[-1]}")

        if model_file is not None:
            model.cpu()
            torch.save(model.get_model_state(), model_file)
            model.to(device)

    return TrainResults(
        train_losses,
        val_losses
    )


def main():
    args = parse_args()

    # torch.set_num_threads(multiprocessing.cpu_count())

    data_train, data_test = LaneVehicleCountDataset.train_test_from_files(args.roadnet_file, args.data_file)

    train_dl = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    if args.model_file is not None and os.path.isfile(args.model_file):
        state = torch.load(args.model_file)
        model = GNNVAEModel.from_model_state(state)
    else:
        model = GNNVAEModel(data_train.sample_shape()[1], data_train.graph_adjacency_list())

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    results = train(
        model,
        optimizer,
        loss_fn,
        train_dl,
        val_dl,
        n_epochs=args.n_epochs,
        model_file=args.model_file
    )


    if args.model_file is not None:
        model.cpu()
        torch.save(model.get_model_state(), args.model_file)

    if args.result_dir is not None:
        os.makedirs(args.result_dir, exist_ok=True)

        fig, subplots = plt.subplots(2, 1, figsize=(10, 5))

        p0, p1 = subplots.flatten()

        p0.set_title("Train losses")
        p0.plot(results.train_losses)

        p1.set_title("Validation losses")
        p1.plot(results.val_losses)

        fig.tight_layout()
        fig.savefig(os.path.join(args.result_dir, "losses.png"))

        plt.close(fig)


if __name__ == '__main__':
    main()