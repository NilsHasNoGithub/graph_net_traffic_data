import multiprocessing

import numpy as np
from dataclasses import dataclass
from typing import AnyStr, Optional, List, Callable
import argparse
import os
import torch
from torch.optim import Optimizer

from load_data import LaneVehicleCountDataset, LaneVehicleCountDatasetMissing, RandData
from torch.utils.data import DataLoader
from torch import nn, Tensor
from torch.nn import functional

from gnn_model import IntersectionGNN
from full_model import GNNVAEModel, GNNVAEForwardResult
from vae_net import VAELogNormalDistr, VAECategoricalDistr, VAEDistr, VAENormalDistr
import matplotlib.pyplot as plt
import time

from utils import DEVICE
from vae_net import VAEEncoderForwardResult

from itertools import product as cart_product

from enum import Enum

# from agents import CoLightAgent
# import keras

class SupportedVaeDistr(Enum):
    LOG_NORMAL = 1
    NORMAL = 2
    CATEGORICAL = 3

    @staticmethod
    def from_str(s: str):
        s = s.lower()

        if s == "lognormal":
            return SupportedVaeDistr.LOG_NORMAL
        elif s == "normal":
            return SupportedVaeDistr.NORMAL
        elif s == "categorical":
            return SupportedVaeDistr.CATEGORICAL

        raise ValueError(f"'{s}' is not a supported distribution")

    def to_distr(self) -> VAEDistr:
        if self == SupportedVaeDistr.LOG_NORMAL:
            return VAELogNormalDistr()
        elif self == SupportedVaeDistr.NORMAL:
            return VAENormalDistr()
        elif self == SupportedVaeDistr.CATEGORICAL:
            return VAECategoricalDistr(30)

        raise ValueError(f"Illegal enum state: {self}")


@dataclass
class Args:
    roadnet_file: str
    data_file: str
    model_file: Optional[str]
    result_dir: Optional[str]
    n_epochs: int
    batch_size: int
    p_missing: float
    vae_distr: Optional[SupportedVaeDistr]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roadnet-file", "-r", type=str, required=True, help="roadnet data file")
    parser.add_argument("--data-file", "-d", type=str, required=True, help="data file containing the vehicles on each lane at each time point")
    parser.add_argument("--model-file", "-f", type=str)
    parser.add_argument("--result-dir", "-R", type=str)
    parser.add_argument("--n-epochs", "-n", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=50)
    parser.add_argument("--p-missing", "-p", type=float, default=0.3)
    parser.add_argument("--vae-distr", type=str, required=False)

    parsed = parser.parse_args()

    return Args(
        parsed.roadnet_file,
        parsed.data_file,
        parsed.model_file,
        parsed.result_dir,
        parsed.n_epochs,
        parsed.batch_size,
        parsed.p_missing,
        SupportedVaeDistr.from_str(parsed.vae_distr) if parsed.vae_distr else None
    )

def _averaged(items: List[List[float]]) -> List[float]:
    return [float(np.mean(l)) for l in items]

@dataclass
class Losses:
    losses: List[List[float]]
    mse_losses: List[List[float]]
    mse_losses_observed: List[List[float]]
    mse_losses_unobserved: List[List[float]]

    @staticmethod
    def empty():
        return Losses(
            *[[] for _ in range(4)]
        )

    def losses_avgd(self):
        return _averaged(self.losses)

    def mse_losses_avgd(self):
        return _averaged(self.mse_losses)

    def mse_losses_observed_avgd(self):
        return _averaged(self.mse_losses_observed)

    def mse_losses_unobserved_avgd(self):
        return _averaged(self.mse_losses_unobserved)


@dataclass
class TrainResults:
    train_losses: Losses
    val_losses: Losses

    @staticmethod
    def empty():
        return TrainResults(
            Losses.empty(),
            Losses.empty(),
        )


    def new_epoch(self):
        for losses in (self.train_losses, self.val_losses):
            for l in (
                losses.losses,
                losses.mse_losses,
                losses.mse_losses_observed,
                losses.mse_losses_unobserved
            ):
                l.append([])

    def _update(self, train: bool, inputs: Tensor, output: GNNVAEForwardResult, targets: Tensor, loss_fn: Callable[[GNNVAEForwardResult, Tensor], Tensor]) -> Tensor:
        losses = self.train_losses if train else self.val_losses

        modes = output.x

        loss_tnsr = loss_fn(output, targets)
        loss = loss_tnsr.item()
        mse_loss = functional.mse_loss(modes, targets).item()

        hidden_intersections_idxs = inputs[..., 0].long() == 1 # first feature encodes whether an intersecition is hidden

        modes_unobserved = modes[hidden_intersections_idxs]
        targets_unobserved = targets[hidden_intersections_idxs]

        modes_observed = modes[~hidden_intersections_idxs]
        targets_observed = targets[~hidden_intersections_idxs]

        mse_loss_observed = functional.mse_loss(modes_observed, targets_observed).item()
        mse_loss_unobserved = functional.mse_loss(modes_unobserved, targets_unobserved).item()

        losses.losses[-1].append(loss)
        losses.mse_losses[-1].append(mse_loss)
        losses.mse_losses_observed[-1].append(mse_loss_observed)
        losses.mse_losses_unobserved[-1].append(mse_loss_unobserved)

        return loss_tnsr


    def update_train(self, inputs: Tensor, output: GNNVAEForwardResult, targets: Tensor, loss_fn: Callable[[GNNVAEForwardResult, Tensor], Tensor]) -> Tensor:
        """
                Update all train losses
                :param inputs:
                :param output:
                :param targets:
                :param loss_fn:
                :return: returns the loss computed by `loss_fn`
                """
        return self._update(True, inputs, output, targets, loss_fn)

    def update_val(self, inputs: Tensor, output: GNNVAEForwardResult, targets: Tensor, loss_fn: Callable[[GNNVAEForwardResult, Tensor], Tensor]) -> Tensor:
        """
        Update all validation losses
        :param inputs:
        :param output:
        :param targets:
        :param loss_fn:
        :return: the loss computed by `loss_fn`
        """
        return self._update(False, inputs, output, targets, loss_fn)


def train(
        model: GNNVAEModel,
        optimizer: Optimizer,
        loss_fn: Callable,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device=None,
        n_epochs: int = 1000,
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

    results = TrainResults.empty()

    if device is None:
        device = DEVICE
    model.to(device)

    for i_epoch in range(n_epochs):

        results.new_epoch()

        t = time.time()

        for i, (inputs, targets) in enumerate(train_dl):

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            output: GNNVAEForwardResult= model(inputs)
            loss = results.update_train(inputs, output, targets, loss_fn)

            loss.backward()

            optimizer.step()

            t1 = time.time()
            print(
                f"\repoch {i_epoch + 1}/{n_epochs}, batch {i + 1}/{len(train_dl)}, train_loss: {loss.item()} , took {t1 - t}",
                end="")
            t = t1

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)

                output: GNNVAEForwardResult = model(inputs)
                loss = results.update_val(inputs, output, targets, loss_fn)

                t1 = time.time()
                print(
                    f"\repoch {i_epoch + 1}/{n_epochs}, batch {i + 1}/{len(val_dl)}, val_loss: {loss.item()} , took {t1 - t}",
                    end=""
                )
                t = t1

        train_loss = np.mean(results.train_losses.losses[-1])
        val_loss = np.mean(results.val_losses.losses[-1])
        mse_train_loss = np.mean(results.train_losses.mse_losses[-1])
        mse_val_loss = np.mean(results.val_losses.mse_losses[-1])

        print(f"\repoch {i_epoch + 1}/{n_epochs}: train_loss: {train_loss}, val_loss: {val_loss}, mse_train_loss: {mse_train_loss}, mse_val_loss: {mse_val_loss}")

        if model_file is not None:
            model.cpu()
            torch.save(model.get_model_state(), model_file)
            model.to(device)

    return results


def mk_loss_fn(model: GNNVAEModel, log_prob_weight=10.0) -> Callable[[GNNVAEForwardResult, Tensor], Tensor]:
    def loss_fn(result: GNNVAEForwardResult, targets: Tensor):

        return result.kl_div + log_prob_weight * -1.0 * torch.mean(model.distr().log_prob(result.params_decoder, targets).sum(-1))
        # return result.kl_div  + log_prob_weight * functional.mse_loss(result.x, targets)


    return loss_fn


def main():
    args = parse_args()

    # torch.set_num_threads(multiprocessing.cpu_count())

    p_intersection_hidden_distr = torch.distributions.Beta(1.575, 3.675)
    # p_intersection_hidden_distr = 0.0

    data_train, data_test = LaneVehicleCountDatasetMissing.train_test_from_files(args.roadnet_file, args.data_file, p_missing=p_intersection_hidden_distr, scale_by_road_len=False)
    # data_train, data_test = RandData(args.roadnet_file, p_missing=p_intersection_hidden_distr), RandData(args.roadnet_file, size=500, p_missing=p_intersection_hidden_distr)

    train_dl = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(data_test, batch_size=args.batch_size, shuffle=True)


    if args.model_file is not None and os.path.isfile(args.model_file):
        state = torch.load(args.model_file)
        model = GNNVAEModel.from_model_state(state)

        err_str = "Program argument does not match distribution of loaded model"
        if args.vae_distr == SupportedVaeDistr.LOG_NORMAL:
            assert isinstance(model.distr(), VAELogNormalDistr), err_str
        elif args.vae_distr == SupportedVaeDistr.CATEGORICAL:
            assert isinstance(model.distr(), VAECategoricalDistr), err_str
    else:

        model = GNNVAEModel(
            data_train.input_shape()[1],
            data_train.graph_adjacency_list(),
            n_out=data_train.output_shape()[1],
            decoder_distr=args.vae_distr.to_distr() if args.vae_distr is not None else None
        )

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adagrad(model.parameters())

    results = train(
        model,
        optimizer,
        mk_loss_fn(model),
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

        ### losses for which network optimizes

        fig = plt.figure(figsize=(10,5))

        p = fig.gca()

        p.set_title("Losses: combination of Kullback Leibler divergence and log probabilities")
        p.plot(results.train_losses.losses_avgd(), label="train loss")
        p.plot(results.val_losses.losses_avgd(), label="validation loss")
        p.set_xlabel("$i$th epoch")
        p.set_ylabel("Loss")
        p.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.result_dir, "losses.png"))

        plt.close(fig)

        ### all mse losses

        fig = plt.figure(figsize=(10, 5))

        p = fig.gca()

        p.set_title("Losses: Mean squared error")
        p.plot(results.train_losses.mse_losses_avgd(), label="train loss")
        p.plot(results.val_losses.mse_losses_avgd(), label="validation loss")
        p.set_xlabel("$i$th epoch")
        p.set_ylabel("Loss")
        p.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.result_dir, "mse_losses.png"))

        plt.close(fig)

        ### observed only

        fig = plt.figure(figsize=(10, 5))

        p = fig.gca()

        p.set_title("Losses: Mean squared error at observed intersections")
        p.plot(results.train_losses.mse_losses_observed_avgd(), label="train loss")
        p.plot(results.val_losses.mse_losses_observed_avgd(), label="validation loss")
        p.set_xlabel("$i$th epoch")
        p.set_ylabel("Loss")
        p.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.result_dir, "mse_losses_observed.png"))

        plt.close(fig)

        ### Unobserved only

        fig = plt.figure(figsize=(10, 5))

        p = fig.gca()

        p.set_title("Losses: Mean squared error at unobserved intersections")
        p.plot(results.train_losses.mse_losses_unobserved_avgd(), label="train loss")
        p.plot(results.val_losses.mse_losses_unobserved_avgd(), label="validation loss")
        p.set_xlabel("$i$th epoch")
        p.set_ylabel("Loss")
        p.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(args.result_dir, "mse_losses_unobserved.png"))

        plt.close(fig)


if __name__ == '__main__':
    main()