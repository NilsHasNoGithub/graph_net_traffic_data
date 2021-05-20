import multiprocessing
import time
import numpy as np
import cityflow
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Any, Optional
import argparse
import os
import torch
from torch._C import Module
from load_data import LaneVehicleCountDataset, LaneVehicleCountDatasetMissing, RandData
from full_model import GNNVAEModel, GNNVAEForwardResult
from torch import nn
from torch.nn import functional
from torch.distributions import Categorical
from agent import Agent, MaxPressureAgent, FixedTimeAgent
from roadnet_graph import RoadnetGraph
from utils import store_json, load_json, Point, store_pkl
import random
from full_model import GNNVAEModel
from typing import List, AnyStr, Dict, Set, Optional, Iterator, Callable
import gen_data


N_STEPS = 100_000

@dataclass
class Args:
    cfg_file: AnyStr
    out_file: AnyStr
    shuffle_file: Optional[AnyStr]
    result_dir: str
    p_missing: float

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", "-c", type=str, required=True)
    parser.add_argument("--out-file", "-o", type=str, required=True)
    parser.add_argument("--model-file", "-f", type=str, required=True)
    parser.add_argument("--missing-file", "-m", type=float, default=0.3)

    parsed = parser.parse_args()

    return Args(
        parsed.cfg_file,
        parsed.out_file,
        parsed.model_file,
        parsed.missing_file
    )

def collect_data(engine: cityflow.Engine, graph: RoadnetGraph, n_steps: int, model: GNNVAEModel=None, hidden_intersections: Set[str]=None, agents: Optional[List[Agent]]=None, reset_pre=True, reset_post=True, print_info=True) -> List[Dict[str, Any]]:
    if reset_pre:
        engine.reset()

    data = []
    try:
        for i_step in range(n_steps):
            step_data = gen_data.gather_step_data(engine, graph, agents=agents)
            output = model(RoadnetGraph.tensor_data_from_time_step_data(step_data, hidden_intersections))
            mle = RoadnetGraph.lane_feats_per_intersection_from_tensor(output.x)
            

            for agent in agents:
                agent.act(engine, mle[agent.get_intersection()])

            engine.next_step()

            if print_info:
                print(f"\r i: {i_step}, avg travel time: " + str(engine.get_average_travel_time()), end="")

            data.append(step_data)

            if len(engine.get_vehicles(True)) == 0:
                break

        if reset_post:
            engine.reset()

    except KeyboardInterrupt:
        pass

    return data

def main(args: Args = None):

    ##Gendata init
    if args is None:
        args = parse_args()

    cityflow_cfg = load_json(args.cfg_file)

    graph = RoadnetGraph(cityflow_cfg["roadnetFile"])

    use_rl = cityflow_cfg["rlTrafficLight"]

    engine = cityflow.Engine(config_file=args.cfg_file, thread_num=multiprocessing.cpu_count())

    ##VAE init
    state = torch.load(args.model_file)
    model = GNNVAEModel.from_model_state(state)

    ##missing intersection
    hidden_intersections = np.loadtxt(args.missing)

    agents = []
    fta = 0
    mpa = 0
    if use_rl:
        for intersection in graph.intersection_list():
            if intersection.id in hidden_intersections:
                agents.append(FixedTimeAgent(intersection))
                fta += 1
            else:
                agents.append(MaxPressureAgent(intersection))
                mpa += 1

    print(f"fta: {fta}")
    print(f"mpa: {mpa}")

    data = collect_data(engine, graph, N_STEPS, model, hidden_intersections, agents=agents)

if __name__ == '__main__':
    main()