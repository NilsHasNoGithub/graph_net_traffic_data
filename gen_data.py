import multiprocessing
import cityflow
from dataclasses import dataclass
from typing import AnyStr, Dict, List
import argparse
import pandas as pd
import json
from utils import store_json
import random


N_STEPS = 3600

@dataclass
class Args:
    cfg_file: AnyStr
    out_file: AnyStr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", "-c", type=str, required=True)
    parser.add_argument("--out-file", "-o", type=str, required=True)

    parsed = parser.parse_args()

    return Args(
        parsed.cfg_file,
        parsed.out_file
    )

def collect_data(engine: cityflow.Engine, n_steps: int, reset_pre=True, reset_post=True):
    if reset_pre:
        engine.reset()

    data = []

    for _ in range(n_steps):
        engine.next_step()
        vh_counts = engine.get_lane_vehicle_count()
        data.append(vh_counts)

    if reset_post:
        engine.reset()


    return data

def chunks(l, n) -> List[List]:
    return [l[i:i+n] for i in range(0, len(l), n)]

def flatten(l: List[List]) -> List:
    return [item for sl in l for item in sl]

def main(args: Args = None):

    if args is None:
        args = parse_args()

    engine = cityflow.Engine(config_file=args.cfg_file, thread_num=multiprocessing.cpu_count())

    data = collect_data(engine, N_STEPS)

    data = chunks(data, 10)
    random.shuffle(data)
    data = flatten(data)
    store_json(data, args.out_file)



if __name__ == '__main__':
    main()