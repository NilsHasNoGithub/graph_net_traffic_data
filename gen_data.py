import multiprocessing
import time

import cityflow
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Any, Optional
import argparse

from roadnet_graph import RoadnetGraph
from utils import store_json, load_json, Point, store_pkl
import random


N_STEPS = 3600

@dataclass
class Args:
    cfg_file: AnyStr
    out_file: AnyStr
    shuffle_file: Optional[AnyStr]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", "-c", type=str, required=True)
    parser.add_argument("--out-file", "-o", type=str, required=True)
    parser.add_argument("--shuffled-out-file", "-s", type=str)

    parsed = parser.parse_args()

    return Args(
        parsed.cfg_file,
        parsed.out_file,
        parsed.shuffled_out_file
    )

@dataclass
class CarInfo:
    id: str
    closest_intersection_id: str

    @staticmethod
    def from_engine(engine: cityflow.Engine, graph: RoadnetGraph, vh_id: str) -> 'CarInfo':
        vh_info = engine.get_vehicle_info(vh_id)

        road_id = vh_info["road"]
        road = graph.road_dict()[road_id]

        distance = vh_info["distance"]

        if float(distance) < road.length() / 2:
            cl_i = road.start_intersection_id
        else:
            cl_i = road.end_intersection_id

        return CarInfo(
            vh_id,
            cl_i
        )



def gather_step_data(engine: cityflow.Engine, graph: RoadnetGraph) -> dict:
    vh_counts = engine.get_lane_vehicle_count()
    lane_vhs = engine.get_lane_vehicles()

    lane_vh_infos = {}

    for lane_id, vhs in lane_vhs.items():
        car_infos = (CarInfo.from_engine(engine, graph, vh_id) for vh_id in vhs)
        lane_vh_infos[lane_id] = [{"id": ci.id, "closestIntersection": ci.closest_intersection_id} for ci in car_infos]

    return {
        "laneCounts": vh_counts,
        "laneVehicleInfos": lane_vh_infos
    }




def collect_data(engine: cityflow.Engine, graph: RoadnetGraph, n_steps: int, reset_pre=True, reset_post=True) -> List[Dict[str, Any]]:
    if reset_pre:
        engine.reset()

    data = []

    for _ in range(n_steps):
        engine.next_step()
        step_data = gather_step_data(engine, graph)

        data.append(step_data)

    if reset_post:
        engine.reset()


    return data



def main(args: Args = None):

    if args is None:
        args = parse_args()

    cityflow_cfg = load_json(args.cfg_file)

    graph = RoadnetGraph(cityflow_cfg["roadnetFile"])

    engine = cityflow.Engine(config_file=args.cfg_file, thread_num=multiprocessing.cpu_count())

    data = collect_data(engine, graph, N_STEPS)

    store_json(data, args.out_file)
    if args.shuffle_file is not None:
        random.shuffle(data)
        store_json(data, args.shuffle_file)


if __name__ == '__main__':
    main()