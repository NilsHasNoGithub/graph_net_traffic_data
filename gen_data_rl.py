import multiprocessing
import time
import numpy as np

import cityflow
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Any, Optional
import argparse

from roadnet_graph import RoadnetGraph
from utils import store_json, load_json, Point, store_pkl
import random

N_STEPS = 3600  # Previously 3600


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


def collect_data(engine: cityflow.Engine, graph: RoadnetGraph, n_steps: int, reset_pre=True, reset_post=True) -> List[
    Dict[str, Any]]:
    if reset_pre:
        engine.reset()

    data = []

    for temp_step in range(n_steps):
        temp_data = gather_step_data(engine, graph)  # Better to add last datapoint in function header.

        for intersection in graph.intersection_list():
            # Get ordered counts of incoming and outgoing lanes.
            incoming_counts = []
            outgoing_counts = []

            # Get densities for MaxPressure calculations
            for incoming_road in intersection.incoming_roads:
                road_length = incoming_road.length()
                for incoming_lane in incoming_road.lanes:
                    incoming_counts.append(temp_data['laneCounts'][incoming_lane] / road_length)

            for outgoing_road in intersection.outgoing_roads:
                road_length = outgoing_road.length()
                for outgoing_lane in outgoing_road.lanes:
                    outgoing_counts.append(temp_data['laneCounts'][outgoing_lane] / road_length)


            # TODO: Put in an RL class file.
            # MaxPressure algo: https://arxiv.org/pdf/1904.08117.pdf
            # (possible addition is density https://faculty.ist.psu.edu/jessieli/Publications/2019-KDD-presslight.pdf)

            # Lanes start from innerside.

            # Phase id 1: W-E, E-W.
            west_east_count = incoming_counts[1] - sum(outgoing_counts[9:12])
            east_west_count = incoming_counts[10] - sum(outgoing_counts[0:3])

            phase_1 = west_east_count + east_west_count

            # Phase id 2: N-S, S-N.
            north_south_count = incoming_counts[7] - sum(outgoing_counts[3:6])
            south_north_count = incoming_counts[4] - sum(outgoing_counts[6:9])

            phase_2 = north_south_count + south_north_count

            # Phase id 3: W-N, E-S.
            west_north_count = incoming_counts[0] - sum(outgoing_counts[6:9])
            east_south_count = incoming_counts[9] - sum(outgoing_counts[3:6])

            phase_3 = west_north_count + east_south_count

            # Phase id 4: N-E, S-W
            north_east_count = incoming_counts[6] - sum(outgoing_counts[9:12])
            south_west_count = incoming_counts[3] - sum(outgoing_counts[0:3])

            phase_4 = north_east_count + south_west_count

            all_phases = [phase_1, phase_2, phase_3, phase_4]

            if random.random() < EPSILON:
                chosen_phase_id = random.randrange(0, len(all_phases))  # Choose random action
            else:
                chosen_phase_id = np.argmax(all_phases) + 1  # Choose "best" action

            t_since_last_change[intersection.id][0] += 1

            if t_since_last_change[intersection.id][0] >= T_MIN:
                # Change the phase only when T-minus has been surpassed
                engine.set_tl_phase(intersection.id, chosen_phase_id)

                # If we have a new phase, reset the timer.
                if chosen_phase_id != t_since_last_change[intersection.id][1]:
                    t_since_last_change[intersection.id] = [0, chosen_phase_id]

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
