import multiprocessing
import cityflow
from dataclasses import dataclass
from typing import AnyStr, Dict, List, Any, Optional
import argparse
import torch
from full_model import GNNVAEModel
from agents import Agent, MaxPressureAgent, FixedTimeAgent, UncertainMaxPressureAgent
from roadnet_graph import RoadnetGraph
from utils import store_json, load_json, store_pkl
import random
from full_model import GNNVAEModel
from typing import List, AnyStr, Dict, Set, Optional
import gen_data


N_STEPS =  3600

@dataclass
class Args:
    cfg_file: AnyStr
    out_file: AnyStr
    model_file: AnyStr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-file", "-c", type=str, required=True)

    parsed = parser.parse_args()

    return Args(
        parsed.cfg_file,
        None,
        None
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


def gather_step_data(engine: cityflow.Engine, graph: RoadnetGraph, agents: Optional[List[Agent]]=None) -> dict:
    vh_counts = engine.get_lane_vehicle_count()
    lane_vhs = engine.get_lane_vehicles()

    lane_vh_infos = {}

    for lane_id, vhs in lane_vhs.items():
        car_infos = (CarInfo.from_engine(engine, graph, vh_id) for vh_id in vhs)
        lane_vh_infos[lane_id] = [{"id": ci.id, "closestIntersection": ci.closest_intersection_id} for ci in car_infos]

    intersection_phases = {}

    if agents is not None:
        for agent in agents:
            intersection_phases[agent.get_intersection().id] = agent.get_prev_phase()


    return {
        "laneCounts": vh_counts,
        "laneVehicleInfos": lane_vh_infos,
        "intersectionPhases": intersection_phases
    }


def collect_data(engine: cityflow.Engine, graph: RoadnetGraph, n_steps: int, hidden_intersections: Set[str]=None, agents: List[Agent]=None, reset_pre=True, reset_post=True, print_info=True) -> List[Dict[str, Any]]:
    if reset_pre:
        engine.reset()

    data = []
    try:
        for i_step in range(n_steps):
            step_data = gather_step_data(engine, graph, agents=agents)
            
            for agent in agents:
                agent.act(engine, step_data)

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

    agents = []
    agents = []
    hidden_observations = set()

    non_observed_count = 0
    fully_observed_count = 0

    amount_unobserved = 5
    rand_intersections = False

    unobserved_intersections = set()

    if use_rl:
        if rand_intersections:
            for intersection in graph.intersection_list():
                if random.random() < 0.1:
                    unobserved_intersections.add(intersection.id)
                    agents.append(FixedTimeAgent(intersection))
                    non_observed_count += 1
                else:
                    agents.append(MaxPressureAgent(intersection))
                    fully_observed_count += 1
        else:
            intersections = set(graph.intersection_list())
            #unobserved_intersections = set(random.choices(graph.intersection_list(), k=amount_unobserved))
            #unobserved_intersections = {'intersection_1_14', 'intersection_2_12', 'intersection_2_13',
            #                            'intersection_3_2',
            #                            'intersection_1_12'}
            unobserved_intersections = {'intersection_1_2', 'intersection_2_1', 'intersection_2_8', 'intersection_1_4',
                                        'intersection_2_4', 'intersection_1_6', 'intersection_1_14', 'intersection_3_11',
                                        'intersection_2_12', 'intersection_3_16', 'intersection_3_3', 'intersection_3_14',
                                        'intersection_1_13', 'intersection_3_9', 'intersection_3_12', 'intersection_3_4'}

            observed_intersections = intersections.difference(unobserved_intersections)

            for intersection_id in unobserved_intersections:
                hidden_observations.add(intersection_id)
                agents.append(FixedTimeAgent(graph.intersection_dict()[intersection_id]))
                non_observed_count += 1

            for intersection_id in observed_intersections:
                agents.append(MaxPressureAgent(intersection_id))
                fully_observed_count += 1

    print(f"non_observed_count: {non_observed_count}")
    print("Unobserved intersections:", [x for x in unobserved_intersections])
    print(f"fully_observed_count: {fully_observed_count}")

    data = collect_data(engine, graph, N_STEPS, hidden_intersections=hidden_observations, agents=agents)

if __name__ == '__main__':
    main()