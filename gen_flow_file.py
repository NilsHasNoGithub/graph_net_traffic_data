import argparse
from dataclasses import dataclass

from roadnet_graph import RoadnetGraph
import random

from utils import store_json

VEHICLE = {
      "length": 5.0,
      "width": 2.0,
      "maxPosAcc": 2.0,
      "maxNegAcc": 4.5,
      "usualPosAcc": 2.0,
      "usualNegAcc": 4.5,
      "minGap": 2.5,
      "maxSpeed": 16.67,
      "headwayTime": 1.5
}


@dataclass
class Args:
    road_net_file: str
    n_flows_min: int
    n_flows_max: int
    interval_min: float
    interval_max: float
    n_epochs: int
    epoch_length: int
    out_file: str

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road-net-file", "-r", type=str, required=True)
    parser.add_argument("--n-flows-min", type=int, required=True)
    parser.add_argument("--n-flows-max", type=int, required=True)
    parser.add_argument("--n-epochs", "-N", type=int, default=10)
    parser.add_argument("--epoch-length", "-l", type=int, default=1000)
    parser.add_argument("--interval-min", "-i", type=float, required=True)
    parser.add_argument("--interval-max", "-I", type=float, required=True)
    parser.add_argument("--out-file", "-o", type=str, required=True)

    parsed = parser.parse_args()

    return Args(
        parsed.road_net_file,
        parsed.n_flows_min,
        parsed.n_flows_max,
        parsed.interval_min,
        parsed.interval_max,
        parsed.n_epochs,
        parsed.epoch_length,
        parsed.out_file
    )

FLOWS = [
    5,
    10,
    15,
    # 20,
    # 30,
    # 50,
    # 75,
    # 100,
]


def main():
    args = get_args()

    graph = RoadnetGraph(args.road_net_file)

    incoming_edge_roads = graph.incoming_edge_roads()
    outgoing_edge_roads = graph.outgoing_edge_roads()

    flows = []

    logs = []

    for i_epoch in range(args.n_epochs):
        logs.append({})

        routes = set()

        # n_flows = random.randint(args.n_flows_min, args.n_flows_max)
        # logs[-1]["n_flows"] = n_flows

        n_flows = FLOWS[i_epoch % len(FLOWS)]

        while len(routes) < n_flows:
            start_id = random.choice(incoming_edge_roads).id
            end_id = random.choice(outgoing_edge_roads).id
            routes.add((start_id, end_id))

        start_time = i_epoch * args.epoch_length
        end_time = start_time + args.epoch_length

        interval = random.uniform(args.interval_min, args.interval_max)


        for route in routes:
            flows.append({
                "vehicle": VEHICLE,
                "route": list(route),
                "interval": interval,
                "startTime": start_time + random.random() * interval,
                "endTime": end_time,
            })

    store_json(flows, args.out_file)

    print(logs)

if __name__ == '__main__':
    main()

