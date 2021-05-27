from dataclasses import dataclass
from typing import List, AnyStr, Dict, Set, Optional, Iterator, Callable
import cityflow
import torch
from collections import defaultdict

from utils import load_json, Point



@dataclass
class Road:
    id: str
    lanes: List[str]
    start_intersection_id: str
    end_intersection_id: str
    start: Point
    end: Point

    def __eq__(self, other):
        if isinstance(other, Road):
            return self.id == other.id
        return False

    def __hash__(self):
        return

    def length(self):
        return self.start.distance(self.end)

    def middle(self):
        return (self.start + self.end) / 2


@dataclass
class Intersection:
    id: str
    incoming_roads: List[Road]
    outgoing_roads: List[Road]
    incoming_lanes: List[str]
    outgoing_lanes: List[str]
    pos: Point

    def __eq__(self, other):
        if isinstance(other, Intersection):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id

@dataclass
class _RoadSort:
    road: Road

    def __lt__(self, other):
        assert isinstance(other, _RoadSort)
        middle_self = self.road.middle()
        middle_other = other.road.middle()

        key_self = middle_self.x + middle_self.y
        key_other = middle_other.x + middle_other.y

        if key_self != key_other:
            return key_self < key_other

        return middle_self.x < middle_other.x

class RoadnetGraph:

    @staticmethod
    def get_road_map(roadnet: dict) -> Dict[str, Road]:
        result = {}

        for road in roadnet["roads"]:
            id = road["id"]
            n_lanes = len(road["lanes"])

            lane_ids = [f"{id}_{i}" for i in range(n_lanes)]
            result[id] = Road(
                id,
                lane_ids,
                road["startIntersection"],
                road["endIntersection"],
                Point(road["points"][0]["x"], road["points"][0]["y"]),
                Point(road["points"][1]["x"], road["points"][1]["y"]),
            )

        return result


    @staticmethod
    def create_intersections(roadnet: dict) -> List[Intersection]:
        result = []

        road_map = RoadnetGraph.get_road_map(roadnet)

        for intersection_dict in roadnet["intersections"]:

            if len(intersection_dict["roadLinks"]) > 0:

                intersect_id = intersection_dict["id"]

                incoming_roads = []
                outgoing_roads = []
                incoming_lanes = []
                outgoing_lanes = []

                roads = (_RoadSort(road_map[road_id]) for road_id in intersection_dict["roads"])
                roads = (r.road for r in sorted(roads))

                for road in roads:

                    lane_ids = road.lanes

                    if road.start_intersection_id == intersect_id:
                        outgoing_lanes.extend(lane_ids)
                        outgoing_roads.append(road)
                    else:
                        assert road.end_intersection_id == intersect_id
                        incoming_lanes.extend(lane_ids)
                        incoming_roads.append(road)

                result.append(Intersection(
                    intersect_id,
                    incoming_roads,
                    outgoing_roads,
                    incoming_lanes,
                    outgoing_lanes,
                    Point(
                        intersection_dict["point"]["x"],
                        intersection_dict["point"]["y"]
                    ),
                ))

        return result

    @staticmethod
    def get_intersection_graph(roadnet: dict) -> Dict[Intersection, Set[Intersection]]:


        intersections = RoadnetGraph.create_intersections(roadnet)

        adj_dict: Dict[Intersection, Set[Intersection]] = {i: set() for i in intersections}

        n_intersections = len(intersections)
        assert n_intersections >= 1

        for i0 in range(n_intersections - 1):
            for i1 in range(i0 + 1, n_intersections):

                inter0 = intersections[i0]
                inter1 = intersections[i1]

                if len(set(inter0.incoming_lanes).intersection(inter1.outgoing_lanes)) > 0:
                    adj_dict[inter0].add(inter1)
                    adj_dict[inter1].add(inter0)

        return adj_dict

    def __init__(self, roadnet_file: AnyStr):
        """

        :param roadnet_file:
        """

        roadnet: dict = load_json(roadnet_file)
        self.adj_dict = RoadnetGraph.get_intersection_graph(roadnet)

        self._road_dict = RoadnetGraph.get_road_map(roadnet)
        self._road_list = list(self._road_dict.values())

        self._intersection_list: List[Intersection] = []
        self._intersection_to_idx: Dict[Intersection, int] = {}
        self._intersection_dict: Dict[str, Intersection] = {}

        self._phases = set()

        for intersection in sorted(self.adj_dict.keys()):

            self._intersection_list.append(intersection)
            self._intersection_to_idx[intersection] = len(self._intersection_list) - 1

            self._intersection_dict[intersection.id] = intersection


    def intersection_list(self) -> List[Intersection]:
        """
        Ordered list of intersections
        :return:
        """
        return self._intersection_list

    def road_list(self) -> List[Road]:
        """
        All roads in the graph
        :return:
        """
        return self._road_list


    def incoming_edge_roads(self) -> List[Road]:
        return [r for r in self._road_list if r.start_intersection_id not in self._intersection_dict.keys()]

    def outgoing_edge_roads(self) -> List[Road]:
        return [r for r in self._road_list if r.end_intersection_id not in self._intersection_dict.keys()]

    def intersection_dict(self) -> Dict[str, Intersection]:
        return self._intersection_dict

    def road_dict(self) -> Dict[str, Road]:
        return self._road_dict

    def lanes_iter(self) -> Iterator[str]:
        for road in self._road_list:
            for lane in road.lanes:
                yield lane

    def road_of_lane(self, lane) -> Road:
        road_id = lane

        while road_id[-1] != "_":
            road_id = road_id[:-1]

        if road_id[-1] != "_":
            raise ValueError()

        road_id = road_id[:-1]
        return self._road_dict[road_id]

    def idx_adjacency_lists(self) -> List[List[int]]:
        """
        Adjacency list with indexes of intersections. Intersections can be
        looked up using `IntersectionGraph.intersection_list`
        :return:
        """
        result = []

        for intersection in self._intersection_list:
            nbs = []

            for nb in self.adj_dict[intersection]:
                nbs.append(self._intersection_to_idx[nb])

            result.append(nbs)

        return result

    def n_intersection_phases(self) -> int:
        return 9

    def tensor_data_from_time_step_data(self, data_t: dict,
                                        hidden_intersections: Optional[Set[str]]=None) -> torch.Tensor:
        """
        Converts a result from gather_step_data to a tensor which can be used as input for the model

        :param data_t: dict with keys ["laneCounts", "laneVehicleInfos", "intersectionPhases"]
        :param hidden_intersections:
        :return: tensor which can be used as input for this network
        """
        lane_vh_infos = data_t["laneVehicleInfos"]
        intersection_phases = data_t["intersectionPhases"]

        result = []

        debug_counts = defaultdict(int)

        for intersection in self.intersection_list():
            is_observed = hidden_intersections is None or (intersection.id not in hidden_intersections)
            lane_counts = []

            for lane in intersection.incoming_lanes + intersection.outgoing_lanes:

                count = 0.0
                if is_observed:
                    for car_info in lane_vh_infos[lane]:
                        if car_info["closestIntersection"] == intersection.id:
                            count += 1.0
                            debug_counts[lane] += 1.0

                lane_counts.append(count)

            hidden_feat = 0.0 if is_observed else 1.0

            phase_one_hot = [0.0] * self.n_intersection_phases()
            # self._phases.add(int(intersection_phases[intersection.id]))
            phase_one_hot[int(intersection_phases[intersection.id])] = 1.0

            result.append([hidden_feat] + phase_one_hot + lane_counts)

        return torch.Tensor(result)

    def lane_feats_per_intersection_from_tensor(self, tensor: torch.Tensor) -> Dict[str, Dict[str, float]]:
        result = {}

        for i_intersection, intersection in enumerate(self.intersection_list()):
            intersection_data = {}
            for i_lane, lane in enumerate(intersection.incoming_lanes + intersection.outgoing_lanes):
                intersection_data[lane] = tensor[i_intersection, i_lane].item()

            result[intersection.id] = intersection_data

        return result

    def lane_feats_from_tensor(self, tensor: torch.Tensor, agg: Optional[Callable[[float, float], float]]=None) -> Dict[str, float]:
        """
        Combines features from the tensor into a single feature per lane

        :param tensor:
        :param agg: How features from the lanes are combined, default takes the sum
        :return:
        """

        if agg is None:
            agg = lambda a, b: a + b

        feats_per_intersection = self.lane_feats_per_intersection_from_tensor(tensor)

        result: Dict[str, float] = {}

        for lane_feats in feats_per_intersection.values():
            for lane_id, feat in lane_feats.items():
                if lane_id not in result.keys():
                    result[lane_id] = feat
                else:
                    result[lane_id] = agg(feat, result[lane_id])

        return result







if __name__ == '__main__':
    data = load_json("generated_data/manhattan_16_3_data.json")

    graph = RoadnetGraph("sample-code/data/manhattan_16x3/roadnet_16_3.json")
    adj_dict = graph.adj_dict

    i=0
    print(len(adj_dict))
    for k, vs in adj_dict.items():
        print(f"{k.id}: {[v.id for v in vs]}")

    # for v in graph.feature_vecs_iter():
    #     print(v)

    for l in graph.idx_adjacency_lists():
        print(l)

    print(i)


