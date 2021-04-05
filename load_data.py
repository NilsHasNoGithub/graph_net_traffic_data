import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Iterator, AnyStr, Any
from roadnet_graph import RoadnetGraph, Intersection
from utils import load_json

def chunks(l, n) -> List[List]:
    return [l[i:i+n] for i in range(0, len(l), n)]

def flatten(l: List[List]) -> List:
    return [item for sl in l for item in sl]

class LaneVehicleCountDataset(Dataset):

    @staticmethod
    def train_test_from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, shuffle=True, shuffle_chunk_size=10):
        return (
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=True, shuffle=shuffle, shuffle_chunk_size=shuffle_chunk_size),
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=False, shuffle=shuffle, shuffle_chunk_size=shuffle_chunk_size)
        )

    @staticmethod
    def from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, train=True, shuffle=True, shuffle_chunk_size=10) -> "LaneVehicleCountDataset":
        data = load_json(lane_data_file)
        graph = RoadnetGraph(roadnet_file)

        return LaneVehicleCountDataset(graph, data, train=train, shuffle=shuffle, shuffle_chunk_size=shuffle_chunk_size)

    @staticmethod
    def _data_pre_process(graph: RoadnetGraph, data: List[Dict]) -> List[Dict[str, Dict[str, float]]]:
        intersections = graph.intersection_list()

        result = []

        for data_t in data:

            # Initialize all vh counts with 0
            new_data_t = {}
            for intersection in intersections:
                new_data_t[intersection.id] = {}
                for lane_id in intersection.incoming_lanes + intersection.outgoing_lanes:
                    new_data_t[intersection.id][lane_id] = 0.0

            lane_car_infos: Dict[str, Dict] = data_t["laneVehicleInfos"]

            # For each car, increment lane count of the closest intersection
            for lane_id, car_infos in lane_car_infos.items():
                for car_info in car_infos:
                    closest_intersection = car_info["closestIntersection"]

                    # Edge intersections are not included in graph
                    try:
                        new_data_t[closest_intersection][lane_id] += 1.0
                    except KeyError:
                        pass

            result.append(new_data_t)

        return result


    def __init__(self, graph: RoadnetGraph, data: List[Dict[str, int]], train=True, shuffle=True, shuffle_chunk_size=10):
        assert len(data) > 5, "data should contain at least 5 elements"
        i_split = int(0.8*len(data))

        if shuffle:
            data = chunks(data, shuffle_chunk_size)
            random.shuffle(data)
            data = flatten(data)

        data = data[:i_split] if train else data[i_split:]
        self._data = LaneVehicleCountDataset._data_pre_process(graph, data)
        self._graph = graph

    def graph(self) -> RoadnetGraph:
        return self._graph

    def sample_shape(self) -> torch.Size:
        return self[0].shape

    def graph_adjacency_list(self) -> List[List[int]]:
        return self._graph.idx_adjacency_lists()

    def feature_vecs_iter(self) -> Iterator[List[List[float]]]:
        for data_t in self._data:
            result = []
            for intersection in self._graph.intersection_list():
                counts_incoming = [data_t[intersection.id][lane_id] for lane_id in intersection.incoming_lanes]
                counts_outgoing = [data_t[intersection.id][lane_id] for lane_id in intersection.outgoing_lanes]

                result.append(counts_incoming + counts_outgoing)
            yield result

    def get_feature_vecs(self, t: int) -> List[List[float]]:

        result = []

        for intersection in self._graph.intersection_list():

            counts_incoming = [self._data[t][intersection.id][lane_id] for lane_id in intersection.incoming_lanes]
            counts_outgoing = [self._data[t][intersection.id][lane_id] for lane_id in intersection.outgoing_lanes]

            result.append(counts_incoming + counts_outgoing)

        return result

    def get_feature_dict(self, t: int) -> Dict[str, Dict[str, float]]:

        counts = {lane: 0.0 for lane in self._graph.lanes_iter()}

        for intersection in self._graph.intersection_list():
            for lane in intersection.incoming_lanes + intersection.outgoing_lanes:
                counts[lane] += self._data[t][intersection.id][lane]

        return counts

    def extract_vehicles_per_lane(self, t: Tensor) -> Dict[str, float]:
        """

        :param t: Tensor should be of shape [n_agents, n_features]
        :return: Map from lane id to the vehicles on each intersection
        """

        n_intersections, n_features = t.size()
        intersections = self._graph.intersection_list()

        result = {lane: 0.0 for lane in self._graph.lanes_iter()}

        for i in range(n_intersections):
            feats = t[i, :]
            intersection: Intersection = intersections[i]
            for j, lane_id in enumerate(intersection.incoming_lanes + intersection.outgoing_lanes):
                result[lane_id] += feats[j].item()

        return result


    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> torch.Tensor:
        feature_vecs = self.get_feature_vecs(idx)
        return torch.tensor(feature_vecs, dtype=torch.float32)


if __name__ == "__main__":
    roadnet_file = "sample-code/data/manhattan_16x3/roadnet_16_3.json"
    data_file = "generated_data/manhattan_16_3_data.json"
    data_train, data_val = LaneVehicleCountDataset.train_test_from_files(roadnet_file, data_file)

    t = 600
    a = data_train[t]

    feat_dict_original = data_train.get_feature_dict(t)
    feat_dict_processed = data_train.extract_vehicles_per_lane(a)
    assert feat_dict_processed == feat_dict_original


