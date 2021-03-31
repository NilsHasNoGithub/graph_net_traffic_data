import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import List, Dict, Iterator, AnyStr
from generate_graph import IntersectionGraph, Intersection
from utils import load_json

class LaneVehicleCountDataset(Dataset):

    @staticmethod
    def train_test_from_files(roadnet_file: AnyStr, lane_data_file: AnyStr):
        return (
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=True),
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=False)
        )

    @staticmethod
    def from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, train=True) -> "LaneVehicleCountDataset":
        data = load_json(lane_data_file)
        graph = IntersectionGraph(roadnet_file)

        return LaneVehicleCountDataset(graph, data, train=train)

    def __init__(self, graph: IntersectionGraph, data: List[Dict[str, int]], train=True):
        assert len(data) > 5, "data should contain at least 5 elements"
        i_split = int(0.8*len(data))

        self._data = data[:i_split] if train else data[i_split:]
        self._graph = graph

    def sample_shape(self) -> torch.Size:
        return self[0].shape

    def graph_adjacency_list(self) -> List[List[int]]:
        return self._graph.idx_adjacency_lists()

    def feature_vecs_iter(self) -> Iterator[List[List[float]]]:
        for counts_dict in self._data:
            result = []
            for intersection in self._graph.intersection_list():
                result.append([float(counts_dict[road_id]) for road_id in intersection.incoming_lanes])
            yield result

    def get_feature_vecs(self, t: int) -> List[List[float]]:

        result = []

        for intersection in self._graph.intersection_list():
            counts = [float(self._data[t][lane_id]) for lane_id in intersection.incoming_lanes]
            result.append(counts)

        return result

    def get_feature_dict(self, t: int) -> Dict[str, Dict[str, float]]:
        result = {}

        for intersection in self._graph.intersection_list():
            counts = {lane_id: float(self._data[t][lane_id]) for lane_id in intersection.incoming_lanes}
            result[intersection.id] = counts

        return result

    def extract_vehicles_per_lane(self, t: Tensor) -> Dict[str, float]:
        """

        :param t: Tensor should be of shape [n_agents, n_features]
        :return: Map from lane id to the vehicles on each intersection
        """

        n_intersections, n_features = t.size()
        intersections = self._graph.intersection_list()

        result = {}

        for i in range(n_intersections):
            feats = t[i, :]
            intersection: Intersection = intersections[i]
            for j, lane_id in enumerate(intersection.incoming_lanes):
                result[lane_id] = feats[j].item()

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

    print(data_train.extract_vehicles_per_lane(a))
    print(data_train.get_feature_dict(t))
