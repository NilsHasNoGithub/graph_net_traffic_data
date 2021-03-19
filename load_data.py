import torch
from torch.utils.data import Dataset
from typing import List, Dict, Iterator, AnyStr
from generate_graph import IntersectionGraph
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
                result.append([float(counts_dict[road_id]) for road_id in intersection.lanes])
            yield result

    def get_feature_vecs(self, t: int) -> List[List[float]]:

        result = []

        for intersection in self._graph.intersection_list():
            counts = [float(self._data[t][lane_id]) for lane_id in intersection.lanes]
            result.append(counts)

        return result

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> torch.Tensor:
        feature_vecs = self.get_feature_vecs(idx)
        return torch.tensor(feature_vecs, dtype=torch.float32)
