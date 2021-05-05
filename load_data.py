import random
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.distributions.distribution import  Distribution
from typing import List, Dict, Iterator, AnyStr, Any, Tuple, Set, Optional, Union
from roadnet_graph import RoadnetGraph, Intersection
from utils import load_json

def chunks(l, n) -> List[List]:
    return [l[i:i+n] for i in range(0, len(l), n)]

def flatten(l: List[List]) -> List:
    return [item for sl in l for item in sl]

class LaneVehicleCountDataset(Dataset):

    @staticmethod
    def train_test_from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, **kwargs):
        return (
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=True, **kwargs),
            LaneVehicleCountDataset.from_files(roadnet_file, lane_data_file, train=False, **kwargs)
        )

    @staticmethod
    def from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, **kwargs) -> "LaneVehicleCountDataset":
        data = load_json(lane_data_file)
        graph = RoadnetGraph(roadnet_file)

        return LaneVehicleCountDataset(graph, data, **kwargs)

    @staticmethod
    def _data_pre_process(graph: RoadnetGraph, data: List[Dict], scale_by_road_len: bool) -> List[Dict[str, Dict[str, Dict[str, float]]]]:
        intersections = graph.intersection_list()

        result = []

        for data_t in data:

            # Initialize all vh counts with 0
            new_data_t = {}
            for intersection in intersections:
                i_lane_data = {}
                for lane_id in intersection.incoming_lanes + intersection.outgoing_lanes:
                    i_lane_data[lane_id] = 0.0

                new_data_t[intersection.id] = {}
                new_data_t[intersection.id]["laneVehicleInfos"] = i_lane_data


            lane_car_infos: Dict[str, Dict] = data_t["laneVehicleInfos"]
            phase_infos: Dict[str, int] = data_t["intersectionPhases"]

            for k, v in phase_infos.items():
                new_data_t[k]["phase"] = v

            # For each car, increment lane count of the closest intersection
            for lane_id, car_infos in lane_car_infos.items():
                for car_info in car_infos:
                    closest_intersection = car_info["closestIntersection"]

                    # Edge intersections are not included in graph
                    try:
                        new_data_t[closest_intersection]["laneVehicleInfos"][lane_id] += 1.0
                    except KeyError:
                        pass

            result.append(new_data_t)

        if scale_by_road_len:
            for data_t in result:
                for intersection in intersections:
                    for road in intersection.incoming_roads + intersection.outgoing_roads:
                        for lane_id in road.lanes:
                            try:
                                data_t[intersection.id]["laneVehicleInfos"][lane_id] /= road.length() / 2
                            except KeyError:
                                pass

        return result


    def __init__(self, graph: RoadnetGraph, data: List[Dict[str, int]], train=True, shuffle=True, shuffle_chunk_size=1, scale_by_road_len=False):
        # assert len(data) > 5, "data should contain at least 5 elements"
        i_split = int(0.8*len(data))

        data = data[:i_split] if train else data[i_split:]

        if shuffle:
            data = chunks(data, shuffle_chunk_size)
            random.shuffle(data)
            data = flatten(data)

        self._data = LaneVehicleCountDataset._data_pre_process(graph, data, scale_by_road_len)
        self._graph = graph

    def graph(self) -> RoadnetGraph:
        return self._graph

    def input_shape(self) -> torch.Size:
        return self[0].shape

    def output_shape(self) -> torch.Size:
        return self[0].shape

    def graph_adjacency_list(self) -> List[List[int]]:
        return self._graph.idx_adjacency_lists()

    def feature_vecs_iter(self) -> Iterator[List[List[float]]]:
        for data_t in self._data:
            result = []
            for intersection in self._graph.intersection_list():
                counts_incoming = [data_t[intersection.id]["laneVehicleCounts"][lane_id] for lane_id in intersection.incoming_lanes]
                counts_outgoing = [data_t[intersection.id]["laneVehicleCounts"][lane_id] for lane_id in intersection.outgoing_lanes]

                result.append(counts_incoming + counts_outgoing)
            yield result

    def get_feature_vecs(self, t: int) -> List[List[float]]:

        result = []

        for intersection in self._graph.intersection_list():
            intersection_data = self._data[t][intersection.id]["laneVehicleInfos"]

            counts = [intersection_data[lane_id] for lane_id in intersection.incoming_lanes + intersection.outgoing_lanes]

            result.append(counts)

        return result

    def get_feature_dict(self, t: int) -> Dict[str, Dict[str, float]]:

        counts = {lane: 0.0 for lane in self._graph.lanes_iter()}

        for intersection in self._graph.intersection_list():
            for lane in intersection.incoming_lanes + intersection.outgoing_lanes:
                counts[lane] += self._data[t][intersection.id]["laneVehicleCounts"][lane]

        return counts

    def extract_data_per_lane_per_intersection(self, t:Tensor) -> Dict[Tuple[str, str], float]:
        n_intersections, n_features = t.size()
        intersections = self._graph.intersection_list()

        assert n_intersections == len(intersections)

        result = {}

        for i, intersection in enumerate(intersections):
            feats = t[i, :]
            for j, lane_id in enumerate(intersection.incoming_lanes + intersection.outgoing_lanes):
                result[(intersection.id, lane_id)] = feats[j].item()

        return result

    def extract_data_per_lane(self, t: Tensor) -> Dict[str, float]:
        """

        :param t: Tensor should be of shape [n_agents, n_features]
        :return: Map from lane id to the vehicles on each intersection
        """

        n_intersections, n_features = t.size()
        intersections = self._graph.intersection_list()

        result = {lane: 0.0 for lane in self._graph.lanes_iter()}

        feats = self.extract_data_per_lane_per_intersection(t)

        for ((_, lane), v) in feats:
            result[lane] += v

        return result


    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx) -> torch.Tensor:
        feature_vecs = self.get_feature_vecs(idx)
        return torch.tensor(feature_vecs, dtype=torch.float32)

@dataclass
class TimeStepDataMissing:
    is_missing: bool
    phase: int
    data: Dict

class LaneVehicleCountDatasetMissing(LaneVehicleCountDataset):

    @staticmethod
    def train_test_from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, **kwargs):
        return (
            LaneVehicleCountDatasetMissing.from_files(roadnet_file, lane_data_file, train=True, **kwargs),
            LaneVehicleCountDatasetMissing.from_files(roadnet_file, lane_data_file, train=False, **kwargs)
        )

    @staticmethod
    def from_files(roadnet_file: AnyStr, lane_data_file: AnyStr, **kwargs) -> "LaneVehicleCountDataset":
        data = load_json(lane_data_file)
        graph = RoadnetGraph(roadnet_file)

        return LaneVehicleCountDatasetMissing(graph, data, **kwargs)

    @staticmethod
    def _generate_missing_sensor_data(data_t: dict, graph: RoadnetGraph, p_missing: float) -> Dict:
        intersections = graph.intersection_list()

        new_data_t = {}
        for intersection in intersections:
            data_t_i = data_t[intersection.id]["laneVehicleInfos"]
            is_missing = random.random() < p_missing

            if is_missing: #TODO insert intersection.id that's right or top
                intersection_data = {lane_id:0.0 for lane_id in data_t_i.keys()}
            else:
                intersection_data = data_t_i

            phase = data_t[intersection.id]["phase"]

            new_data_t[intersection.id] = TimeStepDataMissing(is_missing, phase, intersection_data)

        return new_data_t



    def input_shape(self) -> torch.Size:
        return self[0][0].shape

    def output_shape(self) -> torch.Size:
        return self[0][1].shape

    def __init__(self, graph: RoadnetGraph, data: List[Dict[str, int]], train=True, shuffle=True, shuffle_chunk_size=1, p_missing: Optional[Union[Distribution, float]]=None, scale_by_road_len=False):
        LaneVehicleCountDataset.__init__(self, graph, data, train=train, shuffle=shuffle, shuffle_chunk_size=shuffle_chunk_size, scale_by_road_len=scale_by_road_len)

        if p_missing is None:
            p_missing = 0.2

        self._p_missing = p_missing


    def get_feature_vecs_hidden(self, t: int, return_hidden_intersections=False) -> Any:
        inputs = []

        if issubclass(type(self._p_missing), Distribution):
            p_missing = self._p_missing.sample(sample_shape=[1]).item()
        else:
            p_missing = self._p_missing

        assert isinstance(p_missing, float)

        data_t = LaneVehicleCountDatasetMissing._generate_missing_sensor_data(self._data[t], self._graph, p_missing)

        for intersection in self._graph.intersection_list():
            intersection_data = data_t[intersection.id]


            counts = [intersection_data.data[lane_id] for lane_id in
                      intersection.incoming_lanes + intersection.outgoing_lanes]

            phase_one_hot = [0.0] * 5

            phase_one_hot[int(intersection_data.phase)] = 1.0

            inputs.append([1.0 if intersection_data.is_missing else 0.0] + phase_one_hot + counts)

        if return_hidden_intersections:
            hidden_intersections = {i_id for (i_id, i_data) in data_t.items() if i_data.is_missing}
            return inputs, self.get_feature_vecs(t), hidden_intersections

        #TODO return hidden intersections
        return inputs, self.get_feature_vecs(t)

    # def get_no_data_intersections(self, t: int) -> Set[str]:
    #     data_t = self._data_hidden[t]
    #     result = {i_id for (i_id, i_data) in data_t.items() if i_data.is_missing}

    #     return result

    def __len__(self):
        return len(self._data)

    def get_item(self, item, return_hidden_intersections=False):
        result = list(self.get_feature_vecs_hidden(item, return_hidden_intersections=return_hidden_intersections))

        for i in range(2):
            result[i] = torch.tensor(result[i])

        return tuple(result)

    def __getitem__(self, item):

        return self.get_item(item)


class RandData(LaneVehicleCountDatasetMissing):

    def __init__(self, road_net_file, p_missing=0.5, size=10_000):
        graph = RoadnetGraph(road_net_file)

        data = []

        for _ in range(size):



            data_t = {}
            for intersection in graph.intersection_list():

                data_t[intersection.id] = {}
                data_t[intersection.id]["laneVehicleInfos"] = {}
                data_t[intersection.id]["phase"] = 0

                for lane_id in intersection.incoming_lanes + intersection.outgoing_lanes:
                    data_t[intersection.id]["laneVehicleInfos"][lane_id] = float(random.randint(0, 29))

            data.append(data_t)

        LaneVehicleCountDatasetMissing.__init__(self, graph, [], p_missing=p_missing)
        self._data = data


if __name__ == "__main__":
    roadnet_file = "sample-code/data/manhattan_16x3/roadnet_16_3.json"
    data_file = "generated_data/manhattan_16_3_data.json"
    data_train, data_val = LaneVehicleCountDataset.train_test_from_files(roadnet_file, data_file)
    data_train_m, data_val_m = LaneVehicleCountDatasetMissing.train_test_from_files(roadnet_file, data_file)

    t = 600
    a = data_train[t]

    feat_dict_original = data_train.get_feature_dict(t)
    feat_dict_processed = data_train.extract_data_per_lane(a)
    assert feat_dict_processed == feat_dict_original

    a,b = data_train_m[t]

    for i in range(a.shape[0]):
        if a[i, 0] == 0:
            assert torch.all(a[i,1:] == b[i,:])


