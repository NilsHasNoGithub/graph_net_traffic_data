from dataclasses import dataclass
from typing import List, AnyStr, Dict, Set, Optional, Iterator
import cityflow
from utils import load_json

@dataclass
class Point:
    x: float
    y: float

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


@dataclass
class Intersection:
    id: str
    roads: List[Road]
    lanes: List[str]
    pos: Point

    def __eq__(self, other):
        if isinstance(other, Intersection):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)

    def __lt__(self, other):
        return self.id < other.id



class IntersectionGraph:

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

        road_map = IntersectionGraph.get_road_map(roadnet)

        for intersection_dict in roadnet["intersections"]:
            if len(intersection_dict["roadLinks"]) > 0:

                roads = []
                lanes = []

                for road_id in intersection_dict["roads"]:

                    road = road_map[road_id]
                    lane_ids = road.lanes

                    lanes.extend(lane_ids)
                    roads.append(road)

                result.append(Intersection(
                    intersection_dict["id"],
                    roads,
                    lanes,
                    Point(
                        intersection_dict["point"]["x"],
                        intersection_dict["point"]["y"]
                    ),
                ))

        return result

    @staticmethod
    def get_intersection_graph(roadnet_file: AnyStr) -> Dict[Intersection, Set[Intersection]]:
        roadnet: dict = load_json(roadnet_file)

        intersections = IntersectionGraph.create_intersections(roadnet)

        adj_dict: Dict[Intersection, Set[Intersection]] = {i: set() for i in intersections}

        n_intersections = len(intersections)
        assert n_intersections >= 1

        for i0 in range(n_intersections - 1):
            for i1 in range(i0 + 1, n_intersections):

                inter0 = intersections[i0]
                inter1 = intersections[i1]

                if len(set(inter0.lanes).intersection(inter1.lanes)) > 0:
                    adj_dict[inter0].add(inter1)
                    adj_dict[inter1].add(inter0)

        return adj_dict

    def __init__(self, roadnet_file: AnyStr):
        """

        :param roadnet_file:
        """
        self.adj_dict = IntersectionGraph.get_intersection_graph(roadnet_file)

        self._intersection_list: List[Intersection] = []
        self._intersection_to_idx: Dict[Intersection, int] = {}

        for intersection in sorted(self.adj_dict.keys()):
            self._intersection_list.append(intersection)
            self._intersection_to_idx[intersection] = len(self._intersection_list) - 1

    def intersection_list(self):
        """
        Ordered list of intersections
        :return:
        """
        return self._intersection_list

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



if __name__ == '__main__':
    data = load_json("generated_data/manhattan_16_3_data.json")

    graph = IntersectionGraph("sample-code/data/manhattan_16x3/roadnet_16_3.json")
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
