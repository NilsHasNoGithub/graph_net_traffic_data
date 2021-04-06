from dataclasses import dataclass
from typing import List, AnyStr, Dict, Set, Optional, Iterator
import cityflow
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

    def intersection_dict(self) -> Dict[str, Intersection]:
        return self._intersection_dict

    def road_dict(self) -> Dict[str, Road]:
        return self._road_dict

    def lanes_iter(self) -> Iterator[str]:
        for road in self._road_list:
            for lane in road.lanes:
                yield lane


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


