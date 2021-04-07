import math
from copy import deepcopy
from typing import Dict, Iterable, Tuple, Set, Optional

import cairo
from cairo import Context as CContext
from torch import Tensor

from roadnet_graph import RoadnetGraph, Point, Road, Intersection
from utils import load_json

from load_data import LaneVehicleCountDataset
from math import sqrt, asin, pi, atan

class _RoadPlotter:

    @staticmethod
    def _get_road_extreme_vals(roads: Iterable[Road]) -> Tuple[Point, Point]:
        """
        :param roads:
        :return: a point with min_x and min_y and one with max_x and max_y
        """
        road_min_x = math.inf
        road_min_y = math.inf
        road_max_x = -math.inf
        road_max_y = -math.inf

        for road in roads:
            min_x = min(road.start.x, road.end.x)
            max_x = max(road.start.x, road.end.x)
            min_y = min(road.start.y, road.end.y)
            max_y = max(road.start.y, road.end.y)

            if min_x < road_min_x:
                road_min_x = min_x

            if min_y < road_min_y:
                road_min_y = min_y

            if max_x > road_max_x:
                road_max_x = max_x

            if max_y > road_max_y:
                road_max_y = max_y

        return (
            Point(road_min_x, road_min_y),
            Point(road_max_x, road_max_y)
        )

    @staticmethod
    def _calc_data_max(graph: RoadnetGraph, data: Dict[str, float]):
        max_ = 0

        for intersection in graph.intersection_list():
            for road in intersection.incoming_roads:
                for lane_id in road.lanes:
                    val = data[lane_id] / road.length()
                    if val > max_:
                        max_ = val

        return max_

    def __init__(
            self,
            graph: RoadnetGraph,
            data: Dict[str, float],
            no_data_intersections: Optional[Set[str]] = None,
            intersection_size=55,
            padding=30,
            legend_width=100,
            legend_height=500
    ):
        """

        :param ctx: Cairo context
        :param graph: Intersection graph containing data to draw
        :param data: data to display on edges
        """
        graph = deepcopy(graph)

        self._min_point, self._max_point = _RoadPlotter._get_road_extreme_vals(graph.road_list())

        self._min_point -= Point(padding, padding)
        self._max_point += Point(padding, padding)

        width, height = self._max_point - self._min_point
        width += legend_width + 2 * padding

        min_height = 2 * padding + legend_height
        height = max(height, min_height)

        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self._ctx = cairo.Context(self._surface)
        self._data = data
        self._graph = graph
        self._no_data_intersections = set() if no_data_intersections is None else no_data_intersections
        self._intersection_size = intersection_size
        self._legend_height = legend_height
        self._legend_width = legend_width
        self._padding = padding
        self._width = width
        self._height = height

        self._data_max = _RoadPlotter._calc_data_max(graph, data)

        self._drawn_road_ids = set()

    def _draw_intersections_and_lanes(self):
        intersections = self._graph.intersection_list()

        for intersection in intersections:
            self._draw_intersection(intersection)

        for intersection in intersections:
            for road in intersection.outgoing_roads:
                if road.id not in self._drawn_road_ids:
                    self._draw_lanes(road, False)

        for intersection in intersections:
            pos = intersection.pos - self._min_point
            pos.y = self._height - pos.y
            if intersection.id in self._no_data_intersections:
                self._draw_circle(pos, self._intersection_size, fill=(1,1,1))
            else:
                self._draw_circle(pos, self._intersection_size)

    def draw_all(self):
        ctx = self._ctx
        ctx.save()
        ctx.set_source_rgb(1,1,1)
        ctx.paint()
        ctx.restore()

        self._draw_intersections_and_lanes()
        self._draw_legend()

    def get_surface(self) -> cairo.Surface:
        return self._surface

    def _calc_rgb_from_business(self, business: float):
        assert 0.0 <= business <= 1.0
        return business, 0.5 - 0.5 * business, 0.0

    def _calc_lane_rgb_val(self, lane_id: str, road: Road) -> (float, float, float):
        vh_count = self._data[lane_id]
        business = max(0.0, (vh_count / road.length()) / self._data_max)

        return self._calc_rgb_from_business(business)

    def _draw_intersection(self, intersection: Intersection):
        for road in intersection.incoming_roads:
            self._drawn_road_ids.add(road.id)
            self._draw_lanes(road, True)

    def _draw_legend(self, txt_size=30, txt_spacing=20):
        ctx = self._ctx

        ctx.save()

        legend_pos = Point(
            self._width - self._padding - self._legend_width,
            self._padding + txt_spacing
        )

        for i in range(self._legend_height - int(1.5 * txt_spacing)):
            business = 1 - (i / (self._legend_height - 1))
            ctx.set_source_rgb(*self._calc_rgb_from_business(business))
            ctx.rectangle(*(legend_pos + Point(0, i)), self._legend_width, 1)
            ctx.fill()

        ctx.set_source_rgb(0, 0, 0)
        ctx.set_font_size(txt_size)
        ctx.move_to(*(legend_pos + Point(0, -2)))
        ctx.show_text("{:.2f}".format(self._data_max))

        ctx.move_to(*(legend_pos + Point(0, self._legend_height)))
        ctx.show_text("0")
        ctx.restore()

    def _draw_lanes(self, road: Road, is_incoming: bool, offset=5.5):
        start = deepcopy(road.start)
        end = deepcopy(road.end)

        start -= self._min_point
        start.y = self._height - start.y

        end -= self._min_point
        end.y = self._height - end.y

        rel_end = end - start

        road_len = start.distance(end)

        angle = -atan((rel_end.x / rel_end.y) if rel_end.y != 0 else math.inf)

        ctx = self._ctx

        ctx.save()

        if rel_end.y < 0 or (rel_end.x < 0 and rel_end.y == 0):
            angle += pi

        offset *= -1

        for i, lane_id in enumerate(sorted(road.lanes)):
            ctx.move_to(start.x, start.y)
            ctx.rotate(angle)
            if is_incoming:
                ctx.set_source_rgb(*self._calc_lane_rgb_val(lane_id, road))
            else:
                ctx.set_source_rgb(0,0,0)
            # ctx.set_source_rgb(i*(1/2),0,0)
            self._draw_line_rel(Point((i + 1) * offset, 0), Point(0, road_len))

            ctx.rotate(-angle)
        ctx.restore()

    def _draw_line_rel(self, pos1: Point, pos2: Point, width=5):
        ctx = self._ctx
        ctx.set_line_width(width)
        ctx.rel_move_to(pos1.x, pos1.y)
        ctx.rel_line_to(pos2.x, pos2.y)
        ctx.stroke()

    def _draw_circle(self, pos: Point, size: float, fill=(0,0,0), stroke=(0,0,0)):
        ctx = self._ctx
        ctx.save()
        x, y = pos.x, pos.y

        ctx.arc(x, y, size / 2, 0, 2 * pi)
        ctx.set_source_rgb(*fill)
        ctx.fill()
        ctx.set_source_rgb(*stroke)
        ctx.stroke()
        ctx.restore()

def gen_data_visualization(dataset: LaneVehicleCountDataset, data_tensor: Tensor, no_data_intersections: Optional[Set[str]] = None) -> cairo.Surface:
    """

    :param dataset:
    :param data_tensor: should be of shape `[n_intersections, n_features]`
    :return:
    """

    drawer = _RoadPlotter(dataset.graph(), dataset.extract_vehicles_per_lane(data_tensor), no_data_intersections=no_data_intersections)
    drawer.draw_all()
    return drawer.get_surface()


def main():
    roadnet_file = "sample-code/data/manhattan_16x3/roadnet_16_3.json"
    data_file = "generated_data/manhattan_16_3_data.json"

    data = load_json(data_file)
    data_set = LaneVehicleCountDataset.from_files(roadnet_file, data_file)

    graph = RoadnetGraph(roadnet_file)
    adj_dict = graph.adj_dict

    n_intersections = 48
    n_intersections_per_row = 3
    n_intersections_per_height = math.ceil(n_intersections / n_intersections_per_row)
    drawer = _RoadPlotter(graph, data_set.extract_vehicles_per_lane(data_set[2567]))

    drawer.draw_all()

    drawer.get_surface().write_to_png("results/test.png")


if __name__ == '__main__':
    main()
