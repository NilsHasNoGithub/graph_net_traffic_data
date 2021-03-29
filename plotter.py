import math
from copy import deepcopy

import cairo

from generate_graph import IntersectionGraph, Point
from utils import load_json

SIZE = 12


def line(ctx: cairo.Context, pos1, pos2, offset):
    x_offset, y_offset = 0, 0

    # Infer if the x changes of the roads, then we should offset with y.
    if pos1.x == pos2.x:
        x_offset = offset
    else:
        # If the y changes then we should offset with x.
        y_offset = offset

    ctx.move_to(pos1.x + x_offset + 400, pos1.y + 100 + y_offset)
    ctx.line_to(pos2.x + x_offset + 400, pos2.y + 100 + y_offset)
    ctx.stroke()

    print("[", pos1.x + x_offset, pos1.y + y_offset, "]", "[", pos2.x + x_offset, pos2.y + y_offset, "]")


def square(ctx):
    ctx.move_to(0, 0)
    ctx.rel_line_to(2 * SIZE, 0)
    ctx.rel_line_to(0, 2 * SIZE)
    ctx.rel_line_to(-2 * SIZE, 0)

def draw_shapes(ctx: cairo.Context, x, y, fill):
    ctx.save()

    ctx.new_path()
    ctx.translate(x, y)
    square(ctx)
    ctx.close_path()

    if fill:
        ctx.fill()
    else:
        ctx.stroke()

    ctx.restore()


def fill_shapes(ctx, x, y):
    draw_shapes(ctx, x, y, True)


def stroke_shapes(ctx, x, y):
    draw_shapes(ctx, x, y, False)


def draw(ctx: cairo.Context, graph: IntersectionGraph):
    ctx.set_source_rgb(0, 0, 0)

    ctx.set_line_width(0.25)
    ctx.set_tolerance(0.1)

    placed_lane_set = set()

    adj_dict = deepcopy(graph.adj_dict)
    adj_dict_drawn_road_count = {}

    for key, items in adj_dict.items():
        adj_dict_drawn_road_count[key.id] = {}
        for neighbor in items:
            adj_dict_drawn_road_count[key.id][neighbor.id] = 0

    for intersection in graph.intersection_list():
        all_roads = intersection.incoming_roads + intersection.outgoing_roads

        for index, road in enumerate(all_roads):
            if index >= len(intersection.incoming_roads):
                ctx.set_source_rgb(255, 0, 0)
            else:
                ctx.set_source_rgb(0, 0, 0)

            if road.start_intersection_id not in adj_dict_drawn_road_count:
                adj_dict_drawn_road_count[road.start_intersection_id] = {}
                adj_dict_drawn_road_count[road.start_intersection_id][road.end_intersection_id] = 0
                adj_dict_drawn_road_count[road.end_intersection_id][road.start_intersection_id] = 0

            for lane in road.lanes:
                if lane not in placed_lane_set:
                    offset = adj_dict_drawn_road_count[road.start_intersection_id][road.end_intersection_id]
                    line(ctx, road.start, road.end, offset * 3)
                    adj_dict_drawn_road_count[road.start_intersection_id][road.end_intersection_id] += 1
                    adj_dict_drawn_road_count[road.end_intersection_id][road.start_intersection_id] += 1
                    placed_lane_set.add(lane)

        position = intersection.pos
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgb(0, 0, 0)
        fill_shapes(ctx, position.x - 5 + 400, position.y + 100 - 5)
        """
        offset = 0
        index = 0
        all_roads = intersection.incoming_roads + intersection.outgoing_roads
        
        while index < len(all_roads):
            road = all_roads[index]

            if index >= len(intersection.incoming_roads):
                ctx.set_source_rgb(255, 0, 0)
            else:
                ctx.set_source_rgb(0, 0, 0)

            if road.id not in placed_road_id_set:
                offset += 1
                print(offset)
                line(ctx, road.start, road.end, offset * 2)
                placed_road_id_set.add(road.id)

            index += 1
        """

    # line(ctx, Point(0, 0), Point(2000,2000), 0)
    # line(ctx, Point(2000, 0), Point(0, 2000), 0)

    """
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_dash([SIZE / 4.0, SIZE / 4.0], 0)
    stroke_shapes(ctx, 0, 0)

    ctx.set_dash([], 0)
    stroke_shapes(ctx, 0, 3 * SIZE)

    ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
    stroke_shapes(ctx, 0, 6 * SIZE)

    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    stroke_shapes(ctx, 0, 9 * SIZE)

    fill_shapes(ctx, 0, 12 * SIZE)

    ctx.set_line_join(cairo.LINE_JOIN_BEVEL)
    fill_shapes(ctx, 0, 15 * SIZE)
    ctx.set_source_rgb(1, 0, 0)
    stroke_shapes(ctx, 0, 15 * SIZE)
    """


def main():
    data = load_json("generated_data/manhattan_16_3_data.json")

    graph = IntersectionGraph("sample-code/data/manhattan_16x3/roadnet_16_3.json")
    adj_dict = graph.adj_dict

    n_intersections = 48
    n_intersections_per_row = 3
    n_intersections_per_height = math.ceil(n_intersections / n_intersections_per_row)
    WIDTH, HEIGHT = 1550, 1750
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    draw(ctx, graph)

    surface.write_to_png("test.png")


if __name__ == '__main__':
    main()
