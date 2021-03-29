import math

import cairo

from generate_graph import IntersectionGraph, Point
from utils import load_json

SIZE = 10


def line(ctx: cairo.Context, pos1, pos2, offset):
    x_offset, y_offset = 0, 0

    # Infer if the x changes of the roads, then we should offset with y.
    if pos1.x == pos2.x:
        x_offset = offset
    else:
        # If the y changes then we should offset with x.
        y_offset = offset

    ctx.move_to(pos1.x + x_offset, pos1.y + y_offset)
    ctx.line_to(pos2.x + x_offset, pos2.y + y_offset)
    ctx.stroke()

    print("[", pos1.x + x_offset, pos1.y + y_offset, "]", "[", pos2.x + x_offset, pos2.y + y_offset, "]")


def square(ctx):
    ctx.move_to(0, 0)
    ctx.rel_line_to(2 * SIZE, 0)
    ctx.rel_line_to(0, 2 * SIZE)
    ctx.rel_line_to(-2 * SIZE, 0)
    ctx.close_path()


def draw_shapes(ctx: cairo.Context, x, y, fill):
    ctx.save()

    ctx.new_path()
    ctx.translate(x, y)
    square(ctx)
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

    for intersection in graph.intersection_list():
        position = intersection.pos
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        fill_shapes(ctx, position.x, position.y)

        for index, road in enumerate(intersection.roads):
            index += 0.5
            line(ctx, road.start, road.end, index * 2)

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
    WIDTH, HEIGHT = 2000, 3000
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)

    draw(ctx, graph)

    surface.write_to_png("test.png")


if __name__ == '__main__':
    main()
