import json
from typing import AnyStr, Union
import torch
from dataclasses import dataclass
from math import sqrt
from multiprocessing import cpu_count

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cpu")
# torch.set_num_threads(cpu_count())

def load_json(fp: AnyStr) -> Union[dict, list]:
    with open(fp, "r") as f:
        return json.load(f)

def store_json(o: Union[dict, list], fp: AnyStr):
    with open(fp, "w") as f:
        json.dump(o, f)

@dataclass
class Point:
    x: Union[float, int]
    y: Union[float, int]

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y

        raise ValueError("index must be 1 or 0")

    def __add__(self, other):
        if not isinstance(other, Point):
            raise ValueError()

        return Point(
            self.x + other.x,
            self.y + other.y
        )

    def __sub__(self, other):
        if not isinstance(other, Point):
            raise ValueError()

        return Point(
            self.x - other.x,
            self.y - other.y
        )

    def __iter__(self):
        return iter((self.x, self.y))

    def distance(self, other):

        if not isinstance(other, Point):
            raise ValueError()

        dx = other.x - self.x
        dy = other.y - self.y

        dist = sqrt(dx ** 2 + dy ** 2)
        return dist
