import orjson
import pickle
from typing import AnyStr, Union, Any
import torch
from dataclasses import dataclass
from math import sqrt
from multiprocessing import cpu_count

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device("cpu")
# torch.set_num_threads(cpu_count())

def load_json(fp: AnyStr) -> Union[dict, list]:
    with open(fp, "rb") as f:
        return orjson.loads(f.read())

def store_json(o: Union[dict, list], fp: AnyStr):
    with open(fp, "wb") as f:
        f.write(orjson.dumps(o))

def load_pkl(fp: AnyStr) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)

def store_pkl(item: Any, fp: AnyStr):
    with open(fp, "wb") as f:
        pickle.dump(item, f)

@dataclass
class Point:
    x: Union[float, int]
    y: Union[float, int]

    @staticmethod
    def _parse_type(other) -> 'Point':
        if isinstance(other, Point):
            return other
        elif isinstance(other, float) or isinstance(other, int):
            return Point(other, other)

        raise ValueError()

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y

        raise ValueError("index must be 1 or 0")

    def __add__(self, other):
        other = Point._parse_type(other)

        return Point(
            self.x + other.x,
            self.y + other.y
        )

    def __sub__(self, other):
        other = Point._parse_type(other)

        return Point(
            self.x - other.x,
            self.y - other.y
        )

    def __truediv__(self, other):
        other = Point._parse_type(other)

        return Point(
            self.x / other.x,
            self.y / other.y
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
