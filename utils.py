import json
from typing import AnyStr, Union
import torch
from dataclasses import dataclass

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")


def load_json(fp: AnyStr) -> Union[dict, list]:
    with open(fp, "r") as f:
        return json.load(f)

def store_json(o: Union[dict, list], fp: AnyStr):
    with open(fp, "w") as f:
        json.dump(o, f)

@dataclass
class Point:
    x: int
    y: int

    def __getitem__(self, item):
        if item == 0:
            return self.x
        if item == 1:
            return self.y

        raise ValueError("index must be 1 or 0")