from abc import abstractmethod, ABC

from roadnet_graph import Intersection
import random
import numpy as np


class Agent(ABC):

    @abstractmethod
    def get_intersection(self) -> Intersection:
        raise NotImplementedError()

    @abstractmethod
    def act(self, engine, step_data: dict):
        raise NotImplementedError()

    @abstractmethod
    def get_prev_phase(self) -> int:
        raise NotImplementedError()


