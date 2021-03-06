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


class MaxPressureAgent(Agent):

    def __init__(self, intersection: Intersection, t_min: int = 3, epsilon: float = 0.1):
        self.intersection = intersection
        self.t_min = t_min
        self.epsilon = epsilon
        self.t_since_last_change = 0

        self.prev_phase = 0

    def get_intersection(self) -> Intersection:
        return self.intersection

    def act(self, engine, step_data: dict):
        # Get ordered counts of incoming and outgoing lanes.

        incoming_counts = []
        outgoing_counts = []

        # Get densities for MaxPressure calculations
        for incoming_road in self.intersection.incoming_roads:
            road_length = incoming_road.length()
            for incoming_lane in incoming_road.lanes:
                incoming_counts.append(step_data['laneCounts'][incoming_lane] / road_length)

        for outgoing_road in self.intersection.outgoing_roads:
            road_length = outgoing_road.length()
            for outgoing_lane in outgoing_road.lanes:
                outgoing_counts.append(step_data['laneCounts'][outgoing_lane] / road_length)

        # TODO: Put in an RL class file.
        # MaxPressure algo: https://arxiv.org/pdf/1904.08117.pdf
        # (possible addition is density https://faculty.ist.psu.edu/jessieli/Publications/2019-KDD-presslight.pdf)

        # Lanes start from innerside.

        # Phase id 1: W-E, E-W.
        west_east_count = incoming_counts[1] - sum(outgoing_counts[9:12])
        east_west_count = incoming_counts[10] - sum(outgoing_counts[0:3])

        phase_1 = west_east_count + east_west_count

        # Phase id 2: N-S, S-N.
        north_south_count = incoming_counts[7] - sum(outgoing_counts[3:6])
        south_north_count = incoming_counts[4] - sum(outgoing_counts[6:9])

        phase_2 = north_south_count + south_north_count

        # Phase id 3: W-N, E-S.
        west_north_count = incoming_counts[0] - sum(outgoing_counts[6:9])
        east_south_count = incoming_counts[9] - sum(outgoing_counts[3:6])

        phase_3 = west_north_count + east_south_count

        # Phase id 4: N-E, S-W
        north_east_count = incoming_counts[6] - sum(outgoing_counts[9:12])
        south_west_count = incoming_counts[3] - sum(outgoing_counts[0:3])

        phase_4 = north_east_count + south_west_count

        all_phases = [phase_1, phase_2, phase_3, phase_4]

        if random.random() < self.epsilon:
            chosen_phase_id = random.randrange(0, len(all_phases))  # Choose random action
        else:
            chosen_phase_id = np.argmax(all_phases) + 1  # Choose "best" action

        self.t_since_last_change += 1

        if self.t_since_last_change >= self.t_min:
            # Change the phase only when T-minus has been surpassed
            engine.set_tl_phase(self.intersection.id, chosen_phase_id)

            # If we have a new phase, reset the timer.
            if chosen_phase_id != self.prev_phase:
                self.t_since_last_change = 0

            self.prev_phase = chosen_phase_id

    def get_prev_phase(self) -> int:
        return int(self.prev_phase)

