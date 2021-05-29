from typing import List

import numpy as np
from . import BaseGenerator

class LaneVehicleGenerator(BaseGenerator):
    """
    Generate State or Reward based on statistics of lane vehicles.

    Parameters
    ----------
    world : World object
    I : Intersection object
    fns : list of statistics to get, currently support "lane_count", "lane_waiting_count" , "lane_waiting_time_count", "lane_delay" and "pressure"
    in_only : boolean, whether to compute incoming lanes only
    average : None or str
        None means no averaging
        "road" means take average of lanes on each road
        "all" means take average of all lanes
    negative : boolean, whether return negative values (mostly for Reward)
    """
    def __init__(self, world, I, fns, in_only=False, average=None, negative=False, include_phase=False):
        self.world = world
        self.I = I

        # get lanes of intersections
        self.lanes = []
        if in_only:
            roads = I.in_roads
        else:
            roads = I.roads
        for road in roads:
            from_zero = (road["startIntersection"] == I.id) if self.world.RIGHT else (road["endIntersection"] == I.id)
            self.lanes.append([road["id"] + "_" + str(i) for i in range(len(road["lanes"]))[::(1 if from_zero else -1)]])

        # subscribe functions
        self.world.subscribe(fns)
        self.fns = fns

        # calculate result dimensions
        size = sum(len(x) for x in self.lanes)
        if average == "road":
            size = len(roads)
        elif average == "all":
            size = 1
        self.ob_length = len(fns) * size

        uses_autoencoder = "lane_count_with_hidden_autoencoder" in fns or "auto_encoder_output" in fns
        uses_hidden = "lane_count_with_hidden" in fns or "lane_count_with_hidden_autoencoder" in fns

        if uses_autoencoder:
            self.ob_length += size

        if uses_hidden:
            self.ob_length += 4

        if self.ob_length == 3:
            self.ob_length = 4

        if include_phase:
            self.ob_length += 9

        self.average = average
        self.negative = negative
        self.include_phase = include_phase

    def generate(self):
        results = [self.world.get_info(fn) for fn in self.fns]

        ret = np.array([])
        for i in range(len(self.fns)):
            result = results[i]
            hidden_infos = []
            contains_hidden_label = self.fns[i] in {"lane_count_with_hidden", "lane_count_with_hidden_autoencoder"}
            fn_result = np.array([])

            for road_lanes in self.lanes:
                road_result = []

                for lane_id in road_lanes:
                    if contains_hidden_label:
                        is_hidden, *feats = result[lane_id]
                        hidden_infos.append(is_hidden)
                        road_result.append(feats)
                    else:
                        road_result.append(result[lane_id])

                if len(road_result) >= 1 and isinstance(road_result[0], List):
                    road_result = [r for rs in road_result for r in rs]

                if self.average == "road" or self.average == "all":
                    road_result = np.mean(road_result)
                else:
                    if contains_hidden_label:
                        assert all(hidden_infos[i] == hidden_infos[0] for i in range(len(hidden_infos)))
                        road_result.append(hidden_infos[0])
                    road_result = np.array(road_result)
                fn_result = np.append(fn_result, road_result)
            
            if self.average == "all":
                fn_result = np.mean(fn_result)
            ret = np.append(ret, fn_result)
        if self.negative:
            ret = ret * (-1)
        origin_ret = ret
        if len(ret) == 3:
            ret_list = list(ret)
            ret_list.append(0)
            ret = np.array(ret_list)
        if len(ret) == 2:
            ret_list = list(ret)
            ret_list.append(0)
            ret_list.append(0)
            ret = np.array(ret_list)

        if self.include_phase:
            ret_list = list(ret)
            phase_one_hot = [0.0] * 9
            phase_one_hot[self.I.current_phase] = 1.0
            ret = np.array(phase_one_hot + ret_list)

        return ret

#TODO: Add self.I.current_phase

if __name__ == "__main__":
    from world import World
    world = World("examples/config.json", thread_num=1)
    laneVehicle = LaneVehicleGenerator(world, world.intersections[0], ["count"], False, "road")
    for _ in range(100):
        world.step()
    print(laneVehicle.generate())