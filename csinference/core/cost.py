'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-09 22:27:45
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-30 11:42:28
FilePath: /CSInference/csinference/core/cost.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

import math
from csinference.core.latency import Latency
from csinference.core.util import Instance, batch_distribution
from typing import List


class FunctionCost():
    def __init__(self) -> None:
        self.cpu_cost = 0.00009
        self.mem_cost = 0.000009
        self.gpu_cost = 0.00011
        self.invocation_cost = 0.009 / 10000

    def cost(self, duration: float, batch: int, instance: Instance, billed_second : bool = True) -> float:
        if instance.gpu is None or billed_second is False:
            gpu = 0
        else:
            gpu = instance.gpu
            duration = math.ceil(duration)
        return (self.invocation_cost +
                (instance.cpu * self.cpu_cost +
                 instance.mem * self.mem_cost +
                    gpu * self.gpu_cost) * duration) / batch

    def cost_with_distribution(self, time_out: float, rps: float, batch_max: int, lat_cal: Latency, instance: Instance) -> float:
        if batch_max == 1:
            return self.cost(lat_cal.lat_avg(instance, 1), 1, instance)
        p = batch_distribution(rps, batch_max, time_out)
        return self.cost_with_probability(instance, p, lat_cal)

    def cost_with_probability(self, instance: Instance, probability: List[float], lat_cal: Latency) -> float:
        c = 0.0
        for i in range(len(probability)):
            c += self.cost(lat_cal.lat_avg(instance, i + 1),
                           i+1, instance) * probability[i]
        return c
