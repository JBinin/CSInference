'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-09 22:27:45
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-25 16:11:33
FilePath: /CSInference/csinference/core/cost.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

import math
from csinference.core.latency import Latency
from csinference.core.util import Instance, batch_distribution
from typing import List


class FunctionCost():
    def __init__(self, instance: Instance) -> None:
        self.cpu = instance.cpu
        self.mem = instance.mem
        self.gpu = instance.gpu
        self.cpu_cost = 0.000127
        self.mem_cost = 0.0000127
        self.gpu_cost = 0.00011
        self.invocation_cost = 0.009 / 10000

    def cost(self, duration: float, batch: int) -> float:
        if self.gpu is None:
            gpu = 0
        else:
            gpu = self.gpu
            duration = math.ceil(duration)
        return (self.invocation_cost + (self.cpu * self.cpu_cost + self.mem * self.mem_cost + gpu * self.gpu_cost) * duration) / batch

    def cost_with_distribution(self, time_out: float, rps: float, batch_max: int, lat_cal: Latency, instance) -> float:
        if batch_max == 1:
            return self.cost(lat_cal.lat_avg(instance, 1), 1)
        p = batch_distribution(rps, batch_max, time_out)
        return self.cost_with_probability(instance, p, lat_cal)

    def cost_with_probability(self, instance: Instance, probability: List[float], lat_cal: Latency) -> float:
        c = 0.0
        for i in range(len(probability)):
            c += self.cost(lat_cal.lat_avg(instance, i + 1),
                           i+1) * probability[i]
        return c
