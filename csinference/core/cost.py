'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-09 22:27:45
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-19 22:41:23
FilePath: /CSInference/csinference/core/cost.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
import math
from csinference.core.util import Instance

class FunctionCost():
    def __init__(self, instance : Instance) -> None:
        self.cpu = instance.cpu
        self.mem = instance.mem
        self.gpu = instance.gpu
        self.cpu_cost = 0.000127
        self.mem_cost = 0.0000127
        self.gpu_cost = 0.00011
        self.invocation_cost = 0.009 / 10000

    def cost(self, duration: float, batch: int = 1) -> float:
        if self.gpu is None:
            gpu = 0
        else:
            gpu = self.gpu
            duration = math.ceil(duration)
        return (self.invocation_cost + (self.cpu * self.cpu_cost + self.mem * self.mem_cost + gpu * self.gpu_cost) * duration) / batch

