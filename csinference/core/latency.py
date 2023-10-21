'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 21:04:25
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-21 22:17:26
FilePath: /CSInference/csinference/core/latency.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
import math
from pyexpat import model
from csinference.core.util import Instance


class CPULatency:
    def __init__(self, params: dict, model_name: str) -> None:
        self.model_name = model_name

        self.k1 = params['k1']
        self.k2 = params['k2']
        self.k3 = params['k3']
        self.k4 = params['k4']
        self.k5 = params['k5']
        self.k6 = params['k6']
        self.k7 = params['k7']
        self.k8 = params['k8']

        if model_name == 'ResNet50':
            self.k9 = params['k9']
            self.k10 = params['k10']
            self.k11 = params['k11']
            self.k12 = params['k12']

    def lat_avg(self, instance: Instance, batch_size: int):
        cpu = instance.cpu
        if self.model_name == 'ResNet50':
            return (self.k1 * batch_size + self.k2 * batch_size * batch_size + self.k3) / (cpu + self.k4 * cpu * cpu + self.k5) + self.k6
        return (self.k1 * batch_size + self.k2) / (cpu + self.k3) + self.k4

    def lat_max(self, instance: Instance, batch_size: int):
        cpu = instance.cpu
        if self.model_name == 'ResNet50':
            return (self.k7 * batch_size + self.k8 * batch_size * batch_size + self.k9) / (cpu + self.k10 * cpu * cpu + self.k11) + self.k12
        return (self.k5 * batch_size + self.k6) / (cpu + self.k7) + self.k8


class GPULatency:
    def __init__(self, params: dict, model_name: str) -> None:
        self.model_name = model_name
        self.g1 = params['g1']
        self.g2 = params['g2']
        self.t = params['t']
        self.G = params['G']

    def lat_avg(self, instance: Instance, batch_size: int):
        gpu = instance.gpu
        L = self.g1 * batch_size + self.g2
        return L * self.G / gpu

    def lat_max(self, instance: Instance, batch_size: int):
        gpu = instance.gpu
        L = self.g1 * batch_size + self.g2
        n = math.floor(L / (gpu * self.t))
        return (self.G - gpu) * (n+1) * self.t + L
