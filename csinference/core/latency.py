'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 21:04:25
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-25 16:06:02
FilePath: /CSInference/csinference/core/latency.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
from typing import List
import math
from csinference.core.util import Instance, batch_distribution
import numpy as np

from abc import ABC, abstractmethod


class Latency(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        pass
    
    def lat_with_distribution(self, time_out : float, rps : float, batch_max : int, instance : Instance) -> float:
        if batch_max == 1:
            return self.lat_avg(instance, 1)

        p = batch_distribution(rps, batch_max, time_out)
        return self.lat_with_probability(instance, p)

    def lat_with_probability(self, instance : Instance, probability : List[float]) ->float:
        tmp = 0.0
        for i in range(len(probability)):
            tmp += probability[i] * (i+1)
        for i in range(len(probability)):
            probability[i] = probability[i] * (i+1) / tmp

        l = 0.0
        for i in range(len(probability)):
            l += self.lat_avg(instance, i + 1) * probability[i]
        return l

class CPULatency(Latency):
    def __init__(self, params: dict, model_name: str, fitting_metod : str = 'Exponential') -> None:
        super().__init__()
        self.model_name = model_name
        self.fitting_metod = fitting_metod

        self.params_avg = params['avg'][self.fitting_metod]
        self.params_max = params['max'][self.fitting_metod]

    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            f = self.params_avg['f']
            g = self.params_avg['g']
            k = self.params_avg['k']
            F = f[0] * batch_size + f[1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return k[0] * F * G + k[1] * F + k[2] * G + k[3]
        elif self.fitting_metod == 'Polynomial':
            f = self.params_avg['f']
            g = self.params_avg['g']
            k = self.params_avg['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return F / G + k[0]
        return np.Inf

    def lat_max(self, instance: Instance, batch_size: int) -> float:
        cpu = instance.cpu
        if self.fitting_metod == 'Exponential':
            f = self.params_max['f']
            g = self.params_max['g']
            k = self.params_max['k']
            F = f[0] * batch_size + f[1]
            G = g[0] * np.exp(-cpu / g[1]) + g[2]
            return k[0] * F * G + k[1] * F + k[2] * G + k[3]
        elif self.fitting_metod == 'Polynomial':
            f = self.params_max['f']
            g = self.params_max['g']
            k = self.params_max['k']
            F = f[0] * batch_size + f[1]
            G = cpu + g[0]
            return F / G + k[0]
        return np.Inf


class GPULatency(Latency):
    def __init__(self, params: dict, model_name: str) -> None:
        super().__init__()
        self.model_name = model_name
        self.g1 = params['l1']
        self.g2 = params['l2']
        self.t = params['t']
        self.G = params['G']

    def lat_avg(self, instance: Instance, batch_size: int) -> float:
        gpu = instance.gpu
        L = self.g1 * batch_size + self.g2
        return L * self.G / gpu

    def lat_max(self, instance: Instance, batch_size: int) -> float:
        gpu = instance.gpu
        L = self.g1 * batch_size + self.g2
        n = math.floor(L / (gpu * self.t))
        return (self.G - gpu) * (n+1) * self.t + L
