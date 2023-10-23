'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 21:04:25
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-23 16:34:43
FilePath: /CSInference/csinference/core/latency.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
import math
from csinference.core.util import Instance
import numpy as np

class CPULatency:
    def __init__(self, params: dict, model_name: str, fitting_metod : str = 'Exponential') -> None:
        self.model_name = model_name
        self.fitting_metod = fitting_metod

        self.params_avg = params['avg'][self.fitting_metod]
        self.params_max = params['max'][self.fitting_metod]

    def lat_avg(self, instance: Instance, batch_size: int):
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

    def lat_max(self, instance: Instance, batch_size: int):
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


class GPULatency:
    def __init__(self, params: dict, model_name: str) -> None:
        self.model_name = model_name
        self.g1 = params['l1']
        self.g2 = params['l2']
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
