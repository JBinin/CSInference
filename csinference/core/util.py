'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 22:26:08
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-29 16:41:03
FilePath: /CSInference/csinference/core/util.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Union, List
import numpy as np
from scipy.linalg import expm


def get_timeout_list(rps: float, batch_size: int, slo: float, step: float = 0.1):
    if batch_size == 1:
        return [0]
    
    if rps == 0:
        return [np.Inf]

    max_timeout = min(slo, batch_size / rps)
    timeout_list = np.arange(step, max_timeout, step).tolist()
    return timeout_list


def batch_distribution(lam: float, B: int, T: float) -> List[float]:
    init_state = [0] * B
    init_state[0] = 1

    Q = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i != B-1 and j != B-1 and i == j:
                Q[i][j] = -lam
            elif j == i+1:
                Q[i][j] = lam

    p = [0.0] * B
    pTmp = np.dot(init_state, expm(Q * T))
    for i in range(B):
        if i < B-1:
            p[i] = pTmp[i]
        else:
            p[i] = 1 - sum(p)
    return p


class Instance:
    def __init__(self, cpu: float, mem: Union[float, None], gpu: Union[int, None]) -> None:
        self.cpu = cpu
        if mem is None:
            self.mem = self.cpu * 2
        else:
            self.mem = mem
        self.gpu = gpu

    def set_cpu(self, cpu: float):
        self.cpu = cpu

    def set_mem(self, mem: float):
        self.mem = mem

    def set_gpu(self, gpu: int):
        self.gpu = gpu


class Cfg:
    def __init__(self, instance: Instance, batch_size: int, cost: float, rps: float, slo: float, timeout: float, proportion: float, latency: float) -> None:
        self.instance = instance
        self.batch_size = batch_size
        self.cost = cost
        self.rps = rps
        self.timeout = timeout
        self.slo = slo
        self.proportion = proportion
        self.latency = latency

    def __str__(self):
        ret = "cpu:\t\t{%0.2f}" % self.instance.cpu + "\n" + \
            "mem:\t\t{%0.2f}" % self.instance.mem + "\n" + \
            "batch:\t\t{%d}" % self.batch_size + "\n" + \
            "rps:\t\t{%0.2f}" % self.rps + "\n" + \
            "timeout:\t{%0.2f}" % self.timeout + "\n" + \
            "proportion:\t{%0.2f}" % self.proportion + "\n" + \
            "cost:\t\t{%0.2e}" % self.cost + "\n" \
            "latency:\t{%0.2f}" % self.latency + "\n"
        if self.instance.gpu is not None:
            ret = "gpu:\t\t{%d}" % self.instance.gpu + "\n" + ret
        return ret


class Cfgs:
    def __init__(self, *cfgs : Union[Cfg, None]) -> None:
        self.cfgs = cfgs
    
    def get_cfgs(self):
        return self.cfgs

    def cost(self):
        cost = 0
        for cfg in self.cfgs:
            if cfg is not None:
                cost += cfg.cost * cfg.proportion
        if cost == 0:
            return np.Inf
        return cost

    def lat(self):
        lat = 0
        for cfg in self.cfgs:
            if cfg is not None:
                lat += cfg.latency * cfg.proportion
        if lat == 0:
            return np.Inf
        return lat

    def __str__(self):
        ret = ""
        slo = None
        for cfg in self.cfgs:
            if cfg is not None:
                ret += "-" * 24 + "\n"
                ret += str(cfg)
                slo = cfg.slo
        if ret == "":
            return "None\n"
        ret = "total cost:\t{%0.2e}" % self.cost() + "\n" + \
            "total latency:\t{%0.2f}" % self.lat() + "\n" + \
            "slo:\t{%0.2e}" % slo+ "\n" + ret
        return "*" * 24 + "\n" + ret + "*" * 24 + "\n"
