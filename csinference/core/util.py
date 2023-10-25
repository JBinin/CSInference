'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 22:26:08
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-25 15:54:40
FilePath: /CSInference/csinference/core/util.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Union, List
import numpy as np
from scipy.linalg import expm


def batch_distribution(lam, B, T) -> List[float]:
    init_state = [0] * B
    init_state[0] = 1

    Q = np.zeros((B, B))
    for i in range(B):
        for j in range(B):
            if i != B-1 and j != B-1 and i == j:
                Q[i][j] = -lam
            elif j == i+1:
                Q[i][j] = lam

    p = [0] * B
    pTmp = np.dot(init_state, expm(Q * T))
    i = 0
    j = 0
    while j < B:
        if i < B - 1:
            p[i] = sum(pTmp[j:j+1])
        else:
            p[i] = 1 - sum(p)
        i += 1
        j += 1

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
            "slo:\t\t{%0.2f}" % self.slo + "\n" + \
            "timeout:\t{%0.2f}" % self.timeout + "\n" + \
            "proportion:\t{%0.2f}" % self.proportion + "\n" + \
            "cost:\t\t{%0.2e}" % self.cost + "\n" \
			"latency:\t{%0.2f}" % self.latency + "\n"
        if self.instance.gpu is not None:
            ret = "gpu:\t\t{%d}" % self.instance.gpu + "\n" + ret
        return "-" * 24 + "\n" + ret + "-" * 24 + "\n"


class Cfgs:
    def __init__(self, *cfgs) -> None:
        self.cfgs = cfgs

    def __str__(self):
        ret = ""
        cost = 0
        for cfg in self.cfgs:
            if cfg is not None:
                cost += cfg.cost * cfg.proportion
                ret += str(cfg)
        if ret == "":
            return "None\n"
        return "total cost:\t{%0.2e}" % cost + "\n" + ret
