'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 22:26:08
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-06 01:37:23
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


class Mem:
    def __init__(self, model_config: dict, model_name : str) -> None:
        self.model_config = model_config[model_name]
        # CPU
        self.a, self.b = self.model_config["CPU"]["mem"]
        # GPU
        self.mem = self.model_config["GPU"]["mem"]
        self.gpu_mem = self.model_config["GPU"]["gpu_mem"]

    def get_mem(self, cpu, mem):
        mem = mem / 1024
        if cpu > mem * 4:
            return None
        elif cpu > mem:
            mem = cpu
        return mem

    def get_cpu_mem(self, cpu : float, batch : int):
        mem =  self.a * batch + self.b
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return self.get_mem(cpu, mem)

    def get_gpu_mem(self, cpu : float, batch : Union[int, None] = None):
        mem = self.mem
        mem = int(mem)
        if mem % 64 != 0:
            mem = ((mem // 64) + 1) * 64
        return self.get_mem(cpu, mem)
    
    def get_gpu_gpu_mem(self, batch : int):
        for i in range(len(self.gpu_mem)):
            if batch <= self.gpu_mem[i]:
                return i+1
        return len(self.gpu_mem) + 1



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
            "cost:\t\t{%0.3e}" % self.cost + "\n" \
            "latency:\t{%0.3f}" % self.latency + "\n"
        if self.instance.gpu is not None:
            ret = "gpu:\t\t{%d}" % self.instance.gpu + "\n" + ret
        return ret


class Cfgs:
    def __init__(self, *cfgs : Union[Cfg, None]) -> None:
        self.cfgs = cfgs

    def get_resource(self):
        for cfg in self.cfgs:
            if cfg is not None:
                if cfg.instance.gpu is not None:
                    return cfg.instance.gpu, True
                else:
                    return cfg.instance.cpu, False
        return None, False
    
    def get_batch(self):
        for cfg in self.cfgs:
            if cfg is not None:
                return cfg.batch_size
        return None
    
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
        ret = "total cost:\t{%0.3e}" % self.cost() + "\n" + \
            "total latency:\t{%0.3f}" % self.lat() + "\n" + \
            "slo:\t{%0.2e}" % slo+ "\n" + ret
        return "*" * 24 + "\n" + ret + "*" * 24 + "\n"
