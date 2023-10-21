'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 17:10:16
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-21 11:12:49
FilePath: /CSInference/csinference/algorithm/algorithm.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Tuple, List, Union
import json
from numpy import Inf
import os

from csinference.core.cost import FunctionCost
from csinference.core.util import Instance, Cfg
from csinference.core.latency import CPULatency, GPULatency


class FunctionCfg():
    def __init__(self, config) -> None:
        self.config = config
        pass

    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Tuple[dict, dict]:
        self.arrival_rate = arrival_rate
        self.slo = slo
        return {}, {}


# TODO: support BATCH for baseline
class BATCH(FunctionCfg):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Tuple[dict, dict]:
        self.arrival_rate = arrival_rate
        self.slo = slo
        return {}, {}


class CSInference(FunctionCfg):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_name = self.config['model_name']
        with open(os.path.join(self.config['cfg_path'], 'model.json'), 'r') as f:
            self.model_config = json.load(f)

        self.get_lat_cal()

    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name)
        # TODO: support both A10 and T4
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)

    def constrant(self, time_out: float, instance: Instance, batch_size: int) -> bool:
        if instance.gpu is None:
            if time_out + self.cpu_lat_cal.lat_max(instance, batch_size) > self.SLO:
                return False
            return True
        else:
            if time_out + self.gpu_lat_cal.lat_max(instance, batch_size) > self.SLO:
                return False
            return True

    def cpu_optimal(self, Res_CPU: List, B_CPU: List[int], R_CPU: float, cpu_min_cost: float, cpu_min_cfg: Union[Cfg, None]) -> Tuple[float, Union[Cfg, None]]:
        for cpu in Res_CPU:
            for b in B_CPU:
                instance_cpu = Instance(cpu, 4 * cpu, None)
                cpu_cost_cal = FunctionCost(instance_cpu)
                if self.constrant((b-1) / R_CPU, instance_cpu, b) is False:
                    continue
                cpu_cost = cpu_cost_cal.cost_with_distribution(b / R_CPU, self.arrival_rate, b, self.cpu_lat_cal, instance_cpu)
                if cpu_cost < cpu_min_cost:
                    cpu_min_cost = cpu_cost
                    cpu_min_cfg = Cfg(instance_cpu, b, cpu_min_cost, R_CPU, self.SLO)
        return cpu_min_cost, cpu_min_cfg

    def gpu_optimal(self, Res_GPU: List, B_GPU: List[int], R_GPU: float, gpu_min_cost: float, gpu_min_cfg: Union[Cfg, None]) -> Tuple[float, Union[Cfg, None]]:
        for gpu in Res_GPU:
            for b in B_GPU:
                # TODO: support both A10 and T4
                instance_gpu = Instance(gpu / 3, gpu / 3 * 4, gpu)
                gpu_cost_cal = FunctionCost(instance_gpu)
                if self.constrant((b-1) / R_GPU, instance_gpu, b) is False:
                    continue
                gpu_cost = gpu_cost_cal.cost_with_distribution(b / R_GPU, self.arrival_rate, b, self.gpu_lat_cal, instance_gpu)
                if gpu_cost < gpu_min_cost:
                    gpu_min_cost = gpu_cost
                    gpu_min_cfg = Cfg(instance_gpu, b, gpu_min_cost, R_GPU, self.SLO)
        return gpu_min_cost, gpu_min_cfg

    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Tuple[Union[Cfg, None], Union[Cfg, None]]:
        self.arrival_rate = arrival_rate
        if slo is not None:
            self.SLO = slo
        else:
            self.SLO = self.config['SLO']

        B_CPU = list(range(1, 17, 1))
        Res_CPU = list(range(1, 17, 1))

        B_GPU = list(range(1, 129, 1))
        # TODO: support both A10 and T4
        Res_GPU = list(range(3, 25, 1))

        cpu_min_cost = Inf
        cpu_min_cfg = None

        gpu_min_cost = Inf
        gpu_min_cfg = None

        for alpha in [0, 1]:
            beta = 1 - alpha
            R_CPU = arrival_rate * alpha
            R_GPU = arrival_rate * beta
            if alpha > 0:
                cpu_min_cost, cpu_min_cfg = self.cpu_optimal(
                    Res_CPU, B_CPU, R_CPU, cpu_min_cost, cpu_min_cfg)

            if beta > 0:
                gpu_min_cost, gpu_min_cfg = self.gpu_optimal(
                    Res_GPU, B_GPU, R_GPU, gpu_min_cost, gpu_min_cfg)

        return cpu_min_cfg, gpu_min_cfg


def NewFunctionCfg(algorithm: str, config: dict) -> FunctionCfg:
    if algorithm == "BATCH":
        return BATCH(config)
    elif algorithm == "CSInference":
        return CSInference(config)
    else:
        return FunctionCfg(config)
