'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 17:10:16
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-11-06 01:32:54
FilePath: /CSInference/csinference/algorithm/algorithm.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Tuple, List, Union
import json
import numpy as np
import os
from abc import ABC, abstractmethod

from csinference.core.cost import FunctionCost
from csinference.core.util import Instance, Cfg, Cfgs, batch_distribution, get_timeout_list, Mem
from csinference.core.latency import CPULatency, GPULatency


class FunctionCfg(ABC):
    def __init__(self, config) -> None:
        self.config = config
        pass
    
    @abstractmethod
    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Cfgs:
        pass


# TODO: support BATCH for baseline
class BATCH(FunctionCfg):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Cfgs:
        self.arrival_rate = arrival_rate
        self.slo = slo
        return Cfgs()


class CSInference(FunctionCfg):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_name = self.config['model_name']
        with open(os.path.join(self.config['cfg_path'], 'model.json'), 'r') as f:
            self.model_config = json.load(f)
        self.mem_cal = Mem(self.model_config, self.model_name)
        self.get_lat_cal()
        self.get_cost_cal()

    def get_lat_cal(self) -> None:
        self.cpu_lat_cal = CPULatency(
            self.model_config[self.model_name]['CPU'], self.model_name)
        # TODO: support both A10 and T4
        self.gpu_lat_cal = GPULatency(
            self.model_config[self.model_name]['GPU']['A10'], self.model_name)
        
    def get_cost_cal(self) -> None:
        self.cost_cal = FunctionCost()

    def constraint_lat(self, time_out: float, instance: Instance, batch_size: int, slo : Union[float, None] = None) -> bool:
        if slo is None:
            slo = self.SLO
        if instance.gpu is None:
            if time_out + self.cpu_lat_cal.lat_max(instance, batch_size) > slo:
                return False
            return True
        else:
            if time_out + self.gpu_lat_cal.lat_max(instance, batch_size) > slo:
                return False
            return True
    
    def get_max_timeout(self, instance : Instance, batch_size : int, slo : Union[float, None] = None) -> float:
        if slo is None:
            slo = self.SLO
        if instance.gpu is None:
            return slo - self.cpu_lat_cal.lat_max(instance, batch_size)
        else:   
            return slo - self.gpu_lat_cal.lat_max(instance, batch_size)

    def constraint_cost(self, cost):
        if cost > self.SLO:
            return False
        return True

    
    def cpu_optimal_cost(self, Res_CPU: List, B_CPU: List[int], cpu_min: float, cpu_min_cfg: Union[Cfg, None], proportion : float) -> Tuple[float, Union[Cfg, None]]:
        R_CPU = self.arrival_rate * proportion
        for cpu in Res_CPU:
            for b in B_CPU:
                mem = self.mem_cal.get_cpu_mem(cpu, b)
                if mem is None:
                    break
                instance_cpu = Instance(cpu, mem, None)
                for t in get_timeout_list(R_CPU, b, self.SLO):
                    
                    p = batch_distribution(R_CPU, b, t)
                    cpu_cost = self.cost_cal.cost_with_probability(instance_cpu, p, self.cpu_lat_cal)
                    cpu_lat = self.cpu_lat_cal.lat_with_probability(instance_cpu, p, t, (b-1)/R_CPU)[1]
                    if self.constraint_lat(t, instance_cpu, b) is False:
                        break
                    else:
                        if cpu_cost < cpu_min:
                            cpu_min = cpu_cost
                            cpu_min_cfg = Cfg(instance_cpu, b, cpu_cost, R_CPU, self.SLO, t, proportion, cpu_lat)      
        return cpu_min, cpu_min_cfg


    def gpu_optimal_cost(self, Res_GPU: List, B_GPU: List[int], gpu_min: float, gpu_min_cfg: Union[Cfg, None], proportion : float) -> Tuple[float, Union[Cfg, None]]:
        R_GPU = self.arrival_rate * proportion
        for gpu in Res_GPU:
            for b in B_GPU:
                if self.mem_cal.get_gpu_gpu_mem(b) > gpu:
                    break
                cpu = gpu / 3
                mem = self.mem_cal.get_gpu_mem(cpu, b)
                if mem is None:
                    break
                instance_gpu = Instance(cpu, mem, gpu)
                # TODO: support both A10 and T4
                for t in get_timeout_list(R_GPU, b, self.SLO):
                    p = batch_distribution(R_GPU, b, t)
                    gpu_cost = self.cost_cal.cost_with_probability(instance_gpu, p, self.gpu_lat_cal)
                    gpu_lat = self.gpu_lat_cal.lat_with_probability(instance_gpu, p, t, (b-1)/R_GPU)[1]
                    if self.constraint_lat(t, instance_gpu, b) is False:
                        break
                    else:
                        if gpu_cost < gpu_min:
                            gpu_min = gpu_cost
                            gpu_min_cfg = Cfg(instance_gpu, b, gpu_cost, R_GPU, self.SLO, t, proportion, gpu_lat)
        return gpu_min, gpu_min_cfg
    
    def get_config_cost(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int]):
        cfgs = Cfgs()        
        for alpha in [0, 1]:
            cpu_cfg = None
            gpu_cfg = None 
            beta = 1 - alpha
            if alpha > 0:
                cpu_cfg = self.get_config_with_one_platform(Res_CPU, B_CPU, False, alpha)
                if cpu_cfg is None:
                    continue
            if beta > 0:
                gpu_cfg = self.get_config_with_one_platform(Res_GPU, B_GPU, True, beta)
                if gpu_cfg is None:
                    continue
            tmp = Cfgs(cpu_cfg, gpu_cfg)
            if tmp.cost() < cfgs.cost():
                cfgs = tmp
        return cfgs

    def get_config_with_one_platform(self, Res: List, B: List[int], is_gpu: bool, proportion = 1):
        rps = self.arrival_rate * proportion
        if is_gpu:
            lat_cal = self.gpu_lat_cal
        else:
            lat_cal = self.cpu_lat_cal
        cfgs = Cfgs()

        if self.constraint == 'cost':
            # if contraint is cost, find optimal Res from max to min
            Res = list(reversed(Res))

        for res in Res:
            for b in B:
                if is_gpu:
                    gpu = self.mem_cal.get_gpu_gpu_mem(b)
                    if res < gpu:
                        break
                    cpu = res / 3
                    mem = self.mem_cal.get_gpu_mem(cpu, b)
                    if mem is None:
                        break
                    ins = Instance(cpu, mem, res)
                else:
                    ins = Instance(res, self.mem_cal.get_cpu_mem(res, b), None)
                tau = (b-1)/rps
                if self.constraint == 'lat': 
                    time_out = self.get_max_timeout(ins, b)
                    # constraint check
                    if time_out < 0:
                        break
                    if b == 1:
                        time_out = 0
                    p = batch_distribution(rps, b, time_out)
                    lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
                    cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
                    tmp = Cfgs(Cfg(ins, b, cost, rps, self.SLO, time_out, proportion, lat))
                    if tmp.cost() < cfgs.cost():
                        cfgs = tmp
                elif self.constraint == 'cost':
                    if b == 1:
                        time_out = 0
                    else:
                        #TODO: how to configure timeout for cost constraint
                        time_out = min(tau, 5)
                    p = batch_distribution(rps, b, time_out)
                    lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
                    cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
                    if cost <= self.SLO:   
                        if lat <= cfgs.lat():
                            tmp = Cfg(ins, b, cost, rps, self.SLO, time_out, proportion, lat)
                            tmp = self.optimize_cfg(tmp)
                            cfgs = Cfgs(tmp)
                        break
            if self.constraint == 'cost':
                c = cfgs.get_cfgs()
                if len(c) != 0 and c[0] != None and c[0].batch_size == 1:
                    break
        
        new_cfg = None
        for cfg in cfgs.get_cfgs():
            if cfg is not None:
                new_cfg = cfg
                break
        if self.constraint == 'lat':
            new_cfg = self.optimize_cfg(new_cfg)
        return new_cfg
    

    def optimize_cfg(self, cfg : Union[Cfg, None]) -> Union[Cfg, None]:
        if cfg is None or cfg.batch_size == 1:
            return cfg
        timeout = cfg.timeout
        timeout_list = np.arange(timeout, 0.01, -0.01)
        for t in timeout_list:
            p = batch_distribution(cfg.rps, cfg.batch_size, t)
            if cfg.instance.gpu is not None:
                lat = self.gpu_lat_cal.lat_with_probability(cfg.instance, p, t, (cfg.batch_size-1)/cfg.rps)[1]
                cost = self.cost_cal.cost_with_probability(cfg.instance, p, self.gpu_lat_cal)
            else:
                lat = self.cpu_lat_cal.lat_with_probability(cfg.instance, p, t, (cfg.batch_size-1)/cfg.rps)[1]
                cost = self.cost_cal.cost_with_probability(cfg.instance, p, self.cpu_lat_cal)
            if self.constraint == 'cost':
                if cost <= self.SLO and lat < cfg.latency:
                    cfg.cost = cost
                    cfg.latency = lat
                    cfg.timeout = t
            elif self.constraint == 'lat':
                if cost <= cfg.cost:
                    cfg.cost = cost
                    cfg.latency = lat
                    cfg.timeout = t
        return cfg
    
    
    def get_config_latency(self, Res_CPU: List, B_CPU: List[int], Res_GPU: List, B_GPU: List[int]):
        cfgs = Cfgs(self.get_config_with_one_platform(Res_CPU, B_CPU, False))
        gpu_cfgs = Cfgs(self.get_config_with_one_platform(Res_GPU, B_GPU, True))
        if gpu_cfgs.lat() < cfgs.lat():
            cfgs = gpu_cfgs

        for alpha in np.arange(0.1, 1.0, 0.1):
            beta = 1-alpha
            R_CPU = self.arrival_rate * alpha
            R_GPU = self.arrival_rate * beta

        return cfgs

    def get_config(self, arrival_rate: float, slo: Union[float, None] = None) -> Cfgs:
        self.arrival_rate = arrival_rate
        if slo is not None:
            self.SLO = slo
        else:
            self.SLO = self.config['SLO']
        self.constraint = self.config["constraint"]

        B_CPU_low, B_CPU_high = self.config["B_CPU"]
        B_CPU = list(range(B_CPU_low, B_CPU_high+1, 1))
        Res_CPU_low, Res_CPU_high = self.config["Res_CPU"]
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+1, 0.1))

        B_GPU_low, B_GPU_high = self.config["B_GPU"]
        B_GPU = list(range(B_GPU_low, B_GPU_high+1, 1))
        # TODO: support both A10 and T4
        Res_GPU_low, Res_GPU_high = self.config["Res_GPU"]
        Res_GPU = list(range(Res_GPU_low, Res_GPU_high+1, 1))

        if self.constraint == "lat":
            return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU)
        elif self.constraint == "cost":
            return self.get_config_latency(Res_CPU, B_CPU, Res_GPU, B_GPU)
        else:
            return self.get_config_cost(Res_CPU, B_CPU, Res_GPU, B_GPU)


def NewFunctionCfg(algorithm: str, config: dict) -> FunctionCfg:
    if algorithm == "BATCH":
        return BATCH(config)
    elif algorithm == "CSInference":
        return CSInference(config)
    # default
    return CSInference(config)
