'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 17:10:16
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-29 19:38:03
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
from csinference.core.util import Instance, Cfg, Cfgs, batch_distribution, get_timeout_list
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

    def constraint_lat(self, time_out: float, instance: Instance, batch_size: int) -> bool:
        if instance.gpu is None:
            if time_out + self.cpu_lat_cal.lat_max(instance, batch_size) > self.SLO:
                return False
            return True
        else:
            if time_out + self.gpu_lat_cal.lat_max(instance, batch_size) > self.SLO:
                return False
            return True
    
    def get_max_timeout(self, instance : Instance, batch_size : int) -> float:
        if instance.gpu is None:
            return self.SLO - self.cpu_lat_cal.lat_max(instance, batch_size)
        else:   
            return self.SLO - self.gpu_lat_cal.lat_max(instance, batch_size)

    def constraint_cost(self, cost):
        if cost > self.SLO:
            return False
        return True

    
    def cpu_optimal_cost(self, Res_CPU: List, B_CPU: List[int], cpu_min: float, cpu_min_cfg: Union[Cfg, None], proportion : float) -> Tuple[float, Union[Cfg, None]]:
        R_CPU = self.arrival_rate * proportion
        for cpu in Res_CPU:
            instance_cpu = Instance(cpu, 4 * cpu, None)
            for b in B_CPU:
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
            instance_gpu = Instance(gpu / 3, gpu / 3 * 4, gpu)
            for b in B_GPU:
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
        for res in Res:
            if is_gpu:
                ins = Instance(res / 3, res / 3 * 4, res)
            else:
                ins = Instance(res, 4 * res, None)
            for b in B:
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
                        time_out = tau
                    p = batch_distribution(rps, b, time_out)
                    lat = lat_cal.lat_with_probability(ins, p, time_out, tau)[1]
                    cost = self.cost_cal.cost_with_probability(ins, p, lat_cal)
                    tmp = Cfgs(Cfg(ins, b, cost, rps, self.SLO, time_out, proportion, lat))
                    if tmp.lat() <= cfgs.lat():
                        cfgs = tmp
        new_cfg = None
        for cfg in cfgs.get_cfgs():
            if cfg is not None:
                new_cfg = cfg
                break
        return self.optimize_cfg(new_cfg)
    

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

            for gpu in Res_GPU:
                instance_gpu = Instance(gpu / 3, gpu / 3 * 4, gpu)
                for b_gpu in B_GPU:
                    for t_gpu in get_timeout_list(R_GPU, b_gpu, cfgs.lat()):
                        p_gpu = batch_distribution(R_GPU, b_gpu, t_gpu)
                        gpu_lat = self.gpu_lat_cal.lat_with_probability(instance_gpu, p_gpu, t_gpu, (b_gpu-1)/R_GPU)[1]
                        gpu_cost = self.cost_cal.cost_with_probability(instance_gpu, p_gpu, self.gpu_lat_cal)
                        gpu_cfg =  Cfg(instance_gpu, b_gpu, gpu_cost, R_GPU, self.SLO, t_gpu, beta, gpu_lat)
                        
                        for cpu in Res_CPU:
                            instance_cpu = Instance(cpu, 4 * cpu, None)
                            for b_cpu in B_CPU:
                                for t_cpu in get_timeout_list(R_CPU, b_cpu, cfgs.lat()):
                                    p_cpu = batch_distribution(R_CPU, b_cpu, t_cpu)
                                    cpu_lat = self.cpu_lat_cal.lat_with_probability(instance_cpu, p_cpu, t_cpu, (b_cpu-1)/R_CPU)[1]
                                    cpu_cost = self.cost_cal.cost_with_probability(instance_cpu, p_cpu, self.cpu_lat_cal)
                                    cpu_cfg = Cfg(instance_cpu, b_cpu, cpu_cost, R_CPU, self.SLO, t_cpu, alpha, cpu_lat)
                                    
                                    tmp = Cfgs(cpu_cfg, gpu_cfg)
                                    if self.constraint_cost(tmp.cost()) is True and tmp.lat() < cfgs.lat():
                                        cfgs = tmp
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
        Res_CPU = list(np.arange(Res_CPU_low, Res_CPU_high+1, 0.2))

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
