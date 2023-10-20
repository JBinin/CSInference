'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 17:42:39
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-20 23:43:42
FilePath: /CSInference/main.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

import json
import os
import csinference

import argparse


def simulation(config: dict, arrival_low, arrival_high):
    print("Simulation: ")
    print("Arrival range: [%d, %d]" % (arrival_low, arrival_high))

    function_cfger = csinference.NewFunctionCfg(config["algorithm"], config)

    cpu_cfgs = []
    gpu_cfgs = []
    for arrival_rate in range(arrival_low, arrival_high + 1, 5):
        cpu_cfg, gpu_cfg = function_cfger.get_config(arrival_rate)
        cpu_cfgs.append(cpu_cfg)
        gpu_cfgs.append(gpu_cfg)
    return cpu_cfgs, gpu_cfgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSInference')
    parser.add_argument('--config', type=str,
                        default='conf', help='config path')

    args = parser.parse_args()

    with open(os.path.join(args.config, "config.json"), 'r') as f:
        config = json.load(f)

    config["cfg_path"] = args.config

    result = simulation(config, 1, 100)
    length = len(result[0])
    cfgs = []
    for i in range(length):
        if result[0][i] is None:
            cfgs.append(result[1][i])
            continue
        elif result[1][i] is None:
            continue
        else:
            if result[0][i].cost < result[1][i].cost:
                cfgs.append(result[0][i])
            else:
                cfgs.append(result[1][i])
    
    for i in cfgs:
        print(i)