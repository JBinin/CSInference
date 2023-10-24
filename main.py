'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-08 17:42:39
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-24 19:49:10
FilePath: /CSInference/main.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

import json
import os
import csinference

import argparse
import numpy as np


def simulation(config: dict, arrival_low, arrival_high):
    print("Simulation: ")
    print("Arrival range: [%d, %d]" % (arrival_low, arrival_high))

    function_cfger = csinference.NewFunctionCfg(config["algorithm"], config)

    cfgs = []
    for arrival_rate in range(arrival_low, arrival_high + 1, 1):
        cfg = function_cfger.get_config(arrival_rate)
        cfgs.append(cfg)
    return cfgs


def simulate_slo(config: dict, arrival : float):
    slo_low = 1
    slo_high = 2
    print("Simulation: ")
    print("SLO range: [%0.2f, %0.2f]" % (slo_low, slo_high))

    function_cfger = csinference.NewFunctionCfg(config["algorithm"], config)

    for slo in np.arange(slo_low, slo_high, 0.3):
        cfgs = function_cfger.get_config(arrival, slo)
        print(cfgs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CSInference')
    parser.add_argument('--config', type=str,
                        default='conf', help='config path')

    args = parser.parse_args()

    with open(os.path.join(args.config, "config.json"), 'r') as f:
        config = json.load(f)

    config["cfg_path"] = args.config
    for rps in [1, 5, 60]:
        simulate_slo(config, rps)