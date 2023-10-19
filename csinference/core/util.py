'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 22:26:08
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-19 22:40:08
FilePath: /CSInference/csinference/core/util.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Union

class Instance:
    def __init__(self, cpu : float, mem : Union[float, None], gpu : Union[int, None]) -> None:
        self.cpu = cpu
        if mem is None:
            self.mem = self.cpu * 2
        self.gpu = gpu
    
    def set_cpu(self, cpu : float):
        self.cpu = cpu
    
    def set_mem(self, mem : float):
        self.mem = mem
    
    def set_gpu(self, gpu : int):
        self.gpu = gpu