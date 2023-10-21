'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-19 22:26:08
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-21 12:31:27
FilePath: /CSInference/csinference/core/util.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''

from typing import Union


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
	def __init__(self, instance: Instance, batch_size : int, cost: float, rps : float, slo : float, timeout : float) -> None:
		self.instance = instance
		self.batch_size = batch_size
		self.cost = cost
		self.rps = rps
		self.timeout = timeout
		self.slo = slo

	def __str__(self):
		if self.instance.gpu is None:
			return "cpu:\t{%0.2f}" % self.instance.cpu + "\n" + \
			"mem:\t{%0.2f}" % self.instance.mem + "\n" + \
			"batch:\t{%d}" % self.batch_size + "\n" + \
			"rps:\t{%0.2f}" % self.rps + "\n" + \
			"slo:\t{%0.2f}" % self.slo + "\n" + \
			"timeout\t{%0.2f}" % self.timeout + "\n" + \
			"cost:\t{%0.2e}" % self.cost + "\n"
		return "cpu:\t{%0.2f}" % self.instance.cpu + "\n" + \
			"mem:\t{%0.2f}" % self.instance.mem + "\n" + \
			"gpu:\t{%d}" % self.instance.gpu + "\n" + \
			"batch:\t{%d}" % self.batch_size + "\n" + \
			"rps:\t{%0.2f}" % self.rps + "\n" + \
			"slo:\t{%0.2f}" % self.slo + "\n" + \
			"timeout\t{%0.2f}" % self.timeout + "\n" + \
			"cost:\t{%0.2e}" % self.cost + "\n"
