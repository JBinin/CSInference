'''
Author: JBinin namechenjiabin@icloud.com
Date: 2023-10-09 22:27:45
LastEditors: JBinin namechenjiabin@icloud.com
LastEditTime: 2023-10-21 10:50:49
FilePath: /CSInference/csinference/core/cost.py
Description: 

Copyright (c) 2023 by icloud-ecnu, All Rights Reserved. 
'''
from typing import List
import math
from csinference.core.util import Instance

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
			elif j==i+1:
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


class FunctionCost():
	def __init__(self, instance : Instance) -> None:
		self.cpu = instance.cpu
		self.mem = instance.mem
		self.gpu = instance.gpu
		self.cpu_cost = 0.000127
		self.mem_cost = 0.0000127
		self.gpu_cost = 0.00011
		self.invocation_cost = 0.009 / 10000


	def cost(self, duration: float, batch: int) -> float:
		if self.gpu is None:
			gpu = 0
		else:
			gpu = self.gpu
			duration = math.ceil(duration)
		return (self.invocation_cost + (self.cpu * self.cpu_cost + self.mem * self.mem_cost + gpu * self.gpu_cost) * duration) / batch
	
	def cost_with_distribution(self, time_out : float, rps : float, batch_max : int, lat_cal, instance) -> float:
		if batch_max == 1:
			return self.cost(lat_cal.lat_avg(instance, 1), 1)
		p = batch_distribution(rps, batch_max, time_out)
		c = 0.0
		for i in range(batch_max):
			c += self.cost(lat_cal.lat_avg(instance, i + 1), i+1) * p[i]
		return c
			
		


