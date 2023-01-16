# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import pdb
import copy
import random
import numpy as np
import pdb
import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import utils
from adapt.solvers.solver import get_solver
import torch.distributed as dist

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, src_dset, train_idx, src_train_idx, model, discriminator, device, args, balanced=False):
		self.dset = dset
		self.src_dset = src_dset
		self.num_classes = self.dset.get_num_classes()
		
		
		self.train_idx = np.array(train_idx)
		self.src_train_idx = np.array(src_train_idx)
		self.model = model
		self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
		
	def query(self, n):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb

	
@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
	"""
	Uniform sampling 
	"""
	def __init__(self, dset, src_dset, train_idx, src_train_idx, model, discriminator, device, args, balanced=False):
		super(RandomSampling, self).__init__(dset, src_dset, train_idx, src_train_idx, model, discriminator, device, args)
		self.dset = dset
		self.balanced = balanced

	def query(self, n):
		aa = np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)
		print(aa)
		pdb.set_trace()
		return aa
