# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import numpy as np
from PIL import Image
from torch.distributions import Categorical
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from .datasets import get_dataset, name2benchmark
import pdb
import utils
from utils import ActualSequentialSampler
from RandAugment import RandAugment

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

sys.path.append('../')


class DatasetWithIndicesWrapper(torch.utils.data.Dataset):

	def __init__(self, name, data, targets, transforms, base_transforms):
		self.name = name
		self.data = data
		self.targets = targets
		self.transforms = transforms
		self.base_transforms = base_transforms
		self.rand_aug_transforms = copy.deepcopy(self.base_transforms)
		self.committee_size = 1
		self.ra_obj = RandAugment(1, 2.0)

		if self.name == 'mnist': # See if this can be dropped entirely without losing performance
			self.transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
			self.base_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
			self.rand_aug_transforms.transforms.insert(0, torchvision.transforms.Grayscale(num_output_channels=1))
		
		self.rand_aug_transforms.transforms.insert(0, self.ra_obj)

	def __len__(self):
		return len(self.targets)
	
	

	def __getitem__(self, index):

		data, target = self.data[index], self.targets[index]

		if self.name == 'mnist':
			data = Image.fromarray(data.numpy(), mode='L')
			data = data.convert('RGB')
		elif self.name == 'svhn':
			data = Image.fromarray(np.transpose(data, (1, 2, 0)))
		else:
			data = utils.default_loader(self.data[index])

		rand_aug_lst = [self.rand_aug_transforms(data) for _ in range(self.committee_size)]
		return (self.transforms(data),self.transforms(data), self.base_transforms(data), rand_aug_lst), int(target), self.data[index], int(index)
	
	def add_item(self, addition):
		lst = list(self.data)
		for (path, target) in addition:
			target = torch.tensor(target)
			target = target.unsqueeze(0)
			lst.append(path)
			self.targets = torch.cat((self.targets,target),0)
		self.data = np.asarray(lst)
		return self.data, self.targets

	def remove_item(self, reduced):
		reduced = reduced.astype('int64')
		self.data = np.delete(self.data, reduced, axis=0)
		self.targets = np.delete(self.targets, reduced, axis=0)
		return self.data, self.targets


class UDADataset:
	"""
	Dataset Class
	"""
	def __init__(self, name, LDS_type, is_target=False, img_dir='data', valid_ratio=0.2, batch_size=128):
		self.name = name
		self.LDS_type = LDS_type
		self.is_target = is_target
		self.img_dir = img_dir
		self.valid_ratio = valid_ratio
		self.batch_size = batch_size
		self.train_size = None
		self.train_dataset = None
		self.num_classes = None
		self.train_transforms = None
		self.test_transforms = None

	def get_num_classes(self):
		return self.num_classes

	def long_tail_train(self, key):
		"""Manually long-tails target training set by loading checkpoint

		Args:
			key: Identifier of long-tailed checkpoint
		"""
		ixs = pickle.load(open(os.path.join('checkpoints', '{}.pkl'.format(key)), 'rb'))
		self.train_dataset.data = self.train_dataset.data[ixs]
		self.train_dataset.targets = torch.from_numpy(np.array(self.train_dataset.targets)[ixs])

	def get_dsets(self):
		"""Generates and return train, val, and test datasets

		Returns:
			Train, val, and test datasets.
		"""
		data_class = get_dataset(self.name, self.name, self.img_dir, self.LDS_type, self.is_target)

		self.num_classes, train_dataset, val_dataset, test_dataset, self.train_transforms, self.test_transforms = data_class.get_data()

		self.train_dataset = DatasetWithIndicesWrapper(self.name, train_dataset.data, train_dataset.targets, self.train_transforms, self.test_transforms)
		self.val_dataset = DatasetWithIndicesWrapper(self.name, val_dataset.data, val_dataset.targets, self.test_transforms, self.test_transforms)
		self.test_dataset = DatasetWithIndicesWrapper(self.name, test_dataset.data, test_dataset.targets, self.test_transforms, self.test_transforms)

		return self.train_dataset, self.val_dataset, self.test_dataset

	def get_loaders(self, shuffle=False, num_workers=8, class_balance_train=False, uncertainty_sampling=False, net=None):
		"""Constructs and returns dataloaders

		Args:
			shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
			num_workers (int, optional): Number of threads. Defaults to 4.
			class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.

		Returns:
			Train, val, test dataloaders, as well as selected indices used for training
		"""
		if not self.train_dataset: self.get_dsets()
		num_train = len(self.train_dataset)
		self.train_size = num_train
		
		benchmark = name2benchmark[self.name]

		if benchmark in ["MNIST", "SVHN"]:
			indices = list(range(num_train))
			split = int(np.floor(self.valid_ratio * num_train))
			if shuffle == True: np.random.shuffle(indices)
			train_idx, valid_idx = indices[split:], indices[:split]
			valid_sampler = SubsetRandomSampler(valid_idx)		
		elif benchmark in ["DomainNet", "OfficeHome", "VisDA2017", "Office31"]:
			# pdb.set_trace()
			train_idx = np.arange(len(self.train_dataset))
			valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))		
		else:
			raise NotImplementedError

		if class_balance_train:
			self.train_dataset.data = [self.train_dataset.data[idx] for idx in train_idx]
			self.train_dataset.targets = self.train_dataset.targets[train_idx]
			if hasattr(self.train_dataset, 'targets_copy'): self.train_dataset.targets_copy = self.train_dataset.targets_copy[train_idx]
			
			if uncertainty_sampling:
				entropy_list, sdsd = [], []
				net.eval()
				loader_ent = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=num_workers)
				with torch.no_grad():
					for (data, _, _), labels, _, indices1 in loader_ent:
						data, labels = data.cuda(), labels.cuda()
						log_logits, _= net(data, with_emb=True)
						ent_closed = 1*Categorical(logits = log_logits).entropy()#.mean(0)
						entropy_list.append(ent_closed)
						sdsd.append(indices1)
					entropy_list = torch.cat(entropy_list).cpu()
					entropy_list = entropy_list/entropy_list.max()
					sdsd = torch.cat(sdsd)	

					
			targets = copy.deepcopy(self.train_dataset.targets)
			if isinstance(targets, torch.Tensor): targets = targets.numpy()
			count_dict = Counter(targets)

			count_dict_full = { lbl: 0 for lbl in range(self.num_classes)}
			for k, v in count_dict.items(): count_dict_full[k] = v

			count_dict_sorted = {k: v for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])}
			class_sample_count = np.array(list(count_dict_sorted.values()))
			class_sample_count = class_sample_count / class_sample_count.max()
			class_sample_count += 1e-8

			weights = 1 / torch.Tensor(class_sample_count)
			if not uncertainty_sampling:
				
				sample_weights = [weights[l] for l in targets]
			else:
				# sample_weights = [weights[l]*entropy_list[l] for l in targets]
				sample_weights = [weights[l] + entropy_list[l] for l in targets]

			sample_weights = torch.DoubleTensor(np.array(sample_weights))
			train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
		else:
			# if not active_sampling:
			train_sampler = SubsetRandomSampler(train_idx)
			# else:
			# 	train_sampler = ActualSequentialSampler(train_idx)
		train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
												   batch_size=self.batch_size, num_workers=num_workers)
		val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, \
												 batch_size=self.batch_size)
		test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
		return train_loader, val_loader, test_loader, train_idx


# def get_loaders(train_dataset, val_dataset, test_dataset, name, num_classes, batch_size,shuffle=False, num_workers=0, class_balance_train=False, active_sampling=False):
# 		"""Constructs and returns dataloaders

# 		Args:
# 			shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.
# 			num_workers (int, optional): Number of threads. Defaults to 4.
# 			class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.

# 		Returns:
# 			Train, val, test dataloaders, as well as selected indices used for training
# 		"""
		
# 		num_train = len(train_dataset)
# 		train_size = num_train
		
# 		benchmark = name2benchmark[name]

# 		if benchmark in ["MNIST", "SVHN"]:
# 			indices = list(range(num_train))
# 			split = int(np.floor(0.2 * num_train))
# 			if shuffle == True: np.random.shuffle(indices)
# 			train_idx, valid_idx = indices[split:], indices[:split]
# 			valid_sampler = SubsetRandomSampler(valid_idx)		
# 		elif benchmark in ["DomainNet", "OfficeHome", "VisDA2017", "Office31"]:
# 			# pdb.set_trace()
# 			train_idx = np.arange(len(train_dataset))
# 			valid_sampler = SubsetRandomSampler(np.arange(len(val_dataset)))		
# 		else:
# 			raise NotImplementedError

# 		if class_balance_train:
# 			train_dataset.data = [train_dataset.data[idx] for idx in train_idx]
# 			train_dataset.targets = train_dataset.targets[train_idx]
# 			if hasattr(train_dataset, 'targets_copy'): train_dataset.targets_copy = train_dataset.targets_copy[train_idx]

# 			targets = copy.deepcopy(train_dataset.targets)
# 			if isinstance(targets, torch.Tensor): targets = targets.numpy()
# 			count_dict = Counter(targets)

# 			count_dict_full = { lbl: 0 for lbl in range(num_classes)}
# 			for k, v in count_dict.items(): count_dict_full[k] = v

# 			count_dict_sorted = {k: v for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])}
# 			class_sample_count = np.array(list(count_dict_sorted.values()))
# 			class_sample_count = class_sample_count / class_sample_count.max()
# 			class_sample_count += 1e-8

# 			weights = 1 / torch.Tensor(class_sample_count)
# 			sample_weights = [weights[l] for l in targets]
# 			sample_weights = torch.DoubleTensor(np.array(sample_weights))
# 			train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
# 		else:
# 			if not active_sampling:
# 				train_sampler = SubsetRandomSampler(train_idx)
# 			else:
# 				train_sampler = ActualSequentialSampler(train_idx)

# 		train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, \
# 												   batch_size=batch_size, num_workers=num_workers)
# 		val_loader = torch.utils.data.DataLoader(val_dataset, sampler=valid_sampler, \
# 												 batch_size=batch_size)
# 		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
# 		return train_loader, val_loader, test_loader, train_idx
