# -*- coding: utf-8 -*-
import sys
from tqdm import tqdm
import random
import copy

import numpy as np
from sklearn.metrics import confusion_matrix
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .solver import register_solver
sys.path.append('../../')
import utils
from utils import prt_img_update, SupConLoss, prt_logits_update, kd_loss, gce, linear_rampup
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from collections import Counter
from torch.distributions import Categorical

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)
TGT_CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'mobile_phone', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']

lambda_u = 100
ema_decay = 0.999
T = 0.5
alpha = 0.75

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        # print("in loss",outputs_x.size(0),targets_x.size(0))
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch)

class BaseSolver:
	def __init__(self, net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args):
		self.net = net
		
		self.src_loader = src_loader
		self.tgt_loader = tgt_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.device = device	
		self.ema_model = copy.deepcopy(net.to(self.device))
		self.num_classes = num_classes
		self.args = args
		self.current_step = 0
		self.param_lr_c = []
		for param_group in self.tgt_opt.param_groups:
			self.param_lr_c.append(param_group["lr"])
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
		for param in self.ema_model.parameters():
			param.detach_()
		self.prototype_feat = F.normalize(torch.rand((len(TGT_CLASSES), 2048), dtype=torch.float), dim=1).cuda()
		self.prototype_logits = F.normalize(torch.rand((len(TGT_CLASSES), len(TGT_CLASSES)), dtype=torch.float), dim=1).cuda()

	def lr_step(self):
		"""
		Learning rate scheduler
		"""
		if self.args.optimizer == 'SGD':
			self.tgt_opt = utils.inv_lr_scheduler(self.param_lr_c, self.tgt_opt, self.current_step, init_lr=self.args.lr)

	def solve(self, epoch):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb
	
	def active_sampling(self, inds, candidate_dataset, aim_dataset):
        
		active_samples = list()
		for i in inds:
			(data, _, _, _), target, path, _ = candidate_dataset[i]
			active_samples.append((path, target))
		# self.idxs_lb[inds] = True
		aim_dataset.add_item(active_samples)
		candidate_dataset.remove_item(inds)

	def pseudo_active(self, choosen, pseudo_chosen, train_target_data, train_source_data):
		active_samples = list()

		inds = choosen
		pseduo_inds = pseudo_chosen['idx']
		pseduo_target = pseudo_chosen['pseudo_gt']

		for i in inds:
			(data, _, _), target, path, _ = train_target_data[i]
			active_samples.append((path, target))

		for count, value in enumerate(pseduo_inds):
			(data, _, _), target, path, _ = train_target_data[value]
			active_samples.append((path, int(pseduo_target[count])))

		net_inds = np.concatenate((inds,pseduo_inds))
		train_source_data.add_item(active_samples)
		train_target_data.remove_item(net_inds)

	def pseudo_active_v1(self, pseudo_chosen, train_target_data, train_source_data):
		active_samples = list()

		pseduo_inds = pseudo_chosen['idx']
		pseduo_target = pseudo_chosen['pseudo_gt']

		for count, value in enumerate(pseduo_inds):
			(data, _, _, _), target, path, _ = train_target_data[value]
			active_samples.append((path, int(pseduo_target[count])))

		train_source_data.add_item(active_samples)
		train_target_data.remove_item(pseduo_inds)

	def random_query(self, num_active):
		print("Doing Random Sampling")
		chosen = np.random.randint(low=0, high=len(self.tgt_loader.dataset), size=(num_active,))
		return chosen

	def clue_query(self, num_active):
		print("Doing CLUE Sampling")
		print(len(self.tgt_loader.dataset))
		self.net.eval()
		indices = []
		encoder_feat, target_gt = [], []
		t_feats = torch.zeros([1, 2048]).cuda()
		t_logits = torch.zeros([1, self.num_classes]).cuda()
		with torch.no_grad():
			for (x, _, _, _), y, _, idxs in self.tgt_loader:	
				x, y = x.to(self.device), y.to(self.device)
				logits, img_feat = self.net(x, with_emb=True)
				indices.append(idxs)
				target_gt.append(y)
				t_feats = torch.cat([t_feats, img_feat.data],0)
				t_logits = torch.cat([t_logits, logits.data],0)
				

			target_gt = torch.cat(target_gt)
			indices = torch.cat(indices)
			pdb.set_trace()
			t_logits = t_logits[1:].squeeze(1)

			tgt_pen_emb = t_feats[1:].cpu().numpy()
			tgt_scores = nn.Softmax(dim=1)(t_logits / 1)
			tgt_scores += 1e-8
			sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()
			
			# Run weighted K-means over embeddings
			km = KMeans(num_active)
			km.fit(tgt_pen_emb, sample_weight=sample_weights)
			
			# pdb.set_trace()
			# Find nearest neighbors to inferred centroids
			dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
			sort_idxs = dists.argsort(axis=1)
			q_idxs = sort_idxs[:, 0][:num_active]

			chosen = indices[q_idxs].cpu().numpy()
		return chosen

	def pgpq_query(self, num_active, coeff):
		print("Doing pgpq Sampling")
		self.net.eval()
		conf_feat, conf_logit, pl_feat, pl_logit, indices = [], [], [], [], []
		encoder_feat, target_gt, entropy_list = [], [], []
		with torch.no_grad():
			for (x, _, _, _), y, _, idxs in self.tgt_loader:	
				x, y = x.to(self.device), y.to(self.device)
				logits, img_feat = self.net(x, with_emb=True)
				value, predicted = torch.max(F.softmax(logits.detach(), 1), 1)
				ent_closed = 1*Categorical(logits = logits.detach()).entropy()#.mean(0)
				entropy_list.append(ent_closed)
				indices.append(idxs)
				conf_logit.append(value)
				pl_logit.append(predicted)
				img_feat = img_feat/img_feat.norm(dim=-1, keepdim=True)
				feat_logits = torch.matmul(img_feat, self.prototype_feat.T)
				feat_value, feat_predicted = torch.max(F.softmax(feat_logits.detach(), 1), 1)
				conf_feat.append(feat_value)
				pl_feat.append(feat_predicted)
				encoder_feat.append(img_feat)

				target_gt.append(y)
			entropy_list = torch.cat(entropy_list).cpu()
			target_gt = torch.cat(target_gt)
			indices = torch.cat(indices).cuda()
			conf_logit = torch.cat(conf_logit)
			pl_logit = torch.cat(pl_logit)
			conf_feat = torch.cat(conf_feat)
			pl_feat = torch.cat(pl_feat)
			
		
			# _, conf_sorted_idx = torch.sort(conf_feat, descending=True)
			_, conf_sorted_idx = torch.sort(conf_feat*conf_logit, descending=True)
			# conf_sorted, conf_sorted_idx = torch.sort(entropy_list, descending=False)
			pseduo_active = int(num_active*coeff)
			req_active = 0
			sm_tgt = target_gt[conf_sorted_idx[:pseduo_active]]
			sm_pl_feat = pl_feat[conf_sorted_idx[:pseduo_active]]
			top_acc = ((sm_pl_feat == sm_tgt).float().sum().item()*100)/len(sm_tgt)
			print("top conf_acc: ", top_acc, "total samples:", pseduo_active)

			sm_tgt = target_gt[conf_sorted_idx[-req_active:]]
			sm_pl_feat = pl_feat[conf_sorted_idx[-req_active:]]
			bottom_acc = ((sm_pl_feat == sm_tgt).float().sum().item()*100)/len(sm_tgt)
			print("bottom conf_acc: ", bottom_acc)
			
			chosen = indices[conf_sorted_idx[-req_active:]].cpu().numpy()
			pseudo_chosen = {"idx":indices[conf_sorted_idx[:pseduo_active]].cpu().numpy(), "pseudo_gt":pl_feat[conf_sorted_idx[:pseduo_active]].cpu().numpy()}
			
		return chosen, pseudo_chosen, top_acc, bottom_acc

	def consistency_query(self, num_active):
		print("Doing Consistency Sampling")
		loader_consistency = self.tgt_loader
		loader_consistency.committee_size = self.committee_size
		loader_consistency.dataset.ra_obj.n = self.randaug_n
		loader_consistency.dataset.ra_obj.m = self.randaug_m

		# self.ema_model.eval()
		indices = []
		self.net.eval()
		consistency = np.zeros([len(loader_consistency.dataset)])
		with torch.no_grad():
			for (_, _, _, x), y, _, idxs in loader_consistency:
				probs = np.zeros([len(x), y.size(0),self.num_classes])
				for i, xi in enumerate(x):
					out1, e1 = self.net(xi.to(self.device), with_emb=True)
					prob = torch.softmax(out1, dim=1).cpu()
					probs[i] = prob

				consistency[idxs] = probs.var(0).sum(1)
		q_idxs = consistency.argsort()
		return q_idxs[-num_active:]

	


@register_solver('SENTRY')
class SENTRYSolver(BaseSolver):
	"""
	Implements SENTRY
	"""
	def __init__(self, net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args):
		super(SENTRYSolver, self).__init__(net, src_loader, tgt_loader, train_idx, tgt_opt, device, num_classes, args)
		self.num_classes = args.num_classes
		self.queue_length = 256 	# Queue length for computing target information entropy loss
		
		# Committee consistency hyperparameters
		self.randaug_n = 3			# RandAugment number of consecutive transformations
		self.randaug_m = 2.0		# RandAugment severity
		self.committee_size = 3 	# Committee size		
		self.positive_threshold, self.negative_threshold = (self.committee_size // 2) + 1, \
														   (self.committee_size // 2) + 1  # Use majority voting scheme		 
		# Pass in hyperparams to dataset
		self.tgt_loader.dataset.committee_size = self.committee_size
		self.tgt_loader.dataset.ra_obj.n = self.randaug_n
		self.tgt_loader.dataset.ra_obj.m = self.randaug_m

		self.src_loader.dataset.committee_size = self.committee_size
		self.src_loader.dataset.ra_obj.n = self.randaug_n
		self.src_loader.dataset.ra_obj.m = self.randaug_m
	
	 
	def compute_prf1(self, true_mask, pred_mask):
		"""
		Compute precision, recall, and F1 metrics for predicted mask against ground truth
		"""
		conf_mat = confusion_matrix(true_mask, pred_mask, labels=[False, True])
		p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
		r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
		f1 = (2*p*r) / (p+r+1e-8)
		return conf_mat, p, r, f1

	def solve(self, epoch):
		"""
		Unsupervised Domain adaptation via SENTRY: Selective Entropy Optimization via Committee Consistency
		"""
		
		joint_loader = zip(self.src_loader, self.tgt_loader)
		lambda_src, lambda_unsup, lambda_ent = self.args.lambda_src, self.args.lambda_unsup, self.args.lambda_ent
		print("length of tgt dset:", len(self.tgt_loader.dataset))
		print("length of src dset:", len(self.src_loader.dataset))
		criterion = SemiLoss()
		self.net.train()
		queue = torch.zeros(self.queue_length).to(self.device)
		pointer = 0
		for batch_idx, (((data_s, _, _, data_s_raug), label_s, _, indices_s), ((_, _, data_t_og, data_t_raug), label_t, _, indices_t)) in enumerate(tqdm(joint_loader)):
			
			self.current_step += 1
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_t_og, label_t = data_t_og.to(self.device), label_t.to(self.device)
			
			# Train with source labels
			score_s, img_feat_s= self.net(data_s, with_emb=True)

			self.prototype_feat = prt_img_update(self.prototype_feat, img_feat_s.detach().clone(), label_s.detach().clone(), TGT_CLASSES)
			self.prototype_logits = prt_logits_update(self.prototype_logits, score_s.detach().clone(), label_s.detach().clone(), TGT_CLASSES)
			
			xeloss_src = lambda_src*nn.CrossEntropyLoss()(score_s, label_s)
			loss = xeloss_src

			info_str = "\n[Train SENTRY] Epoch: {}".format(epoch)
			info_str += " Source XE loss: {:.3f}".format(xeloss_src.item())                    


			score_t_og, img_feat_t = self.net(data_t_og, with_emb=True)
			batch_sz = data_t_og.shape[0]
			tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)

			src_preds = score_s.max(dim=1)[1].reshape(-1)

			if pointer+batch_sz > self.queue_length: # Deal with wrap around when ql % batchsize != 0 
				rem_space = self.queue_length-pointer
				queue[pointer: self.queue_length] = (tgt_preds[:rem_space].detach()+1)
				queue[0:batch_sz-rem_space] = (tgt_preds[rem_space:].detach()+1)
			else: 
				queue[pointer: pointer+batch_sz] = (tgt_preds.detach()+1)
			pointer = (pointer+batch_sz) % self.queue_length

			bincounts = torch.bincount(queue.long(), minlength=self.num_classes+1).float() / self.queue_length
			bincounts = bincounts[1:]
			
			log_q = torch.log(bincounts + 1e-12).detach()
			loss_infoent = lambda_unsup * torch.mean(torch.sum(score_t_og.softmax(dim=1) * log_q.reshape(1, self.num_classes), dim=1))
			loss += loss_infoent
			info_str += " Infoent loss: {:.3f}".format(loss_infoent.item())

			score_t_og = self.net(data_t_og).detach()
			tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)

			score_s = self.net(data_s).detach()
			src_preds = score_s.max(dim=1)[1].reshape(-1)
			
			# When pseudobalancing, label_t will correspond to pseudolabels rather than ground truth, so use backup instead
			if self.args.pseudo_balance_target: label_t = self.tgt_loader.dataset.targets_copy[indices_t]
			
			# Compute actual correctness mask for analysis only
			correct_mask_gt = (tgt_preds.detach().cpu() == label_t.cpu())
			
			# SENTRY loss for TARGET SAMPLES
			correct_mask, incorrect_mask = torch.zeros_like(tgt_preds).to(self.device), \
											torch.zeros_like(tgt_preds).to(self.device)					

			score_t_aug_pos, score_t_aug_neg = torch.zeros_like(score_t_og), torch.zeros_like(score_t_og)
			total_score_pos,  total_score_neg = torch.zeros_like(score_t_og), torch.zeros_like(score_t_og)
			for data_t_aug_curr in data_t_raug:
				score_t_aug_curr = self.net(data_t_aug_curr.to(self.device))
				tgt_preds_aug = score_t_aug_curr.max(dim=1)[1].reshape(-1)
				consistent_idxs = (tgt_preds == tgt_preds_aug).detach()
				inconsistent_idxs = (tgt_preds != tgt_preds_aug).detach()
				correct_mask = correct_mask + consistent_idxs.type(torch.uint8)						
				incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

				score_t_aug_pos[consistent_idxs, :] = score_t_aug_curr[consistent_idxs, :]
				score_t_aug_neg[inconsistent_idxs, :] = score_t_aug_curr[inconsistent_idxs, :]
			correct_mask, incorrect_mask = correct_mask>=self.positive_threshold, incorrect_mask>=self.negative_threshold
			
			# Compute some stats
			correct_ratio = (correct_mask).sum().item() / data_t_og.shape[0]				
			incorrect_ratio = (incorrect_mask).sum().item() / data_t_og.shape[0]
			consistency_conf_mat, correct_precision, correct_recall, correct_f1 = self.compute_prf1(correct_mask_gt.cpu().numpy(), \
																									correct_mask.cpu().numpy())
			info_str += "\n {:d} / {:d} consistent ({:.2f}): GT precision: {:.2f}".format(correct_mask.sum(), data_t_og.shape[0], \
																						  correct_ratio, correct_precision)
			
			if correct_ratio > 0.0:
				probs_t_pos = F.softmax(score_t_aug_pos, dim=1)		
				loss_cent_correct = lambda_ent * correct_ratio * -torch.mean(torch.sum(probs_t_pos[correct_mask] * \
																			(torch.log(probs_t_pos[correct_mask] + 1e-12)), 1))
				loss += loss_cent_correct
				info_str += " SENTRY loss (tgt) (consistent): {:.3f}".format(loss_cent_correct.item())
			
			if incorrect_ratio > 0.0:
				probs_t_neg = F.softmax(score_t_aug_neg, dim=1)
				loss_cent_incorrect = lambda_ent * incorrect_ratio * torch.mean(torch.sum(probs_t_neg[incorrect_mask] * \
																				(torch.log(probs_t_neg[incorrect_mask] + 1e-12)), 1))
				loss += loss_cent_incorrect
				info_str += " SENTRY loss (tgt) (inconsistent): {:.3f}".format(loss_cent_incorrect.item())

			# # SENTRY loss for SOURCE SAMPLES
			# correct_mask_gt_s = (src_preds.detach().cpu() == label_s.cpu())
			# correct_mask, incorrect_mask = torch.zeros_like(src_preds).to(self.device), \
			# 								torch.zeros_like(src_preds).to(self.device)					

			# score_s_aug_pos, score_s_aug_neg = torch.zeros_like(score_s), torch.zeros_like(score_s)
			# total_score_pos,  total_score_neg = torch.zeros_like(score_s), torch.zeros_like(score_s)
			# for data_s_aug_curr in data_s_raug:
			# 	score_s_aug_curr = self.net(data_s_aug_curr.to(self.device))
			# 	src_preds_aug = score_s_aug_curr.max(dim=1)[1].reshape(-1)
			# 	consistent_idxs = (src_preds == src_preds_aug).detach()
			# 	inconsistent_idxs = (src_preds != src_preds_aug).detach()
			# 	correct_mask = correct_mask + consistent_idxs.type(torch.uint8)						
			# 	incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

			# 	score_s_aug_pos[consistent_idxs, :] = score_s_aug_curr[consistent_idxs, :]
			# 	score_s_aug_neg[inconsistent_idxs, :] = score_s_aug_curr[inconsistent_idxs, :]
			# correct_mask, incorrect_mask = correct_mask>=self.positive_threshold, incorrect_mask>=self.negative_threshold
			
			# # Compute some stats
			# correct_ratio = (correct_mask).sum().item() / data_s.shape[0]				
			# incorrect_ratio = (incorrect_mask).sum().item() / data_s.shape[0]
			# consistency_conf_mat, correct_precision, correct_recall, correct_f1 = self.compute_prf1(correct_mask_gt_s.cpu().numpy(), \
			# 																						correct_mask.cpu().numpy())
			# info_str += "\n {:d} / {:d} consistent ({:.2f}): GT precision: {:.2f}".format(correct_mask.sum(), data_t_og.shape[0], \
			# 																			  correct_ratio, correct_precision)
			
			# if correct_ratio > 0.0:
			# 	probs_s_pos = F.softmax(score_s_aug_pos, dim=1)		
			# 	loss_cent_correct = lambda_ent * correct_ratio * -torch.mean(torch.sum(probs_s_pos[correct_mask] * \
			# 																(torch.log(probs_s_pos[correct_mask] + 1e-12)), 1))
			# 	loss += loss_cent_correct
			# 	info_str += " SENTRY loss (src) (consistent): {:.3f}".format(loss_cent_correct.item())
			
			# if incorrect_ratio > 0.0:
			# 	probs_s_neg = F.softmax(score_s_aug_neg, dim=1)
			# 	loss_cent_incorrect = lambda_ent * incorrect_ratio * torch.mean(torch.sum(probs_s_neg[incorrect_mask] * \
			# 																	(torch.log(probs_s_neg[incorrect_mask] + 1e-12)), 1))
			# 	loss += loss_cent_incorrect
			# 	info_str += " SENTRY loss (src) (inconsistent): {:.3f}".format(loss_cent_incorrect.item())

			# Backprop
			self.tgt_opt.zero_grad()
			loss.backward()
			self.tgt_opt.step()

			# Learning rate update (if using SGD)
			self.lr_step()

			if batch_idx%10 == 0: print(info_str)
		
