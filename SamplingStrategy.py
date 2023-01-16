import torch
import numpy as np
import sys
sys.path.append('..')
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from scipy import stats
import torch.optim as optim
import time
import pdb
from sklearn.cluster import KMeans
from .strategy import Strategy
from util.losses import softmax_mse_loss, softmax_kl_loss, adentropy, Margin_loss
from util.utils import *
from functions import SupConLoss, ReverseLayerF, kd_loss, prt_feat_update, prt_logits_update, gce, tsne, prt_emd_update, prt_img_update 
from torch import nn
# from utils.utils import log_set, save_model
from utils.loss import ova_loss, open_entropy
# from utils.lr_schedule import inv_lr_scheduler
import utils.lr_schedule as lr_schedule
# from utils.defaults import get_dataloaders, get_models, get_models_clip, get_dataloaders_new, get_models_clip_da, get_models_clip_direct
# from eval import test, test_clip, test_clip_old, test_dc, test_clip_dc
import os
from torch.distributions import Categorical
torch.backends.cudnn.benchmark=True
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
import numpy as np
from termcolor import colored
from geomloss import SamplesLoss 
from pathlib import Path
import GP_new1
from GP_new1 import GPStruct
# import clip
import argparse

CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'mobile_phone', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
TGT_CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'mobile_phone', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
TEST_TGT_CLASSES = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', 'headphones', 'keyboard', 'laptop_computer', 'letter_tray', 'mobile_phone', 'mobile_phone', 'mouse', 'mug', 'paper_notebook', 'pen', 'phone', 'printer', 'projector', 'punchers', 'ring_binder', 'ruler', 'scissors', 'speaker', 'stapler', 'tape_dispenser', 'trash_can']
DOMAIN_CLASSES = ['source', 'target']
ACTIVE_DOMAIN_CLASSES = ['source', 'target', 'selected samples']

# loss_domain = torch.nn.NLLLoss().cuda()
loss_domain = nn.CrossEntropyLoss().cuda()
alpha = 0.1
output_dir = "results/tsne plots/dif_losses/"


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def domain_da(D, img_feat, domain_label):
    reverse_feature = ReverseLayerF.apply(img_feat, alpha)
    domain_output = D(reverse_feature)
    err_s_domain = loss_domain(domain_output, domain_label)
    return err_s_domain

def image_classification_test(target_test_loader, model):
    start_test = True
    with torch.no_grad():    
        iter_test = iter(target_test_loader)
        for i in range(len(target_test_loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels.cuda()
            feature_out, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float().cpu()
                all_feature = feature_out.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_feature = torch.cat((all_feature, feature_out.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float().cpu() ), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, all_output.numpy(), predict.numpy(), all_label.numpy(), all_feature.numpy()

class ActiveSampling(Strategy):
    def __init__(self, source, target_train, target_test, idxs_lb, net, args, D):
        super(ActiveSampling, self).__init__(source, target_train, target_test, idxs_lb, net, args, D)
        global device;device = torch.device("cuda:" + self.args.DEVICE if self.args.USE_CUDA else "cpu")
        self.clf = self.net.to(device)
        self.D = self.D.to(device)
        self.gp_struct = GPStruct(self.args,len(self.source.dataset), 1, self.args.BATCH_SIZE, feat_dim=256, 
		kernel_type='Linear')

        self.selected_samples = list()
        self.kwargs = {'num_workers': self.args.NUM_WORK, 'pin_memory': True} if self.args.USE_CUDA else {}
        self.epc = self.args.NUM_ITERATIONS / self.args.EPOCH
        self.sampling_rounds = [self.epc*i for i in [10,12,14,16,18]]

    def sdm_active(self, inds, candidate_dataset, aim_dataset):
        
        active_samples = list()
        # self.selected_samples = list()
        for i in inds:
            sample, target, path, _ = candidate_dataset[i]
            active_samples.append([path,target])
            self.selected_samples.append((sample, target))
        self.active_loader = DataLoader(self.selected_samples, batch_size=self.args.BATCH_SIZE,**self.kwargs)
        # self.idxs_lb[inds] = True
        aim_dataset.add_item(active_samples)
        candidate_dataset.remove_item(inds)


    def clue_query(self, num_active):
        self.clf.eval()
		
        joint_loader = zip(self.source, self.target_train)
        s_feats = torch.zeros([1, 256]).cuda()
        s_labels = torch.zeros([1, 1]).cuda()
        t_feats = torch.zeros([1, 256]).cuda()
        t_preds = torch.zeros([1, 1]).cuda()
        t_logits = torch.zeros([1, self.args.DATA_CLASS]).cuda()
        with torch.no_grad():
            # for (img_s, ys, _, _), (img_t, yt, _, idxst) in joint_loader:
            for img_t, yt, _, idxst in self.target_train:
                # target samples
                img_t, yt = Variable(img_t.to(device)), Variable(yt.to(device))
                img_feat_t, emd_feat_t = self.clf(img_t)
                probs_t = F.softmax(emd_feat_t, dim=1)
                value, predicted = torch.max(probs_t, 1)
                t_feats = torch.cat([t_feats, img_feat_t.data],0)
                t_preds = torch.cat([t_preds, predicted.unsqueeze(1).data],0)
                t_logits = torch.cat([t_logits, emd_feat_t.data],0)
            for img_s, ys, _, _ in self.source:
                #source samples
                img_s, ys = Variable(img_s.to(device)), Variable(ys.to(device))
                img_feat_s, emd_feat_s = self.clf(img_s)
                probs_s = F.softmax(emd_feat_s, dim=1)
                value_s, predicted_s = torch.max(probs_s, 1)
               
                # pdb.set_trace()
                s_feats = torch.cat([s_feats, img_feat_s.data],0)
                s_labels = torch.cat([s_labels, ys.unsqueeze(1).data],0)
                
        s_labels = s_labels[1:].squeeze(1)
        t_preds = t_preds[1:].squeeze(1)
        t_logits = t_logits[1:]
        src_gp_feat = s_feats[1:]
        tgt_gp_feat = t_feats[1:]
        	
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
        # pdb.set_trace()
        # sort_idxs = dists.sort(descending=False)[1].cpu().numpy()

        # chosen = sort_idxs[:num_active]
        return np.array(q_idxs)

    