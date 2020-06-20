import os
import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.mkl import MTLearner
from dataloader.dataset_loader import DatasetLoader as Dataset

class MetaTrainer(object):
    def __init__(self, args):
        log_dir = './logs/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        
        meta_dir = os.path.join(log_dir, 'meta')
        if not os.path.exists(meta_dir):
            os.mkdir(meta_dir)

        save_path1 = '_'.join([args.data, 'MTL'])
        save_path2 = str(args.show) + 'shot_' + str(args.way) + 'way_'