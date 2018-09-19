import time

import os
import tensorflow as tf
import pickle as pkl
import numpy as np

from gcn.utils import *


# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

DATASET = '../../data/glove_res50/'
MODEL = 'dense'
LEARNING_RATE = 0.001
SAVE_PATH = '../output/'
EPOCHS = 350
HIDEN1 = 2048
HIDEN2 = 2048
HIDEN3 = 1024
HIDEN4 = 1024
HIDEN5 = 512
DROPOUT = 0.5
WEIGHT_DECAY = 5e-4
EARLY_STOP = 10
MAX_DEGREE = 3
GPU = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU

use_trainval = True
feat_suffix = 'allx_dense'

# Load data
adj, features, y_train, y_val, y_trainval, train_mask, val_mask, trainval_mask = \
        load_data_vis_multi(DATASET, use_trainval, feat_suffix)


