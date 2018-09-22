from easydict import EasyDict as edict
import time
import os


config = edict()

config.person_name = "yuan.liu"
config.task_name = 'object_classification'
config.project_name = 'zero_sample_classification'
config.root_path = "/home/messor/data_center"

config.data_path = os.path.join(config.root_path, config.task_name, config.project_name)
config.exp = os.path.join(config.root_path, config.task_name, config.project_name, config.person_name)
config.pre_train_model_path = os.path.join(config.root_path, config.task_name, config.project_name, 'models')


now_date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
now_time = time.strftime('%H_%M_%S', time.localtime(time.time()))
config.log_path = os.path.join(config.exp, now_date,  now_time + '_logs')
config.model_path = os.path.join(config.exp, now_date, now_time+'_models')

# add dataset
train_data_list = ['DatasetA_train']
test_data_list = ['DatasetA_test']
config.dataset = edict()
config.dataset.b_g_r_mean = [108.74949161620336, 121.93682191782138, 129.882933212282]
config.dataset.b_g_r_std = [76.27990887272082, 72.2227214967396, 73.87829834580016]
config.dataset.input_resolution = [64, 64]

# test
config.test = edict()
config.test.aug_strategy = edict()
config.test.aug_strategy.normalize = True
config.test.aug_strategy.flip = False
config.test.aug_strategy.reize = False
config.test.aug_strategy.random_rotate = False
config.test.aug_strategy.random_crop = False
config.test.aug_strategy.random_color = False
config.test.aug_strategy.max_rotate_angle = 20

# train detail
config.train = edict()
config.train.split_val = 0.2
config.train.aug_strategy = edict()
config.train.aug_strategy.normalize = True
config.train.aug_strategy.resize = True
config.train.aug_strategy.flip = True
config.train.aug_strategy.random_rotate = True
config.train.aug_strategy.random_crop = False
config.train.aug_strategy.random_color = False
config.train.aug_strategy.max_rotate_angle = 20

# model params
config.support_network = ['ResNet']
config.network = "ResNet#50"
assert config.network.split('#')[0] in config.support_network
config.out_classes = [24, 300]
config.epoch = 60
config.train.batch_size = 60
config.test.batch_size = 1500
config.data_loader_num_workers = 8
config.num_gpu = 1

# optimizer params
config.momentum = 0.0
config.weightDecay = 0.0
config.alpha = 0.99
config.epsilon = 1e-8

config.lr_params = edict()
config.lr_params.lr = 0.00125 * config.num_gpu * config.train.batch_size
config.lr_params.lr_step = [40, 55]
config.lr_params.decay = 0.1
config.lr_params.warm_up = True
config.lr_params.warm_up_lr = 0.001
config.lr_params.warm_up_epoch = 2

config.sample_test = None
config.DEBUG = False