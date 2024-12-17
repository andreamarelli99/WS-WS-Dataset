import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import seruso_datasets

from torchvision import transforms
from torch.utils.data import DataLoader

from POF_CAM.train_classification_with_POF_CAM import POF_CAM
from POF_CAM.POFCAM_utils.augment_utils import *


config = {
    'seed': 42,
    'num_workers': 4,
    'architecture': 'resnet50',
    'mode': 'normal',
    'batch_size': 16,
    'max_epoch': 40,
    'lr': 0.1,
    'wd': 1e-4,
    'nesterov': True,
    'image_size': 512,
    'print_ratio': 0.1,
    'augment': 'colorjitter',
    're_loss_option': 'masking',
    're_loss': 'L1_Loss',
    'alpha_schedule': 0.0,
    'glob_alpha': 2.0,
    'beta_schedule': 0.0,
    'glob_beta': 6.0,
    'num_pieces': 4,
    'loss_option': 'cl_re',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'level' : 'cam'
}

# seed = 42
num_workers = 4


dataset_dir_main = '../../Datasets/SERUSO_DATASETS/main_dataset/Before_after_no_backgrounds/' # main_dataset/Before_after_no_backgrounds/' # new_5000/three_classes_5000/ #    Before_after_dataset_1240
flow_dir_main = '../../Datasets/SERUSO_DATASETS/main_dataset/optical_flows/' # main_dataset/optical_flows/' # new_5000/optical_flows_5000/ #    Before_after_dataset_1240

dataset_dir_5000 = '../../Datasets/SERUSO_DATASETS/new_5000/three_classes_5000/'
flow_dir_5000 = '../../Datasets/SERUSO_DATASETS/new_5000/optical_flows_5000/'

# architecture = 'resnet50'
# mode ='normal' # fix
batch_size = 16
# max_epoch = 40
# lr = 0.1
# wd = 1e-4
# nesterov = True
image_size = 512 # 256
# print_ratio = 0.1
augment = 'colorjitter' #'colorjitter'


# re_loss_option = 'masking'   # 'none', 'masking', 'selection'
# re_loss = 'L1_Loss'          # 'L1_Loss', 'L2_Loss'
# alpha_schedule = 0.0 # 0.50 
# glob_alpha = 2.00

# beta_schedule = 0.0
# glob_beta = 6.00
# num_pieces = 4 # For Puzzle-CAM
# loss_option = 'cl_re'

# 'cl_pcl'
# 'cl_re'
# 'cl_pcl_re'

# level = 'cam'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

train_transforms = [
    transforms.Resize(input_size),
]

if 'colorjitter' in augment:
    train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))

train_transform = transforms.Compose(train_transforms + \
    [
        Normalize(imagenet_mean, imagenet_std),
        RandomCrop(image_size),
        Transpose()
    ]
)
test_transform = transforms.Compose([

    transforms.Resize(input_size),
    Normalize(imagenet_mean, imagenet_std),
    RandomCrop(image_size),
    Transpose()
])


# Remake the test tranform, right now using no augmentation

train_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, dstype = 'training', transform = train_transform, augment = False)
val_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, dstype = 'validation', transform = train_transform, augment = False)

train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True, drop_last=True)

class_names = np.asarray(train_dataset.class_names)



pof_cam = POF_CAM(config, train_loader, validation_loader)

pof_cam.train()







