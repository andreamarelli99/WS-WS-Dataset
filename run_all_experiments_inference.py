import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import seruso_datasets

from torchvision import transforms
from torch.utils.data import DataLoader

from general_utils.augment_utils import *

from POF_CAM.inference_cam_generation_POFCAM import POF_CAM_inference
from Puzzle_CAM.train_classification_with_Puzzle_CAM import Puzzle_CAM
from Standard_classifier.train_classification_with_standardClassifier import standardClassifier

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = {
    'seed': 42,
    'architecture': 'resnet50',
    'mode': 'normal',
    'image_size': 512,
    'tag' : 'PuzzleFlowCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0__beta_6.0_0.0',
    'scales' : '0.2, 0.5, 1.0, 2.0, 4.0, 6.0',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'with_flows': True,
    'with_mask': True
}


root_with_temporal_labels = '../../Datasets/SERUSO_DATASETS/test_set'

batch_size = 16
image_size = 512

augment = 'colorjitter' #'colorjitter'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

test_transform = transforms.Compose([
    
    transforms.Resize(input_size),
    Normalize(imagenet_mean, imagenet_std),
])


test_dataset = seruso_datasets.SerusoTestDataset(img_root = root_with_temporal_labels, classes_subfolders = ['before'], transform= test_transform, with_flow = config['with_flows'], with_mask = config['with_mask'])

pof_cam = POF_CAM_inference(config, test_dataset)
pof_cam.make_all_cams()