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
from Puzzle_CAM.inference_cam_generation_Puzzle_CAM import Puzzle_CAM_inference
from Standard_classifier.inference_cam_generation_Standard_classifier import Std_classifier_inference



os.environ['CUDA_VISIBLE_DEVICES'] = '0'



# config = {
#     'seed': 42,
#     'num_workers': 4,
#     'architecture': 'resnet50',
#     'mode': 'normal',
#     'batch_size': 32,
#     'max_epoch': 25,
#     'lr': 0.1,
#     'wd': 1e-4,
#     'nesterov': True,
#     'image_size': 512,
#     'print_ratio': 0.1,
#     'augment': 'colorjitter',
#     're_loss_option': 'masking',
#     're_loss': 'L1_Loss',
#     'alpha_schedule': 0.0,
#     'glob_alpha': 2.0,
#     'beta_schedule': 0.0,
#     'glob_beta': 6.0,
#     'num_pieces': 4,
#     'loss_option': 'cl_pcl_re',
#     'imagenet_mean': [0.485, 0.456, 0.406],
#     'imagenet_std': [0.229, 0.224, 0.225],
#     'level' : 'feature',  # 'feature'  'cam'
#     'optimizer': 'adam' # 'SGD'  'adam'
# }

config = {
    'seed': 42,
    'architecture': 'resnet50',
    'mode': 'normal',
    'image_size': 512,
    # 'tag' : 'GradCAM_seruso_512_no_bg_three_classes_15_epochs_resnet50_batch32_adam', # 'POFCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0__beta_6.0_0.0', # GradCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_SGD # Puzzle_CAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0
    # 'tag' : 'Puzzle_CAM_seruso_512_no_bg_three_classes_15_epochs_resnet50_batch32_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0', # 'POFCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0__beta_6.0_0.0', # GradCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_SGD # Puzzle_CAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0
    'tag' : 'POFCAM_seruso_512_no_bg_three_classes_15_epochs_resnet50_batch32_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0__beta_6.0_0.0', # 'POFCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0__beta_6.0_0.0', # GradCAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_SGD # Puzzle_CAM_seruso_512_no_bg_three_classes_40_epochs_resnet50_batch16_cl_pcl_re_masking_only_lateral_4_pieces_new_dataset_alpha_2.0_0.0
    'scales' : '0.2, 0.5, 1.0, 2.0, 4.0, 6.0',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'with_flows': False,
    'with_mask': False
}


num_workers = 4


dataset_dir_main = '../../Datasets/SERUSO_DATASETS/main_dataset/Before_after_no_backgrounds/' # main_dataset/Before_after_no_backgrounds/' # new_5000/three_classes_5000/ #    Before_after_dataset_1240
flow_dir_main = '../../Datasets/SERUSO_DATASETS/main_dataset/optical_flows/' # main_dataset/optical_flows/' # new_5000/optical_flows_5000/ #    Before_after_dataset_1240

dataset_dir_5000 = '../../Datasets/SERUSO_DATASETS/new_5000/bef_aft_5000/'# three_classes_5000/'
flow_dir_5000 = '../../Datasets/SERUSO_DATASETS/new_5000/optical_flows_5000/'


# batch_size = config['batch_size']
image_size = 512

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

test_transform = transforms.Compose([
    
    transforms.Resize(input_size),
    Normalize(imagenet_mean, imagenet_std),
])


# Remake the test tranform, right now using no augmentation

train_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, classes_subfolders = ['before'], return_img_path = True, dstype = 'training', transform = test_transform, augment = False)
val_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, classes_subfolders = ['before'], return_img_path = True, dstype = 'validation', transform = test_transform, augment = False)


normalize_for_cams = True

class_names = np.asarray(train_dataset.class_names)

print("\n\nStandard\n\n")

folder_path = 'experiments/GradCAM/models/'  # Replace with your folder path

filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for filen in filenames:
    config['tag'] = filen
    print(filen)
    std_classifier = Std_classifier_inference(config, train_dataset, sam_enhance = True)
    std_classifier.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)

print("\n\nStandard_validation\n\n")

folder_path = 'experiments/GradCAM/models/'  # Replace with your folder path

filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for filen in filenames:
    config['tag'] = filen
    print(filen)
    std_classifier = Std_classifier_inference(config, val_dataset, sam_enhance = True)
    std_classifier.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)

# print("\n\nPuzzle\n\n")

# folder_path = 'experiments/Puzzle-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
#     puzzle_cam = Puzzle_CAM_inference(config, train_dataset, sam_enhance = False)
#     puzzle_cam.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)


# print("\n\nPuzzle_validation\n\n")

# folder_path = 'experiments/Puzzle-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
#     puzzle_cam = Puzzle_CAM_inference(config, val_dataset, sam_enhance = False)
#     puzzle_cam.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)


# print("\n\nPOF\n\n")
# folder_path = 'experiments/POF-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
        
#     pof_cam = POF_CAM_inference(config, train_dataset, sam_enhance = True)
#     pof_cam.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)

# print("\n\nPOF_validation\n\n")
# folder_path = 'experiments/POF-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
        
#     pof_cam = POF_CAM_inference(config, val_dataset, sam_enhance = True)
#     pof_cam.make_all_cams(save_mask = True, visualize = False, norm = normalize_for_cams, max_item = 10000)


print("\n\nEND\n\n")