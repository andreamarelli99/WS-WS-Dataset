import os

import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

import seruso_datasets

from torchvision import transforms
from torch.utils.data import DataLoader

from general_utils.augment_utils import *

from POF_CAM.inference_cam_generation_POFCAM import POF_CAM_inference
from Puzzle_CAM.inference_cam_generation_Puzzle_CAM import Puzzle_CAM_inference
from Standard_classifier.inference_cam_generation_Standard_classifier import Std_classifier_inference



os.environ['CUDA_VISIBLE_DEVICES'] = '1'




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
    'with_flows': True,
    'with_mask': True
}


root_with_temporal_labels = '../../Datasets/SERUSO_DATASETS/test_set'

batch_size = 64
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

normalize_for_cams = True


test_dataset = seruso_datasets.SerusoTestDataset(img_root = root_with_temporal_labels, classes_subfolders = ['after'], transform= test_transform, with_flow = config['with_flows'], with_mask = config['with_mask'])

print("\n\nStandard_with_sam\n\n")

folder_path = 'experiments/GradCAM/models/'  # Replace with your folder path

filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for filen in filenames:
    config['tag'] = filen
    print(filen)
    std_classifier = Std_classifier_inference(config, test_dataset, sam_enhance = True)
    std_classifier.make_all_cams(save_mask = False, visualize = False, norm = normalize_for_cams, max_item = 100000)


print("\n\nStandard\n\n")

folder_path = 'experiments/GradCAM/models/'  # Replace with your folder path

filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for filen in filenames:
    config['tag'] = filen
    print(filen)
    std_classifier = Std_classifier_inference(config, test_dataset, sam_enhance = False)
    std_classifier.make_all_cams(save_mask = False, visualize = False, norm = normalize_for_cams, max_item = 100000)

# print("\n\nPuzzle\n\n")

# folder_path = 'experiments/Puzzle-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
#     puzzle_cam = Puzzle_CAM_inference(config, test_dataset, sam_enhance = True)
#     puzzle_cam.make_all_cams(save_mask = False, visualize = False, norm = normalize_for_cams, max_item = 100000)

# print("\n\nPOF\n\n")
# folder_path = 'experiments/POF-CAM/models/'  # Replace with your folder path

# filenames = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# for filen in filenames:
#     config['tag'] = filen
#     print(filen)
        
#     pof_cam = POF_CAM_inference(config, test_dataset, sam_enhance = True)
#     pof_cam.make_all_cams(save_mask = False, visualize = False, norm = normalize_for_cams, max_item = 100000)