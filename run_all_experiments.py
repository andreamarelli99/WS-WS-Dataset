import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import seruso_datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from general_utils.augment_utils import *
from POF_CAM.train_classification_with_POF_CAM import POF_CAM
from Puzzle_CAM.train_classification_with_Puzzle_CAM import Puzzle_CAM
from Standard_classifier.train_classification_with_standardClassifier import standardClassifier

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_dataloaders(config):
    num_workers = config['num_workers']

    batch_size = config['batch_size']
    image_size = config['image_size']
    augment = config['augment']   

    imagenet_mean = config['imagenet_mean'] # [0.485, 0.456, 0.406]
    imagenet_std = config['imagenet_std'] # [0.229, 0.224, 0.225]


    input_size = (image_size, image_size)
    normalize_fn = Normalize(imagenet_mean, imagenet_std)

    dataset_dir = config['dataset_dir']
    flow_dir = config['flow_dir']


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

    train_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir, flow_root = flow_dir, dstype = 'training', transform = train_transform, augment = False)
    val_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir, flow_root = flow_dir, dstype = 'validation', transform = test_transform, augment = False)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True, drop_last=True)
    validation_loader = DataLoader(val_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True, drop_last=True)

    class_names = np.asarray(train_dataset.class_names)

    return train_loader, validation_loader, class_names

def main():
    parser = argparse.ArgumentParser(description="Train a classifier with different training methods.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--method", type=str, choices=["standard", "puzzle_cam", "pof_cam"], required=True, help="Choose the training method.")
    parser.add_argument("--cuda_devices", type=str, default="0", help="Comma-separated list of CUDA device IDs to use (e.g., '0,1').")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    config = load_config(args.config)
    
    train_loader, validation_loader, class_names = get_dataloaders(config)
    
    if args.method == "standard":
        model = standardClassifier(config, train_loader, validation_loader)
    elif args.method == "PuzzleCAM":
        model = Puzzle_CAM(config, train_loader, validation_loader)
    elif args.method == "POF_CAM":
        model = POF_CAM(config, train_loader, validation_loader)
    
    model.train()
    
if __name__ == "__main__":
    main()
