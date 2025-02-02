import os
import argparse
import yaml
import numpy as np
import torch
import seruso_datasets
from torchvision import transforms
from general_utils.augment_utils import *
from POF_CAM.inference_cam_generation_POFCAM import POF_CAM_inference
from Puzzle_CAM.inference_cam_generation_Puzzle_CAM import Puzzle_CAM_inference
from Standard_classifier.inference_cam_generation_Standard_classifier import Std_classifier_inference

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def get_dataloader(config):
    image_size = config['image_size']
    imagenet_mean = config['imagenet_mean']
    imagenet_std = config['imagenet_std']
    dataset_dir = config['dataset_dir']
    classes_subfolders = config['classes_subfolders']
    
    input_size = (image_size, image_size)
    
    test_transform = transforms.Compose([
        
        transforms.Resize(input_size),
        Normalize(imagenet_mean, imagenet_std),
    ])
    
    test_dataset = seruso_datasets.SerusoTestDataset(img_root = dataset_dir, classes_subfolders = classes_subfolders, transform= test_transform, with_flow = config['with_flows'], with_mask = True)
    
    return test_dataset

def main():
    parser = argparse.ArgumentParser(description="Run inference for different CAM methods.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--cuda_devices", type=str, default="0", help="CUDA device IDs (e.g., '0,1').")
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    config = load_config(args.config)
    test_dataset = get_dataloader(config)

    sam_enhance = config['sam_enhance']
    save_mask = config['save_mask']
    visualize = config['visualize']

    if config['model'] == 'GradCAM' or config['model'] == 'GradCAMPlusPlus' or config['model'] == 'LayerCAM':
        model = Std_classifier_inference(config, test_dataset, sam_enhance=sam_enhance, method = config['model'])
    elif config['model'] == 'PuzzleCAM':
        model = Puzzle_CAM_inference(config, test_dataset, sam_enhance=sam_enhance)
    elif config['model'] == 'POF_CAM':
        model = POF_CAM_inference(config, test_dataset, sam_enhance=sam_enhance)
    else:
        raise ValueError("Invalid model type. Choose from 'GradCAM', 'PuzzleCAM', or 'POF_CAM'.")
    
    model.make_all_cams(save_mask=save_mask, visualize=visualize, norm=True, max_item=100000)

    print("\n\nEND\n\n")

if __name__ == "__main__":
    main()
