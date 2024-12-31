import os
import numpy as np
import matplotlib.pyplot as plt
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

from general_utils.augment_utils import *
from general_utils.frame_utils import *
from general_utils.torch_utils import *
from general_utils.cam_utils import *
from general_utils.log_utils import *
from general_utils.io_utils import *


class Cam_generator_inference:

    def __init__(self, config, test_dataset):
        # Set all attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)

        self.test_dataset = test_dataset
        self.scales = [float(scale) for scale in self.scales.split(',')]

        self.class_dict = {v: k for k, v in test_dataset.class_dic.items()}
        self.denormalizer = Denormalize()
        set_seed(self.seed)
        self.set_log()
        self.set_model()


    def generate_masks_no_gt(self, hr, img):
        stacked_maps = torch.stack(hr)
        # Find the index of the attribution map with the maximum value for each pixel
        max_map_index = stacked_maps.argmax(dim=0)

        num_rows= 1
        num_cols = 4

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
        fig.suptitle(f"Model: {self.tag}", fontsize=16)

        for i in range(num_cols-1):

            mask_visualized = (max_map_index == i).int()
            
            axes[i].imshow(mask_visualized.cpu(), cmap='binary')
            axes[i].axis('off')    

        axes[-1].imshow(img)
        axes[-1].axis('off')    

        # Plot the mask
        plt.tight_layout()
        plt.show()

    def generate_masks(self, hr, img = None, gt = None, visualize = False):
        stacked_maps = torch.stack(hr)
        # Find the index of the attribution map with the maximum value for each pixel
        max_map_index = stacked_maps.argmax(dim=0)

        masks = []

        if visualize:

            self.visualize_cams(img, hr, mask = gt)
            num_rows= 1

            if gt == None:

                num_cols = self.num_of_classes +1

                fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
                fig.suptitle(f"Model: {self.tag}", fontsize=16)

                for i in range(self.num_of_classes):

                    mask_visualized = (max_map_index == i).int()

                    masks.append(mask_visualized.cpu())
                    
                    axes[i].imshow(mask_visualized.cpu(), cmap='binary')
                    axes[i].axis('off')    

                axes[-1].imshow(img)
                axes[-1].axis('off')  

            else:
                num_cols = self.num_of_classes + 2

                fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
                fig.suptitle(f"Model: {self.tag}", fontsize=16)

                for i in range(self.num_of_classes):
                    mask_visualized = (max_map_index == i).int()

                    masks.append(mask_visualized.cpu())
                    
                    axes[i].imshow(mask_visualized.cpu(), cmap='binary')
                    axes[i].axis('off')    

                axes[-2].imshow(gt, cmap='binary')# Set subplot title
                axes[-2].axis('off')

            axes[-1].imshow(img)
            axes[-1].axis('off')    

            # Plot the mask
            plt.tight_layout()
            plt.show()

        else:
            for i in range(self.num_of_classes):
                mask_visualized = (max_map_index == i).int()
                masks.append(mask_visualized.cpu())

        
        return masks

        
    def visualize_cams(self, sample, hi_res_cams, mask = None):

        den_image = self.denormalizer(sample)

        if mask == None:
            num_rows = 1
            num_cols = 4

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
            fig.suptitle(f"Model: {self.tag}", fontsize=16)
            
            for i in range(self.num_of_classes):

                heatmap_image = show_cam_on_image(den_image, hi_res_cams[i], use_rgb=True)
                axes[i].imshow(heatmap_image)
                axes[i].axis('off')

            axes[-1].imshow(den_image)
            axes[-1].axis('off')
            
            plt.tight_layout()
            plt.show()

        else:   
            num_rows = 1
            num_cols = 5

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
            fig.suptitle(f"Model: {self.tag}", fontsize=16)
            
            for i in range(self.num_of_classes):

                heatmap_image = show_cam_on_image(den_image, hi_res_cams[i], use_rgb=True)
                axes[i].imshow(heatmap_image)
                axes[i].axis('off')

            axes[-2].imshow(mask)
            axes[-2].axis('off')

            axes[-1].imshow(den_image)
            axes[-1].axis('off')
            
            plt.tight_layout()
            plt.show()

    def save_masks(self, msks, ori_path):

        _, rel_path = ori_path.split("images/", 1)
        rel_path = rel_path.replace(".jpg", ".npz")

        for i in range(len(self.class_dict)):

            full_path = os.path.join(self.cam_dir, self.class_dict[i] + '_maps', rel_path)
            
            directory = create_directory(f'{os.path.dirname(full_path)}/')

            np.savez_compressed(full_path, array=msks[i])

