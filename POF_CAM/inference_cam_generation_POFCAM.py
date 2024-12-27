import sys
sys.path.append('POF_CAM')

import os
import numpy as np
import matplotlib.pyplot as plt
import copy
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms

import seruso_datasets

from general_utils.augment_utils import *
from general_utils.frame_utils import *
from general_utils.torch_utils import *

from POF_core.networks import *
from POFCAM_utils.optical_flow_utils import *
from POFCAM_utils.io_utils import *
from POFCAM_utils.log_utils import *
from POFCAM_utils.optim_utils import *
from POFCAM_utils.time_utils import *
from POFCAM_utils.puzzle_utils import *
from POFCAM_utils.cam_utils import *


class POF_CAM_inference:

    def __init__(self, config, test_dataset):
        # Set all attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)

        self.test_dataset = test_dataset
        self.scales = [float(scale) for scale in self.scales.split(',')]

        set_seed(self.seed)
        self.set_log()
        self.set_model()

    def set_log(self):
        self.log_dir = create_directory(f'./experiments/POF-CAM/log/inference/')
        self.cam_dir = create_directory(f'./experiments/POF-CAM/cam/')
        self.log_func = lambda string='': print(string)
    
    def set_model(self):

        self.eval_timer = Timer()
        self.eval_timer.tik()

        model_path = './experiments/POF-CAM/models/' + f'{self.tag}.pth'

        # Using regular expression to find the substring
        match = re.search(r'epochs_(.*?)_batch', self.tag)
        architecture = match.group(1)

        self.cam_model = Classifier(architecture, 3, mode=self.mode)

        self.cam_model = self.cam_model.cuda()
        self.cam_model.eval()

        self.log_func('[i] Architecture is {}'.format(architecture))
        self.log_func('[i] Total Params: %.2fM'%(calculate_parameters(self.cam_model)))
        self.log_func()

        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'

        the_number_of_gpu = len(use_gpu.split(','))
        if the_number_of_gpu > 1:
            self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
            self.cam_model = nn.DataParallel(self.cam_model)

        self.log_func(f'model_path: {model_path}')

        load_model(self.cam_model, model_path, parallel=the_number_of_gpu > 1)


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

    def generate_cams_lateral(self, left_s, right_s, flows, scales, cam_model):
        
        hr_left = generate_cams(left_s, cam_model, scales, normalize = False)
        hr_right = generate_cams(right_s, cam_model, scales, normalize = False)
        
        hr_left = torch.stack(hr_left).unsqueeze(0)
        hr_right = torch.stack(hr_right).unsqueeze(0)
        
        flows_left, flows_right = flows

        flows_left = flows_left.unsqueeze(0)
        flows_right = flows_right.unsqueeze(0)
        
        flows_left = resize_flows_batch(-flows_left, hr_left.shape[-2:])
        flows_right = resize_flows_batch(flows_right, hr_right.shape[-2:])

        warped_left, mask_left = warp(hr_left.cuda(), flows_left.cuda())
        warped_right, mask_right = warp(hr_right.cuda(), flows_right.cuda())
        
        re_hr = torch.max(torch.stack([warped_left, warped_right], dim=1), dim=1)[0]

        re_hr /= F.adaptive_max_pool2d(re_hr, (1, 1)) + 1e-5
        
        re_hr = list(torch.unbind(re_hr[0]))

        return re_hr

    def generate_masks(self, hr, img, gt = None, visualize = False):
        stacked_maps = torch.stack(hr)
        # Find the index of the attribution map with the maximum value for each pixel
        max_map_index = stacked_maps.argmax(dim=0)

        if visualize:
            num_rows= 1
            num_cols = 5

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
            fig.suptitle(f"Model: {self.tag}", fontsize=16)

            for i in range(len(hr)):
                mask_visualized = (max_map_index == i).int()
                
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
            masks = []
            for i in range(len(hr)):
                mask_visualized = (max_map_index == i).int()
                masks.append(mask_visualized.cpu())

            return masks
        

    def make_all_cams(self):

        denormalizer = Denormalize()

        with torch.no_grad():
            samples, flows, masks = self.test_dataset[200] #2368 for can  35 fpr anomaly  1789 for empty
            left_s, sample, right_s = samples
            mask_l, mask, masks_r = masks

            ori_image = sample #.image
            ori_w, ori_h = ori_image.shape[0], ori_image.shape[1]

            num_rows = 1
            num_cols = 5

            den_image = denormalizer(ori_image)

            hi_res_cams  = generate_cams(ori_image, self.cam_model, self.scales, True)
            hi_res_cams_lateral = self.generate_cams_lateral(left_s, right_s, flows, self.scales, self.cam_model)
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 5))
            fig.suptitle(f"Model: {self.tag}", fontsize=16)
            
            print("single")

            for i in range(len(hi_res_cams)):

                heatmap_image = show_cam_on_image(den_image, hi_res_cams[i], use_rgb=True)
                axes[i].imshow(heatmap_image)
                axes[i].axis('off')

            axes[-2].imshow(mask)
            axes[-2].axis('off')

            axes[-1].imshow(den_image)
            axes[-1].axis('off')
            
            plt.tight_layout()
            plt.show()

            self.generate_masks(hi_res_cams, den_image, mask, visualize = True)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
            fig.suptitle(f"Model: {self.tag}", fontsize=16)

            for i in range(len(hi_res_cams_lateral)):

                heatmap_image = show_cam_on_image(den_image, hi_res_cams_lateral[i], use_rgb=True)
                axes[i].imshow(heatmap_image)
                axes[i].axis('off')

            axes[-2].imshow(mask)
            axes[-2].axis('off')

            axes[-1].imshow(den_image)
            axes[-1].axis('off')
            
            plt.tight_layout()
            plt.show()

            self.generate_masks(hi_res_cams_lateral, den_image, mask, visualize = True)


