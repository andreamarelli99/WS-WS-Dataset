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

        self.class_dict = {v: k for k, v in test_dataset.class_dic.items()}
        self.denormalizer = Denormalize()
        set_seed(self.seed)
        self.set_log()
        self.set_model()

    def set_log(self):
        self.log_dir = create_directory(f'./experiments/POF-CAM/log/inference/')
        self.cam_dir = create_directory(f'./experiments/POF-CAM/cams/')
        self.log_func = lambda string='': print(string)
    
    def set_model(self):

        self.eval_timer = Timer()
        self.eval_timer.tik()

        model_path = './experiments/POF-CAM/models/' + f'{self.tag}.pth'

        # Using regular expression to find the substring
        match = re.search(r'epochs_(.*?)_batch', self.tag)
        architecture = match.group(1)

        loaded_dict = torch.load(model_path)

        self.num_of_classes = loaded_dict['classifier.weight'].shape[0]

        self.cam_model = Classifier(architecture, self.num_of_classes, mode=self.mode)

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

        # load_model(self.cam_model, model_path, parallel=the_number_of_gpu > 1)

        if the_number_of_gpu > 1:
            self.cam_model.module.load_state_dict(loaded_dict)
        else:
            self.cam_model.load_state_dict(loaded_dict)

    
    def generate_cams_lateral(self, left_s, sample, right_s, flows, scales, cam_model):
        
        hr = generate_cams(sample, cam_model, scales, normalize = False)
        hr_left = generate_cams(left_s, cam_model, scales, normalize = False)
        hr_right = generate_cams(right_s, cam_model, scales, normalize = False)
        
        hr = torch.stack(hr).unsqueeze(0)
        hr_left = torch.stack(hr_left).unsqueeze(0)
        hr_right = torch.stack(hr_right).unsqueeze(0)
        
        flows_left, flows_right = flows

        flows_left = flows_left.unsqueeze(0)
        flows_right = flows_right.unsqueeze(0)
        
        flows_left = resize_flows_batch(-flows_left, hr_left.shape[-2:])
        flows_right = resize_flows_batch(flows_right, hr_right.shape[-2:])

        warped_left, mask_left = warp(hr_left.cuda(), flows_left.cuda())
        warped_right, mask_right = warp(hr_right.cuda(), flows_right.cuda())
        hr = hr.cuda()
        
        re_hr = torch.max(torch.stack([warped_left, hr, warped_right], dim=1), dim=1)[0]

        re_hr /= F.adaptive_max_pool2d(re_hr, (1, 1)) + 1e-5
        
        re_hr = list(torch.unbind(re_hr[0]))

        return re_hr


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



    def make_all_cams(self, visualize = False):

        with torch.no_grad():

            if self.test_dataset.get_whith_mask_bool():

                if self.test_dataset.get_whith_flows_bool():

                    for index_for_dataset in range(len(self.test_dataset)):
                        samples, flows, masks, path = self.test_dataset[index_for_dataset]
                        left_s, sample, right_s = samples
                        _, mask, _ = masks
                        hi_res_cams = self.generate_cams_lateral(left_s, sample, right_s, flows, self.scales, self.cam_model)
                        masks = self.generate_masks(hi_res_cams, sample, mask, visualize = visualize)
                        self.save_masks(masks, path)

                else:
                    
                    for index_for_dataset in range(len(self.test_dataset)):
                        sample, mask, path = self.test_dataset[index_for_dataset]
                        hi_res_cams  = generate_cams(sample, self.cam_model, self.scales, normalize = True)
                        masks = self.generate_masks(hi_res_cams, sample, mask, visualize = visualize)
                        self.save_masks(masks, path)
                
            else:

                if self.test_dataset.get_whith_flows_bool():
                    for index_for_dataset in range(len(self.test_dataset)):
                        samples, flows, path = self.test_dataset[index_for_dataset]
                        left_s, sample, right_s = samples
                        hi_res_cams = self.generate_cams_lateral(left_s, sample, right_s, flows, self.scales, self.cam_model)
                        masks = self.generate_masks(hi_res_cams, sample, visualize = visualize)
                        self.save_masks(masks, path)

                else:
                    for index_for_dataset in range(len(self.test_dataset)):
                        sample, path  = self.test_dataset[index_for_dataset]
                        hi_res_cams  = generate_cams(sample, self.cam_model, self.scales, normalize = True)
                        masks = self.generate_masks(hi_res_cams, sample, visualize = visualize)
                        self.save_masks(masks, path)


