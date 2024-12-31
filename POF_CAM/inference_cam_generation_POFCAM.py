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


from inference_cam_generation import Cam_generator_inference

from general_utils.augment_utils import *
from general_utils.frame_utils import *
from general_utils.torch_utils import *
from general_utils.cam_utils import *
from general_utils.io_utils import *
from general_utils.log_utils import *

from POF_core.networks import *
from POFCAM_utils.optical_flow_utils import *
from POFCAM_utils.optim_utils import *
from POFCAM_utils.time_utils import *
from POFCAM_utils.puzzle_utils import *


class POF_CAM_inference(Cam_generator_inference):

    def __init__(self, param1, param2):
        super().__init__(param1, param2)

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


