import sys

sys.path.append('Puzzle_CAM')

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

from Puzzle_CAM_core.networks import *
from Puzzle_CAM_utils.optim_utils import *
from Puzzle_CAM_utils.time_utils import *
from Puzzle_CAM_utils.puzzle_utils import *


class Puzzle_CAM_inference(Cam_generator_inference):

    def __init__(self, param1, param2):
        super().__init__(param1, param2)
        self.test_dataset.do_it_without_flows()

    def set_log(self):
        self.log_dir = create_directory(f'./experiments/Puzzle-CAM/log/inference/')
        self.cam_dir = create_directory(f'./experiments/Puzzle-CAM/cams/')
        self.log_func = lambda string='': print(string)
    
    def set_model(self):

        self.eval_timer = Timer()
        self.eval_timer.tik()

        model_path = './experiments/Puzzle-CAM/models/' + f'{self.tag}.pth'

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


    def make_all_cams(self, save_mask = True, visualize = False, norm = True, max_item =10):

        with torch.no_grad():

            if self.test_dataset.get_whith_mask_bool():
                    
                ious = []

                for index_for_dataset in range(len(self.test_dataset)):
                    sample, gt, path = self.test_dataset[index_for_dataset]
                    hi_res_cams  = generate_cams(sample, self.cam_model, self.scales, normalize = norm)
                    mask = self.generate_masks(hi_res_cams, sample, gt, visualize = visualize)
                    ious.append(self.compute_iou(mask, gt))
                    if save_mask:
                        self.save_masks(mask, path)
                    else:
                        if index_for_dataset > max_item:
                            break
                print(f'Mean IoU: {np.mean(ious)}')
                
            else:

                for index_for_dataset in range(len(self.test_dataset)):
                    sample, path  = self.test_dataset[index_for_dataset]
                    hi_res_cams  = generate_cams(sample, self.cam_model, self.scales, normalize = norm)
                    mask = self.generate_masks(hi_res_cams, sample, visualize = visualize)
                    if save_mask:
                        self.save_masks(mask, path)
                    else:
                        if index_for_dataset > max_item:
                            break


