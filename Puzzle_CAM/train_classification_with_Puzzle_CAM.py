import sys
sys.path.append('Puzzle_CAM')

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import seruso_datasets

from general_utils.torch_utils import *
from general_utils.augment_utils import *
from general_utils.frame_utils import *
from general_utils.io_utils import *
from general_utils.log_utils import *

from Puzzle_CAM_core.networks import *
from Puzzle_CAM_utils.optim_utils import *
from Puzzle_CAM_utils.time_utils import *
from Puzzle_CAM_utils.puzzle_utils import *

class Puzzle_CAM:

    def __init__(self, config, train_loader, validation_loader):
        # Set all attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)

        # Save the passed data loaders
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # Class names (assuming the data loader datasets provide access to class names)
        self.class_names = np.asarray(train_loader.dataset.class_names)

        self.tag = f"Puzzle_CAM_seruso_{self.image_size}_no_bg_three_classes_{self.max_epoch}_epochs_{self.architecture}_batch{self.batch_size}_{self.loss_option}_{self.re_loss_option}_only_lateral_{self.num_pieces}_pieces_new_dataset_alpha_{self.glob_alpha}_{self.alpha_schedule}"

        set_seed(self.seed)
        self.set_log()
        self.set_model()
        

    def set_log (self):
        self.log_dir = create_directory(f'./experiments/Puzzle-CAM/logs/')
        self.data_dir = create_directory(f'./experiments/Puzzle-CAM/data/')
        self.model_dir = create_directory(f'./experiments/Puzzle-CAM/models/')
        self.plot_dir = create_directory(f'./experiments/Puzzle-CAM/plots/')
        self.tensorboard_dir = create_directory(f'./experiments/Puzzle-CAM/tensorboards/{ self.tag}/')

        self.log_path = self.log_dir + f'{self.tag}.txt'
        self.data_path = self.data_dir + f'{self.tag}.json'
        self.model_path = self.model_dir + f'{self.tag}.pth'


        self.log_func = lambda string='': log_print(string, self.log_path)

        self.log_func('[i] {}'.format(self.tag))
        self.log_func()

        self.log_func('[i] mean values is {}'.format(self.imagenet_mean))
        self.log_func('[i] std values is {}'.format(self.imagenet_std))
        self.log_func('[i] The number of class is {}'.format(len(self.class_names)))
        # self.log_func('[i] train_transform is {}'.format(self.train_transform))
        # self.log_func('[i] test_transform is {}'.format(self.test_transform))
        self.log_func()

        self.val_iteration = len(self.train_loader)
        self.log_iteration = int(self.val_iteration * self.print_ratio)
        self.max_iteration = self.max_epoch * self.val_iteration

        # val_iteration = log_iteration

        self.log_func('[i] log_iteration : {:,}'.format(self.log_iteration))
        self.log_func('[i] val_iteration : {:,}'.format(self.val_iteration))
        self.log_func('[i] max_iteration : {:,}'.format(self.max_iteration))

    def set_model(self):

        model = Classifier(self.architecture, num_classes=len(self.class_names), mode=self.mode)

        param_groups = model.get_parameter_groups(print_fn=None)

        self.gap_fn = model.global_average_pooling_2d

        model = model.cuda()
        model.train()

        self.log_func('[i] Architecture is {}'.format(self.architecture))
        self.log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
        self.log_func()

        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'

        the_number_of_gpu = len(use_gpu.split(','))
        if the_number_of_gpu > 1:
            self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
            model = nn.DataParallel(model)

            # for sync bn
            # patch_replication_callback(model)
        self.model = model

        self.load_model_fn = lambda: load_model(self.model, self.model_path, parallel=the_number_of_gpu > 1)
        self.save_model_fn = lambda: save_model(self.model, self.model_path, parallel=the_number_of_gpu > 1)

        self.class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

        if self.re_loss == 'L1_Loss':
            self.re_loss_fn = L1_Loss
        else:
            self.re_loss_fn = L2_Loss

        self.log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
        self.log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
        self.log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
        self.log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

        self.optimizer = PolyOptimizer([
            {'params': param_groups[0], 'lr': self.lr, 'weight_decay': self.wd},
            {'params': param_groups[1], 'lr': 2*self.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10*self.lr, 'weight_decay': self.wd},
            {'params': param_groups[3], 'lr': 20*self.lr, 'weight_decay': 0},
        ], lr = self.lr, momentum=0.9, weight_decay=self.wd, max_step=self.max_iteration, nesterov=self.nesterov)

        self.data_dic = {
            'train' : [],
            'validation' : []
        }

        self.train_timer = Timer()
        self.eval_timer = Timer()

        self.train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss_puzz', 'alpha'])

        self.writer = SummaryWriter(self.tensorboard_dir)
        self.train_iterator = seruso_datasets.Iterator(self.train_loader)
        self.valid_iterator = seruso_datasets.Iterator(self.validation_loader)

        self.loss_option = self.loss_option.split('_')
        

    def evaluate_for_validation(self, loader, alpha):

        val_losses = []
        val_class_losses = []
        val_p_class_losses = []
        val_re_losses_puzz = []

        self.model.eval()
        self.eval_timer.tik()

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):

                images = images.cuda()
                labels = labels.cuda()

                ###############################################################################
                # Normal
                ###############################################################################
                logits, features_puzz = self.model(images, with_cam=True)

                ###############################################################################
                # Puzzle Module
                ###############################################################################

                tiled_images = tile_features(images, self.num_pieces)

                tiled_logits, tiled_features = self.model(tiled_images, with_cam=True)

                re_features_puzz = merge_features(tiled_features, self.num_pieces, self.batch_size)

                ###############################################################################
                # Losses
                ###############################################################################
                
                if self.level == 'cam':
                    features = make_cam(features)
                    re_features = make_cam(re_features)

                if 'cl' in self.loss_option:
                    class_loss = self.class_loss_fn(logits, labels).mean()
                else:
                    class_loss = torch.zeros(1).cuda()

                if 'pcl' in self.loss_option:
                    
                    p_class_loss = self.class_loss_fn(self.gap_fn(re_features_puzz), labels).mean()
                else:
                    p_class_loss = torch.zeros(1).cuda()

                if 're' in self.loss_option:
                    if self.re_loss_option == 'masking':
                        class_mask = labels.unsqueeze(2).unsqueeze(3)
                
                        re_loss_puzz = self.re_loss_fn(features_puzz, re_features_puzz) * class_mask
                        re_loss_puzz = re_loss_puzz.mean()

                    elif self.re_loss_option == 'selection':
                        for b_index in range(labels.size()[0]):
                            class_indices = labels[b_index].nonzero(as_tuple=True)
                    
                            selected_features_puzz = features_puzz[b_index][class_indices]
                            selected_re_features_puzz = re_features_puzz[b_index][class_indices]
                            re_loss_per_feature_puzz = self.re_loss_fn(selected_features_puzz, selected_re_features_puzz).mean()
                            re_loss_puzz += re_loss_per_feature_puzz

                        re_loss_puzz /= labels.size()[0]

                    else:
                        re_loss_puzz = self.re_loss_fn(features_puzz, re_features_puzz).mean()
                else:
                    re_loss_puzz = torch.zeros(1).cuda()

                if 'conf' in self.loss_option:
                    conf_loss = shannon_entropy_loss(tiled_logits)
                else:
                    conf_loss = torch.zeros(1).cuda()

                loss = class_loss + p_class_loss + alpha*re_loss_puzz + conf_loss
                #################################################################################################

                val_losses.append(loss.item())
                val_class_losses.append(class_loss.item())
                val_p_class_losses.append(p_class_loss.item())
                val_re_losses_puzz.append(alpha*re_loss_puzz.item())

        print(' ')
        self.model.train()

        return val_losses, val_class_losses, val_p_class_losses, val_re_losses_puzz
    
    def train(self):

        losses = []
        class_losses = []
        p_class_losses= []
        re_losses_puzz = []

        losses_valid = []
        class_losses_valid = []
        p_class_losses_valid = []
        re_losses_puzz_valid = []


        temp_losses = []
        temp_class_losses = []
        temp_p_class_losses= []
        temp_re_losses_puzz = []

        best_val_loss = -1

        for iteration in range(self.max_iteration):

            images, labels = self.train_iterator.get()
            images, labels = images.cuda(), labels.cuda()

            ###############################################################################
            # Normal
            ###############################################################################
            logits, features_puzz = self.model(images, with_cam=True)

            ###############################################################################
            # Puzzle Module
            ###############################################################################

            tiled_images = tile_features(images, self.num_pieces)

            tiled_logits, tiled_features = self.model(tiled_images, with_cam=True)

            re_features_puzz = merge_features(tiled_features, self.num_pieces, self.batch_size)

            ###############################################################################
            # Losses
            ###############################################################################
            
            if self.level == 'cam':
                features = make_cam(features)
                re_features = make_cam(re_features)

            if 'cl' in self.loss_option:
                class_loss = self.class_loss_fn(logits, labels).mean()
            else:
                class_loss = torch.zeros(1).cuda()

            # class_loss = class_loss_fn(logits, labels).mean()

            if 'pcl' in self.loss_option:        
                p_class_loss = self.class_loss_fn(self.gap_fn(re_features_puzz), labels).mean()
            else:
                p_class_loss = torch.zeros(1).cuda()

            if 're' in self.loss_option:
                if self.re_loss_option == 'masking':
                    class_mask = labels.unsqueeze(2).unsqueeze(3)
                    
                    re_loss_puzz = self.re_loss_fn(features_puzz, re_features_puzz) * class_mask
                    re_loss_puzz = re_loss_puzz.mean()


                elif self.re_loss_option == 'selection':
                    for b_index in range(labels.size()[0]):
                        class_indices = labels[b_index].nonzero(as_tuple=True)
                        
                        selected_features_puzz = features_puzz[b_index][class_indices]
                        selected_re_features_puzz = re_features_puzz[b_index][class_indices]
                        re_loss_per_feature_puzz = self.re_loss_fn(selected_features_puzz, selected_re_features_puzz).mean()
                        re_loss_puzz += re_loss_per_feature_puzz

                    re_loss_puzz /= labels.size()[0]

                else:   
                    re_loss_puzz = self.re_loss_fn(features_puzz, re_features_puzz).mean()
            else:
                re_loss_puzz = torch.zeros(1).cuda()
                    
            if 'conf' in self.loss_option:
                conf_loss = shannon_entropy_loss(tiled_logits)
            else:
                conf_loss = torch.zeros(1).cuda()

            if self.alpha_schedule == 0.0:
                alpha = self.glob_alpha
            else:
                alpha = min(self.glob_alpha * (iteration) / (self.max_iteration * self.alpha_schedule), self.glob_alpha)


            loss = class_loss + p_class_loss + alpha*re_loss_puzz + conf_loss
            #################################################################################################

            temp_losses.append(loss.item())
            temp_class_losses.append(class_loss.item())
            temp_p_class_losses.append(p_class_loss.item())
            temp_re_losses_puzz.append(alpha*re_loss_puzz.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_meter.add({
                'loss' : loss.item(),
                'class_loss' : class_loss.item(),
                'p_class_loss' : p_class_loss.item(),
                're_loss_puzz' : re_loss_puzz.item(),
                'alpha' : alpha,
            })

            #################################################################################################
            # For Log
            #################################################################################################
            if (iteration + 1) % self.log_iteration == 0:
                loss, class_loss, p_class_loss, re_loss_puzz, alpha = self.train_meter.get(clear=True)
                learning_rate = float(get_learning_rate_from_optimizer(self.optimizer))

                data = {
                    'iteration' : iteration + 1,
                    'learning_rate' : learning_rate,
                    'alpha' : alpha,
                    'loss' : loss,
                    'class_loss' : class_loss,
                    'p_class_loss' : p_class_loss,
                    're_loss_puzz' : re_loss_puzz,
                    'time' : self.train_timer.tok(clear=True),
                }
                self.data_dic['train'].append(data)

                self.log_func('[i] \
                    iteration={iteration:,}, \
                    learning_rate={learning_rate:.4f}, \
                    alpha={alpha:.2f}, \
                    loss={loss:.4f}, \
                    class_loss={class_loss:.4f}, \
                    p_class_loss={p_class_loss:.4f}, \
                    re_loss_puzz={re_loss_puzz:.4f}, \
                    time={time:.0f}sec'.format(**data)
                )

                self.writer.add_scalar('Train/loss', loss, iteration)
                self.writer.add_scalar('Train/class_loss', class_loss, iteration)
                self.writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
                self.writer.add_scalar('Train/re_loss_puzz', re_loss_puzz, iteration)
                self.writer.add_scalar('Train/learning_rate', learning_rate, iteration)
                self.writer.add_scalar('Train/alpha', alpha, iteration)

            #################################################################################################
            # Evaluation
            #################################################################################################
            if (iteration + 1) % self.val_iteration == 0:
                
                losses.append(np.mean(temp_losses))
                class_losses.append(np.mean(temp_class_losses))
                p_class_losses.append(np.mean(temp_p_class_losses))
                re_losses_puzz.append(np.mean(temp_re_losses_puzz))

                temp_losses = []
                temp_class_losses = []
                temp_p_class_losses= []
                temp_re_losses_puzz = []

                val_losses, val_class_losses, val_p_class_losses, val_re_losses_puzz = self.evaluate_for_validation(self.validation_loader, alpha)

                val_loss = np.mean(val_losses)

                losses_valid.append(np.mean(val_losses))
                class_losses_valid.append(np.mean(val_class_losses))
                p_class_losses_valid.append(np.mean(val_p_class_losses))
                re_losses_puzz_valid.append(np.mean(val_re_losses_puzz))

                if best_val_loss == -1 or best_val_loss > val_loss:
                    best_val_loss = val_loss

                    self.save_model_fn()
                    self.log_func('[i] save model')


                data = {
                    'iteration' : iteration + 1,
                    'val_loss' : val_loss,
                    'best_val_loss' : best_val_loss,
                    'time' : self.eval_timer.tok(clear=True),
                }
                self.data_dic['validation'].append(data)

                self.log_func('[i] \
                    iteration={iteration:,}, \
                    val_loss={val_loss:.4f}, \
                    best_val_loss={best_val_loss:.4f}, \
                    time={time:.0f}sec'.format(**data)
                )


                self.writer.add_scalar('Evaluation/val_loss', val_loss, iteration)
                self.writer.add_scalar('Evaluation/best_val_loss', best_val_loss, iteration)

        self.writer.close()

        train_losses = {
            'Total Loss': losses,
            'Class Loss': class_losses,
            'P Class Loss': p_class_losses,
            'Reconstruction Loss puzz': re_losses_puzz
        }

        valid_losses = {
            'Total Loss': losses_valid,
            'Class Loss': class_losses_valid,
            'P Class Loss': p_class_losses_valid,
            'Reconstruction Loss puzz': re_losses_puzz_valid
        }

        self.plot(train_losses, valid_losses, best_val_loss)


    def plot (self, tr_losses, val_losses, best_val):

        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        for name, values in tr_losses.items():
            # Plot the cumulative mean losses for training
            axs[0].plot(range(len(values)), values, label=name)

        axs[0].set_title('Train Losses')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        # Plot the cumulative mean losses for validation
        for name, values in val_losses.items():
            # Plot the cumulative mean losses for training
            axs[0].plot(range(len(values)), values, label=name)

        axs[1].set_title('Val Losses')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()

        # Plot the comparison of training and validation losses
        axs[2].plot(range(len(tr_losses['Total Loss'])), tr_losses['Total Loss'], label='Train Loss')
        axs[2].plot(range(len(val_losses['Total Loss'])), val_losses['Total Loss'], label='Val Loss')
        axs[2].set_title('Train vs Val Losses')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Loss')
        axs[2].legend()

        # Add a general title to the entire figure
        fig.suptitle('Best val loss = ' + str(best_val), fontsize=12, fontweight='bold', ha='center')

        # Adjust layout
        plt.tight_layout()  # Adjust the position of subplots to make room for the title

        # Save plot
        plt.savefig(os.path.join(self.plot_dir, self.tag + '.png'))

        # Show plot
        # plt.show()

