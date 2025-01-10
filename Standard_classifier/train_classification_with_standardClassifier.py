import sys
sys.path.append('Standard_classifier/')

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import copy
import time
import torch
import torch.cuda
import torch.backends.cudnn
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torchvision.models as models
import torch.utils.data

import torch.utils.model_zoo as model_zoo

from tqdm import tqdm
from matplotlib import pyplot as plt
from tqdm import tqdm

from standard_cl_utils.optim_utils import *
from standard_cl_utils.io_utils import *
from standard_cl_utils.txt_utils import *

from general_utils.augment_utils import *
from general_utils.torch_utils import *
from general_utils.augment_utils import *
from general_utils.frame_utils import *


class standardClassifier:

    def __init__(self, config, train_loader, validation_loader):
        # Set all attributes from the dictionary
        for key, value in config.items():
            setattr(self, key, value)

        # Save the passed data loaders
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        # Class names (assuming the data loader datasets provide access to class names)
        self.class_names = np.asarray(train_loader.dataset.class_names)

        self.tag = f"GradCAM_seruso_{self.image_size}_no_bg_three_classes_{self.max_epoch}_epochs_{self.architecture}_batch{self.batch_size}_{self.optimizer}"

        set_seed(self.seed)
        
        self.set_log()
        self.set_model()

    def set_log(self):
        self.val_iteration = len(self.train_loader)
        self.log_iteration = int(self.val_iteration * self.print_ratio)
        self.max_iteration = self.max_epoch * self.val_iteration

        self.log_dir = create_directory(f'./experiments/GradCAM/logs/')
        self.data_dir = create_directory(f'./experiments/GradCAM/data/')
        self.model_dir = create_directory(f'./experiments/GradCAM/models/')
        self.plot_dir = create_directory(f'./experiments/GradCAM/plots/')
        self.tensorboard_dir = create_directory(f'./experiments/GradCAM/tensorboards/{self.tag}/')
        
        self.log_path = self.log_dir + f'{self.tag}.txt'
        self.data_path = self.data_dir + f'{self.tag}.json'
        self.model_path = self.model_dir + f'{self.tag}.pth'

        self.log_func = lambda string='': log_print(string, self.log_path)

        self.log_func('[i] {}'.format(self.tag))
        self.log_func()

        self.log_func('[i] mean values is {}'.format(self.imagenet_mean))
        self.log_func('[i] std values is {}'.format(self.imagenet_std))
        self.log_func('[i] The number of class is {}'.format(len(self.class_names)))
        self.log_func()

        self.val_iteration = len(self.train_loader)
        self.log_iteration = int(self.val_iteration * self.print_ratio)
        self.max_iteration = self.max_epoch * self.val_iteration

        self.log_func('[i] log_iteration : {:,}'.format(self.log_iteration))
        self.log_func('[i] val_iteration : {:,}'.format(self.val_iteration))
        self.log_func('[i] max_iteration : {:,}'.format(self.max_iteration))

    def set_model(self):

        # loads a pre-trained ResNet-50 model from the torchvision library. The True argument means that the model will be loaded with pre-trained weights.
        model = getattr(models, self.architecture)(pretrained=True)

        # extracts the number of input features (or neurons) in the last fully connected layer (fc) of the ResNet-50 model.
        num_ftrs = model.fc.in_features

        # replaces the fully connected layer (fc) of the ResNet-50 model with a new linear layer (nn.Linear) that has num_ftrs input features and 2 output features.
        model.fc = nn.Linear(num_ftrs, len(self.class_names))

        #  creates a new sequential model that consists of the modified ResNet-50 and a softmax layer (nn.Softmax). The softmax layer is used to convert the model's output into probabilities for each class.
        model = nn.Sequential(model, nn.Softmax())

        model.cuda()
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

        param_groups = get_parameter_groups(model, print_fn=None)

        self.log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
        self.log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
        self.log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
        self.log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

        self.load_model_fn = lambda: load_model(self.model, self.model_path, parallel=the_number_of_gpu > 1)
        self.save_model_fn = lambda: save_model(self.model, self.model_path, parallel=the_number_of_gpu > 1)

        if self.optimizer == 'SGD':
            self.optimizer_ft = PolyOptimizer([
                    {'params': param_groups[0], 'lr': self.lr, 'weight_decay': self.wd},
                    {'params': param_groups[1], 'lr': 2*self.lr, 'weight_decay': 0},
                    {'params': param_groups[2], 'lr': 10*self.lr, 'weight_decay': self.wd},
                    {'params': param_groups[3], 'lr': 20*self.lr, 'weight_decay': 0},
                ], lr = self.lr, momentum=0.9, weight_decay=self.wd, max_step = self.max_iteration, nesterov=self.nesterov)
            
        elif self.optimizer == 'adam':
            feature_extract = True

            params_to_update = self.model.parameters()
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name,param in self.model.named_parameters():
                    if param.requires_grad == True: # This condition checks if the parameter requires gradient computation
                        params_to_update.append(param)
                        print("\t",name)
            else:
                for name,param in self.model.named_parameters():
                    if param.requires_grad == True:
                        print("\t",name)

            # An Adam optimizer is created, and it is passed the parameters that need to be updated (based on the params_to_update list) and a learning rate of 0.0005.
            self.optimizer_ft = torch.optim.Adam(params_to_update, lr=0.0005)

        # An exponential learning rate scheduler is created, adjusting the learning rate during training with a decay factor of 0.9.
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_ft, gamma=0.9)


        self.criterion = torch.nn.BCELoss()
        # self.criterion = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()
        self.dataloaders_dict = {"train": self.train_loader, "val": self.validation_loader}


    def train(self):

        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())

        best_acc = 1.

        
        for epoch in range(self.max_epoch):

            print('Epoch {}/{}'.format(epoch, self.max_epoch - 1))
            print('-' * 10)

            self.log_func('Epoch {}/{}\n'.format(epoch, self.max_epoch - 1))
            self.log_func('-' * 10 + '\n')

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                num_samples = 0

                # Iterate over data
                for images, labels in tqdm(self.dataloaders_dict[phase]):
                    images, labels = images.cuda(), labels.cuda()

                    # Zero the parameter gradients
                    self.optimizer_ft.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(images)
                        # labels.data = labels.data
                        labels = labels.float()

                        loss = self.criterion(outputs, labels).mean()  #(logits, labels).mean()
                        preds = (outputs > 0.5).type(torch.cuda.FloatTensor)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer_ft.step()

                    # Statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.mean(torch.abs(preds - labels.data))
                    num_samples += 1

                epoch_loss = running_loss / self.batch_size * num_samples
                epoch_acc = running_corrects.double() / num_samples

                self.log_func('{} Loss: {:.4f} MSE: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

                # Deep copy the model
                if phase == 'val':

                    valid_losses.append(epoch_loss)
                    valid_accuracies.append(epoch_acc)

                    if epoch_acc < best_acc:
                        best_acc = epoch_acc

                        best_model_wts = copy.deepcopy(self.model.state_dict())

                else:
                    train_losses.append(epoch_loss)
                    train_accuracies.append(epoch_acc)

        time_elapsed = time.time() - since

        self.log_func('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
        self.log_func('Best val MSE: {:4f}\n'.format(best_acc))

        # Load best model weights
        self.model.load_state_dict(best_model_wts)

        self.save_model_fn()
        
        print(train_accuracies, valid_accuracies)
    
        self.plot_loss(train_losses, valid_losses)



    def plot_loss(self, tr_losses, val_losses):
        plt.figure()
        plt.plot(range(len(tr_losses)), tr_losses, label='Training Loss')
        plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, self.tag + '.png'))






