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

from POF_core.networks import *
from POFCAM_utils.optical_flow_utils import *
from POFCAM_utils.io_utils import *
from POFCAM_utils.augment_utils import *
from POFCAM_utils.frame_utils import *
from POFCAM_utils.log_utils import *
from POFCAM_utils.optim_utils import *
from POFCAM_utils.time_utils import *
from POFCAM_utils.torch_utils import *
from POFCAM_utils.puzzle_utils import *

seed = 42
num_workers = 4


dataset_dir_main = '../Datasets/SERUSO_DATASETS/main_dataset/Before_after_no_backgrounds/' # main_dataset/Before_after_no_backgrounds/' # new_5000/three_classes_5000/ #    Before_after_dataset_1240
flow_dir_main = '../Datasets/SERUSO_DATASETS/main_dataset/optical_flows/' # main_dataset/optical_flows/' # new_5000/optical_flows_5000/ #    Before_after_dataset_1240

dataset_dir_5000 = '../Datasets/SERUSO_DATASETS/new_5000/three_classes_5000/'
flow_dir_5000 = '../Datasets/SERUSO_DATASETS/new_5000/optical_flows_5000/'

architecture = 'resnet50'
mode ='normal' # fix
batch_size = 16
max_epoch = 40
lr = 0.1
wd = 1e-4
nesterov = True
image_size = 512 # 256
print_ratio = 0.1
augment = 'colorjitter' #'colorjitter'


re_loss_option = 'masking'   # 'none', 'masking', 'selection'
re_loss = 'L1_Loss'          # 'L1_Loss', 'L2_Loss'
alpha_schedule = 0.0 # 0.50 
glob_alpha = 2.00

beta_schedule = 0.0
glob_beta = 6.00
num_pieces = 4 # For Puzzle-CAM
loss_option = 'cl_re'

# 'cl_pcl'
# 'cl_re'
# 'cl_pcl_re'

tag = f"POFCAM_seruso_{image_size}_no_bg_three_classes_{max_epoch}_epochs_{architecture}_batch{batch_size}_{loss_option}_{re_loss_option}_only_lateral_{num_pieces}_pieces_new_dataset_alpha_{glob_alpha}_{alpha_schedule}__beta_{glob_beta}_{beta_schedule}"

level = 'cam' #  'cam'  'feature'

log_dir = create_directory(f'./experiments/logs/')
data_dir = create_directory(f'./experiments/data/')
model_dir = create_directory(f'./experiments/models/')
plot_dir = create_directory(f'./experiments/plots/')
tensorboard_dir = create_directory(f'./experiments/tensorboards/{tag}/')

log_path = log_dir + f'{tag}.txt'
data_path = data_dir + f'{tag}.json'
model_path = model_dir + f'{tag}.pth'

set_seed(seed)

log_func = lambda string='': log_print(string, log_path)

log_func('[i] {}'.format(tag))
log_func()

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

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

train_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, dstype = 'training', transform = train_transform, augment = False)
val_dataset = seruso_datasets.Seruso_three_classes_flow(img_root = dataset_dir_5000, flow_root = flow_dir_5000, dstype = 'validation', transform = train_transform, augment = False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
validation_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)

class_names = np.asarray(train_dataset.class_names)

log_func('[i] mean values is {}'.format(imagenet_mean))
log_func('[i] std values is {}'.format(imagenet_std))
log_func('[i] The number of class is {}'.format(len(class_names)))
log_func('[i] train_transform is {}'.format(train_transform))
log_func('[i] test_transform is {}'.format(test_transform))
log_func()

val_iteration = len(train_loader)
log_iteration = int(val_iteration * print_ratio)
max_iteration = max_epoch * val_iteration

 # val_iteration = log_iteration

log_func('[i] log_iteration : {:,}'.format(log_iteration))
log_func('[i] val_iteration : {:,}'.format(val_iteration))
log_func('[i] max_iteration : {:,}'.format(max_iteration))

model = Classifier(architecture, num_classes=len(train_dataset.class_dic), mode=mode)

param_groups = model.get_parameter_groups(print_fn=None)

gap_fn = model.global_average_pooling_2d

model = model.cuda()
model.train()

log_func('[i] Architecture is {}'.format(architecture))
log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
log_func()

try:
    use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
except KeyError:
    use_gpu = '0'

the_number_of_gpu = len(use_gpu.split(','))
if the_number_of_gpu > 1:
    log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
    model = nn.DataParallel(model)

    # for sync bn
    # patch_replication_callback(model)

load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)

class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

if re_loss == 'L1_Loss':
    re_loss_fn = L1_Loss
else:
    re_loss_fn = L2_Loss

log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

optimizer = PolyOptimizer([
    {'params': param_groups[0], 'lr': lr, 'weight_decay': wd},
    {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
    {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': wd},
    {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0},
], lr=lr, momentum=0.9, weight_decay=wd, max_step=max_iteration, nesterov=nesterov)

data_dic = {
    'train' : [],
    'validation' : []
}

train_timer = Timer()
eval_timer = Timer()

train_meter = Average_Meter(['loss', 'class_loss', 'p_class_loss', 're_loss', 're_loss_puzz', 'alpha', 'beta'])

best_train_loss = -1

writer = SummaryWriter(tensorboard_dir)
train_iterator = seruso_datasets.Iterator(train_loader)
valid_iterator = seruso_datasets.Iterator(validation_loader)

loss_option = loss_option.split('_')

def make_cam_non_norm(y, epsilon=1e-5):
    return F.relu(y)

def cam_norm(x, epsilon=1e-5):
    b, c, h, w = x.size()

    flat_x = x.view(b, c, (h * w))
    max_value = flat_x.max(axis=-1)[0].view((b, c, 1, 1))
    
    return F.relu(x - epsilon) / (max_value + epsilon)

def evaluate_for_validation(loader, alpha, beta):

    val_losses = []
    val_class_losses = []
    val_p_class_losses = []
    val_re_losses = []
    val_re_losses_puzz = []


    model.eval()
    eval_timer.tik()

    with torch.no_grad():
        length = len(loader)
        for step, (images_lists, flows_lists, labels) in enumerate(loader):

            labels =  labels.cuda()

            flows_left, flows_right = flows_lists

            middle = len(images_lists)//2

            central_images = images_lists [middle].cuda()
            left_images = images_lists [middle-1].cuda()
            right_images = images_lists [middle + 1].cuda()

            ###############################################################################
            # Normal
            ###############################################################################
            logits, features_puzz = model(central_images, with_cam=True)

            ###############################################################################
            # Puzzle Module
            ###############################################################################

            tiled_images = tile_features(central_images, num_pieces)

            tiled_logits, tiled_features = model(tiled_images, with_cam=True)

            re_features_puzz = merge_features(tiled_features, num_pieces, batch_size)

            ###############################################################################
            # FlowCAM Module
            ###############################################################################

            left_logits, left_features = model(left_images, with_cam=True)
            right_logits, right_features = model(right_images, with_cam=True)

            if level == 'cam':
                features = make_cam(features_puzz)
                left_features = make_cam_non_norm(left_features)
                right_features = make_cam_non_norm(right_features)

            flows_left = resize_flows_batch(flows_left, features.shape[-2:])
            flows_right = resize_flows_batch(flows_right, features.shape[-2:])

            warped_left, mask_left = warp(left_features.cuda(), flows_left.cuda())
            warped_right, mask_right = warp(right_features.cuda(), flows_right.cuda())

            re_features = cam_norm(torch.max(torch.stack([warped_left, features, warped_right], dim=1), dim=1)[0])

            ###############################################################################
            # Losses
            ###############################################################################
            if 'cl' in loss_option:
                class_loss = class_loss_fn(logits, labels).mean()
            else:
                class_loss = torch.zeros(1).cuda()

            if 'pcl' in loss_option:
                
                p_class_loss = class_loss_fn(gap_fn(re_features_puzz), labels).mean()
                t_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()
            else:
                p_class_loss = torch.zeros(1).cuda()
                t_class_loss = torch.zeros(1).cuda()

            if 're' in loss_option:
                if re_loss_option == 'masking':
                    class_mask = labels.unsqueeze(2).unsqueeze(3)
                    
                    re_loss = re_loss_fn(features, re_features) * class_mask
                    re_loss = re_loss.mean()

                    re_loss_puzz = re_loss_fn(features_puzz, re_features_puzz) * class_mask
                    re_loss_puzz = re_loss_puzz.mean()

                elif re_loss_option == 'selection':
                    re_loss = 0.
                    for b_index in range(labels.size()[0]):
                        class_indices = labels[b_index].nonzero(as_tuple=True)

                        selected_features = features[b_index][class_indices]
                        selected_re_features = re_features[b_index][class_indices]
                        re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                        re_loss += re_loss_per_feature
                
                        selected_features_puzz = features_puzz[b_index][class_indices]
                        selected_re_features_puzz = re_features_puzz[b_index][class_indices]
                        re_loss_per_feature_puzz = re_loss_fn(selected_features_puzz, selected_re_features_puzz).mean()
                        re_loss_puzz += re_loss_per_feature_puzz

                    re_loss /= labels.size()[0]
                    re_loss_puzz /= labels.size()[0]

                else:
                    re_loss = re_loss_fn(features, re_features).mean()
                    re_loss_puzz = re_loss_fn(features_puzz, re_features_puzz).mean()
            else:
                re_loss = torch.zeros(1).cuda()
                re_loss_puzz = torch.zeros(1).cuda()

            loss = class_loss + p_class_loss + beta*re_loss + alpha*re_loss_puzz
            #################################################################################################

            val_losses.append(loss.item())
            val_class_losses.append(class_loss.item())
            val_p_class_losses.append(p_class_loss.item())
            val_re_losses.append(beta*re_loss.item())
            val_re_losses_puzz.append(alpha*re_loss_puzz.item())

    print(' ')
    model.train()

    return val_losses, val_class_losses, val_p_class_losses, val_re_losses, val_re_losses_puzz

losses = []
class_losses = []
p_class_losses= []
re_losses = []
re_losses_puzz = []

losses_valid = []
class_losses_valid = []
p_class_losses_valid = []
re_losses_valid = []
re_losses_puzz_valid = []


temp_losses = []
temp_class_losses = []
temp_p_class_losses= []
temp_re_losses = []
temp_re_losses_puzz = []

best_val_loss = -1


for iteration in range(max_iteration):

    images_lists, flows_lists, labels = train_iterator.get() # images_lists, flows_lists, labels, params = train_iterator.get()

    labels =  labels.cuda()

    flows_left, flows_right = flows_lists

    middle = len(images_lists)//2

    central_images = images_lists [middle].cuda()
    left_images = images_lists [middle-1].cuda()
    right_images = images_lists [middle + 1].cuda()

    ###############################################################################
    # Normal
    ###############################################################################
    logits, features_puzz = model(central_images, with_cam=True)

    ###############################################################################
    # Puzzle Module
    ###############################################################################

    tiled_images = tile_features(central_images, num_pieces)

    tiled_logits, tiled_features = model(tiled_images, with_cam=True)

    re_features_puzz = merge_features(tiled_features, num_pieces, batch_size)

    ###############################################################################
    # FlowCAM Module
    ###############################################################################

    left_logits, left_features = model(left_images, with_cam=True)
    right_logits, right_features = model(right_images, with_cam=True)


    if level == 'cam':
        features = make_cam(features_puzz)
        left_features = make_cam_non_norm(left_features)
        right_features = make_cam_non_norm(right_features)

    # stored_transform = Three_images_batch_trasform(params)

    flows_left = resize_flows_batch(flows_left, features.shape[-2:])
    flows_right = resize_flows_batch(flows_right, features.shape[-2:])

    warped_left, mask_left = warp(left_features.cuda(), flows_left.cuda())
    warped_right, mask_right = warp(right_features.cuda(), flows_right.cuda())

    re_features = cam_norm(torch.max(torch.stack([warped_left, warped_right], dim=1), dim=1)[0])


    ###############################################################################
    # Losses
    ###############################################################################
    if 'cl' in loss_option:
        class_loss = class_loss_fn(logits, labels).mean()
    else:
        class_loss = torch.zeros(1).cuda()

    # class_loss = class_loss_fn(logits, labels).mean()

    if 'pcl' in loss_option:        
        p_class_loss = class_loss_fn(gap_fn(re_features_puzz), labels).mean()
        t_class_loss = class_loss_fn(gap_fn(re_features), labels).mean()

    else:
        p_class_loss = torch.zeros(1).cuda()
        t_class_loss = torch.zeros(1).cuda()

    if 're' in loss_option:
        if re_loss_option == 'masking':
            class_mask = labels.unsqueeze(2).unsqueeze(3)

            re_loss = re_loss_fn(features, re_features) * class_mask
            re_loss = re_loss.mean()
            
            re_loss_puzz = re_loss_fn(features_puzz, re_features_puzz) * class_mask
            re_loss_puzz = re_loss_puzz.mean()


        elif re_loss_option == 'selection':
            re_loss = 0.
            for b_index in range(labels.size()[0]):
                class_indices = labels[b_index].nonzero(as_tuple=True)

                selected_features = features[b_index][class_indices]
                selected_re_features = re_features[b_index][class_indices]
                re_loss_per_feature = re_loss_fn(selected_features, selected_re_features).mean()
                re_loss += re_loss_per_feature
                
                selected_features_puzz = features_puzz[b_index][class_indices]
                selected_re_features_puzz = re_features_puzz[b_index][class_indices]
                re_loss_per_feature_puzz = re_loss_fn(selected_features_puzz, selected_re_features_puzz).mean()
                re_loss_puzz += re_loss_per_feature_puzz

            re_loss /= labels.size()[0]
            re_loss_puzz /= labels.size()[0]

        else:
            re_loss = re_loss_fn(features, re_features).mean()            
            re_loss_puzz = re_loss_fn(features_puzz, re_features_puzz).mean()
    else:
        re_loss = torch.zeros(1).cuda()
        re_loss_puzz = torch.zeros(1).cuda()

    if alpha_schedule == 0.0:
        alpha = glob_alpha
    else:
        alpha = min(glob_alpha * (iteration) / (max_iteration * alpha_schedule), glob_alpha)

    
    if beta_schedule == 0.0:
        beta = glob_beta
    else:
        beta = min(glob_beta * (iteration) / (max_iteration * beta_schedule), glob_beta)


    loss = class_loss + p_class_loss + beta*re_loss + alpha*re_loss_puzz
    #################################################################################################

    temp_losses.append(loss.item())
    temp_class_losses.append(class_loss.item())
    temp_p_class_losses.append(p_class_loss.item())
    temp_re_losses.append(beta*re_loss.item())
    temp_re_losses_puzz.append(alpha*re_loss_puzz.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_meter.add({
        'loss' : loss.item(),
        'class_loss' : class_loss.item(),
        'p_class_loss' : p_class_loss.item(),
        're_loss' : re_loss.item(),
        're_loss_puzz' : re_loss_puzz.item(),
        'alpha' : alpha,
        'beta': beta,
    })

    #################################################################################################
    # For Log
    #################################################################################################
    if (iteration + 1) % log_iteration == 0:
        loss, class_loss, p_class_loss, re_loss, re_loss_puzz, alpha, beta = train_meter.get(clear=True)
        learning_rate = float(get_learning_rate_from_optimizer(optimizer))

        data = {
            'iteration' : iteration + 1,
            'learning_rate' : learning_rate,
            'alpha' : alpha,
            'beta' : beta,
            'loss' : loss,
            'class_loss' : class_loss,
            'p_class_loss' : p_class_loss,
            're_loss' : re_loss,
            're_loss_puzz' : re_loss_puzz,
            'time' : train_timer.tok(clear=True),
        }
        data_dic['train'].append(data)

        log_func('[i] \
            iteration={iteration:,}, \
            learning_rate={learning_rate:.4f}, \
            alpha={alpha:.2f}, \
            beta={beta:.2f}, \
            loss={loss:.4f}, \
            class_loss={class_loss:.4f}, \
            p_class_loss={p_class_loss:.4f}, \
            re_loss={re_loss:.4f}, \
            re_loss_puzz={re_loss_puzz:.4f}, \
            time={time:.0f}sec'.format(**data)
        )

        writer.add_scalar('Train/loss', loss, iteration)
        writer.add_scalar('Train/class_loss', class_loss, iteration)
        writer.add_scalar('Train/p_class_loss', p_class_loss, iteration)
        writer.add_scalar('Train/re_loss', re_loss, iteration)
        writer.add_scalar('Train/re_loss_puzz', re_loss_puzz, iteration)
        writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        writer.add_scalar('Train/alpha', alpha, iteration)
        writer.add_scalar('Train/beta', beta, iteration)

    #################################################################################################
    # Evaluation
    #################################################################################################
    if (iteration + 1) % val_iteration == 0:
        
        losses.append(np.mean(temp_losses))
        class_losses.append(np.mean(temp_class_losses))
        p_class_losses.append(np.mean(temp_p_class_losses))
        re_losses.append(np.mean(temp_re_losses))
        re_losses_puzz.append(np.mean(temp_re_losses_puzz))

        temp_losses = []
        temp_class_losses = []
        temp_p_class_losses= []
        temp_re_losses = []
        temp_re_losses_puzz = []

        val_losses, val_class_losses, val_p_class_losses, val_re_losses, val_re_losses_puzz = evaluate_for_validation(validation_loader, alpha, beta)

        val_loss = np.mean(val_losses)

        losses_valid.append(np.mean(val_losses))
        class_losses_valid.append(np.mean(val_class_losses))
        p_class_losses_valid.append(np.mean(val_p_class_losses))
        re_losses_valid.append(np.mean(val_re_losses))
        re_losses_puzz_valid.append(np.mean(val_re_losses_puzz))

        if best_val_loss == -1 or best_val_loss > val_loss:
            best_val_loss = val_loss

            save_model_fn()
            log_func('[i] save model')


        data = {
            'iteration' : iteration + 1,
            'val_loss' : val_loss,
            'best_val_loss' : best_val_loss,
            'time' : eval_timer.tok(clear=True),
        }
        data_dic['validation'].append(data)

        log_func('[i] \
            iteration={iteration:,}, \
            val_loss={val_loss:.4f}, \
            best_val_loss={best_val_loss:.4f}, \
            time={time:.0f}sec'.format(**data)
        )


        writer.add_scalar('Evaluation/val_loss', val_loss, iteration)
        writer.add_scalar('Evaluation/best_val_loss', best_val_loss, iteration)

writer.close()

# %%
# Create a figure and subplots
fig, axs = plt.subplots(1, 3, figsize=(20, 5))

# Plot the cumulative mean losses for training
axs[0].plot(range(len(losses)), losses, label='Total Loss')
axs[0].plot(range(len(class_losses)), class_losses, label='Class Loss')
axs[0].plot(range(len(p_class_losses)), p_class_losses, label='P Class Loss')
axs[0].plot(range(len(re_losses)), re_losses, label='Reconstruction Loss')
axs[0].plot(range(len(re_losses_puzz)), re_losses_puzz, label='Reconstruction Loss _puzz')
axs[0].set_title('Train Losses')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot the cumulative mean losses for validation
axs[1].plot(range(len(losses_valid)), losses_valid, label='Total Loss')
axs[1].plot(range(len(class_losses_valid)), class_losses_valid, label='Class Loss')
axs[1].plot(range(len(p_class_losses_valid)), p_class_losses_valid, label='P Class Loss')
axs[1].plot(range(len(re_losses_valid)), re_losses_valid, label='Reconstruction Loss')
axs[1].plot(range(len(re_losses_puzz_valid)), re_losses_puzz_valid, label='Reconstruction Loss Puzzle')
axs[1].set_title('Val Losses')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

# Plot the comparison of training and validation losses
axs[2].plot(range(len(losses)), losses, label='Train Loss')
axs[2].plot(range(len(losses_valid)), losses_valid, label='Val Loss')
axs[2].set_title('Train vs Val Losses')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Loss')
axs[2].legend()

# Add a general title to the entire figure
fig.suptitle('Best val loss = ' + str(best_val_loss), fontsize=12, fontweight='bold', ha='center')

# Adjust layout
plt.tight_layout()  # Adjust the position of subplots to make room for the title

# Save plot
plt.savefig(os.path.join(plot_dir, tag + '.png'))

# Show plot
# plt.show()
