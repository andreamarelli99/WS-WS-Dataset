# %% [markdown]
# ## Fine-tune SegFormer on a custom dataset
# 
# In this notebook, we are going to fine-tune [SegFormerForSemanticSegmentation](https://huggingface.co/docs/transformers/main/model_doc/segformer#transformers.SegformerForSemanticSegmentation) on a custom **semantic segmentation** dataset. In semantic segmentation, the goal for the model is to label each pixel of an image with one of a list of predefined classes.
# 
# We load the encoder of the model with weights pre-trained on ImageNet-1k, and fine-tune it together with the decoder head, which starts with randomly initialized weights.

# %% [markdown]
# ## Download toy dataset
# 
# Here we download a small subset of the ADE20k dataset, which is an important benchmark for semantic segmentation. It contains 150 labels.
# 
# I've made a small subset just for demonstration purposes (namely the 10 first training and 10 first validation images + segmentation maps). The goal for the model is to overfit this tiny dataset (because that makes sure that it'll work on a larger scale).

# %%
# !rm -r /mnt/experiments/andream99/Hugging_face_cache_stuff/metrics    

# %%
import sys

sys.path.append("../")

# %%
import requests, zipfile, io
from datasets import load_dataset
import os
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


# %% [markdown]
# Note that this dataset is now also available on the hub :) you can directly check out the images [in your browser](scene_parse_150)! It can be easily loaded as follows (note that loading will take some time as the dataset is several GB's large):

# %% [markdown]
# ## Define PyTorch dataset and dataloaders
# 
# Here we define a [custom PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). Each item of the dataset consists of an image and a corresponding segmentation map.

# %%
from torch.utils.data import Dataset
import os
from PIL import Image


os.environ["HF_HOME"] = "/mnt/experiments/andream99/Hugging_face_cache_stuff"

class SemanticSegmentationDatasetSeruso(Dataset):
    """Image (semantic) segmentation dataset.""" 

    def __init__(self, root_dir, masks_dir, image_processor, classes_subfolders = ['before', 'after'], train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            image_processor (SegFormerImageProcessor): image processor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir # Thesis/Datasets/SERUSO_DATASETS/new_5000/three_classes_5000/before
        self.masks_dir = masks_dir
        self.image_processor = image_processor
        self.train = train

        sub_path = "training" if train else "validation"
        file_names = []

        for cls_subf in classes_subfolders:
            img_class_dir = os.path.join(sub_path, cls_subf)
            for video_name in sorted(os.listdir(os.path.join(masks_dir, img_class_dir))):
                video_folder = os.path.join(img_class_dir, video_name)
                for file_name in sorted(os.listdir(os.path.join(masks_dir, video_folder))):
                    file_path = os.path.join(video_folder, file_name)
                    file_path_no_ext = os.path.splitext(file_path)[0]  # Remove extension
                    file_names.append(file_path_no_ext)
              
        self.file_names = sorted(file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):

        rel_path = self.file_names[idx]
        image = Image.open(os.path.join(self.root_dir, f'{rel_path}.jpg'))
        segmentation_map = self.read_from_npz(os.path.join(self.masks_dir, f'{rel_path}.npz'))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.image_processor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs
    
    def read_from_npz(self, npz_path):

        # npz_path = os.path.join(folder, filename)
        array = np.load(npz_path)['array'] #+ 1

        array = (array).astype(np.uint8)

        img = Image.fromarray(array)

        return img

# %% [markdown]
# Let's initialize the training + validation datasets. Important: we initialize the image processor with `reduce_labels=True`, as the classes in ADE20k go from 0 to 150, with 0 meaning "background". However, we want the labels to go from 0 to 149, and only train the model to recognize the 150 classes (which don't include "background"). Hence, we'll reduce all labels by 1 and replace 0 by 255, which is the `ignore_index` of SegFormer's loss function.

# %%
from transformers import SegformerImageProcessor

root_dir = '/home/andream99/Thesis/Datasets/SERUSO_DATASETS/new_5000/bef_aft_5000'
masks_dir = '/home/andream99/Thesis/final_Git/Seruso_dataset/experiments/Puzzle-CAM/cams'
image_processor = SegformerImageProcessor() #reduce_labels=True)

train_dataset = SemanticSegmentationDatasetSeruso(root_dir=root_dir, masks_dir = masks_dir, classes_subfolders = ['before'], image_processor=image_processor)
valid_dataset = SemanticSegmentationDatasetSeruso(root_dir=root_dir, masks_dir = masks_dir, classes_subfolders = ['before'], image_processor=image_processor, train=False)

# %%
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(valid_dataset))

# %% [markdown]
# Let's verify a random example:

# %%
encoded_inputs = train_dataset[0]

# %%
encoded_inputs["pixel_values"].shape

# %%
encoded_inputs["labels"].shape

# %%
encoded_inputs["labels"]

# %%
encoded_inputs["labels"].squeeze().unique()

# %% [markdown]
# Next, we define corresponding dataloaders.

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16)

# %%
batch = next(iter(train_dataloader))

# %%
for k,v in batch.items():
  print(k, v.shape)

# %%
batch["labels"].shape

# %%
mask = (batch["labels"] != 255)
mask

# %%
batch["labels"][mask]

# %% [markdown]
# ## Define the model
# 
# Here we load the model, and equip the encoder with weights pre-trained on ImageNet-1k (we take the smallest variant, `nvidia/mit-b0` here, but you can take a bigger one like `nvidia/mit-b5` from the [hub](https://huggingface.co/models?other=segformer)). We also set the `id2label` and `label2id` mappings, which will be useful when performing inference.

# %%
from transformers import SegformerForSemanticSegmentation
import json
from huggingface_hub import hf_hub_download

# load id2label mapping from a JSON on the hub
repo_id = "huggingface/label-files"
filename = "ade20k-id2label.json"
id2label = json.load(open(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"), "r"))
id2label = {int(k): v for k, v in id2label.items()}
label2id = {v: k for k, v in id2label.items()}

# define model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0",
                                                         num_labels=150,
                                                         id2label=id2label,
                                                         label2id=label2id,
)

# %% [markdown]
# ## Fine-tune the model
# 
# Here we fine-tune the model in native PyTorch, using the AdamW optimizer. We use the same learning rate as the one reported in the [paper](https://arxiv.org/abs/2105.15203).
# 
# It's also very useful to track metrics during training. For semantic segmentation, typical metrics include the mean intersection-over-union (mIoU) and pixel-wise accuracy. These are available in the Datasets library. We can load it as follows:

# %%
import evaluate

metric = evaluate.load("mean_iou")

# %%
image_processor.do_reduce_labels

# %%
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm.notebook import tqdm
import os


best_val_score = float('-inf')  # Initialize the best validation score
model_path = "/home/andream99/Thesis/final_Git/Seruso_dataset/SegFormer/models/best_model_puzzzleCAM-3.pth"  # Path to save the best model

def load_model(model, model_path, parallel=False):
    if parallel:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path, parallel=False):
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

save_model_fn = lambda: save_model(model, model_path, parallel=False)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# try:
#     use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
# except KeyError:
#     use_gpu = '0'

# the_number_of_gpu = len(use_gpu.split(','))
# if the_number_of_gpu > 1:
#     # self.log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
#     model = nn.DataParallel(model)

model.to(device)


# %%
# f

# %%

model.train()
for epoch in range(20):  # Loop over the dataset multiple times
    print(f"Epoch {epoch + 1}/{20}")
    
    # Training loop
    for idx, batch in enumerate(tqdm(train_dataloader)):
        # Get the inputs
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # with torch.no_grad():
        #   upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        #   predicted = upsampled_logits.argmax(dim=1)

        #   # note that the metric expects predictions + labels as numpy arrays
        #   metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # Evaluate metrics during training (optional, e.g., every 100 steps)
        if idx % 100 == 0:
            with torch.no_grad():
                upsampled_logits = nn.functional.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = upsampled_logits.argmax(dim=1)
                metric.add_batch(
                    predictions=predicted.detach().cpu().numpy(),
                    references=labels.detach().cpu().numpy()
                )

                # Compute metrics
                metrics = metric._compute(
                    predictions=predicted.cpu(),
                    references=labels.cpu(),
                    num_labels=len(id2label),
                    ignore_index=255,
                    reduce_labels=False,  # We've already reduced the labels ourselves
                )
                print(f"Step {idx}, Loss: {loss.item()}, Mean_iou: {metrics['mean_iou']}, Mean accuracy: {metrics['mean_accuracy']}")

    # Validation loop
    model.eval()  # Switch to evaluation mode
    val_loss = 0
    val_metric = metric._compute(
        predictions=[],
        references=[],
        num_labels=len(id2label),
        ignore_index=255,
        reduce_labels=False,
    )  # Initialize empty metric container

    with torch.no_grad():
        for batch in tqdm(valid_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            val_loss += loss.item()

            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)

            metric.add_batch(
                predictions=predicted.detach().cpu().numpy(),
                references=labels.detach().cpu().numpy()
            )

        # Compute validation metrics
        val_loss /= len(valid_dataloader)  # Average validation loss
        val_metrics = metric._compute(
            predictions=predicted.cpu(),
            references=labels.cpu(),
            num_labels=len(id2label),
            ignore_index=255,
            reduce_labels=False,
        )
        print(f"Validation Loss: {val_loss}, Validation Mean_iou: {val_metrics['mean_iou']}, Validation Mean accuracy: {val_metrics['mean_accuracy']}")

    # Save the model if validation performance improves
    if val_metrics['mean_iou'] > best_val_score:  # Use your desired metric for comparison
        best_val_score = val_metrics['mean_iou']
        save_model_fn()
        print(f"New best model saved with Mean_iou: {best_val_score}")

    model.train()  # Switch back to training mode


# %%
# model_path = '/home/andream99/Thesis/final_Git/Seruso_dataset/SegFormer/models/best_model_puzzle.pth'
# model_path = "/home/andream99/Thesis/final_Git/Seruso_dataset/SegFormer/models/best_model_puzzzleCAM-3.pth"  # Path to save the best model


# %%
load_model_fn = lambda:load_model(model, model_path, parallel=False)
load_model_fn()

# %% [markdown]
# ## Inference
# 
# Finally, let's check whether the model has really learned something.
# 
# Let's test the trained model on an image (refer to my [inference notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb) for details):

# %%
image = Image.open('/home/andream99/Thesis/Datasets/SERUSO_DATASETS/test_set/images/before/before-video-002/frame_0002.jpg')
image

# %%
import torch

# %%
# def ade_palette():
#     """ADE20K palette that maps each class to RGB values."""
#     return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
#             [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
#             [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
#             [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
#             [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
#             [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
#             [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
#             [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
#             [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
#             [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
#             [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
#             [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
#             [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
#             [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
#             [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
#             [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
#             [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
#             [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
#             [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
#             [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
#             [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
#             [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
#             [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
#             [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
#             [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
#             [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
#             [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
#             [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
#             [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
#             [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
#             [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
#             [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
#             [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
#             [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
#             [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
#             [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
#             [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
#             [102, 255, 0], [92, 0, 255]]

# %%
def ade_palette():
    """ADE20K palette that maps only two classes to RGB values."""
    return [[120, 120, 120], [180, 120, 120]]


# %%
# prepare the image for the model
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)


# forward pass
with torch.no_grad():
  outputs = model(pixel_values=pixel_values)

  # logits are of shape (batch_size, num_labels, height/4, width/4)
logits = outputs.logits.cpu()
print(logits.shape)

predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
print(predicted_segmentation_map.shape)

# %%
import matplotlib.pyplot as plt
import numpy as np

color_seg = np.zeros((predicted_segmentation_map.shape[0],
                      predicted_segmentation_map.shape[1], 3), dtype=np.uint8) # height, width, 3

palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[predicted_segmentation_map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

# %%
np.max(predicted_segmentation_map)

# %%
plt.figure(figsize=(10, 10))  # Set the figure size
plt.imshow(predicted_segmentation_map, cmap='gray')  # Display the array as grayscale
# plt.colorbar()  # Optional: Add a colorbar to show intensity scale
plt.title("Predicted mask")
plt.axis('off')  # Optional: Turn off axes
plt.show()

# %% [markdown]
# Compare this to the ground truth segmentation map:

# %%
map = Image.open('/home/andream99/Thesis/Datasets/SERUSO_DATASETS/test_set/masks/before/before-video-002/frame_0002.jpg')
map

# %%


# %%
# # convert map to NumPy array
# map = np.array(map)
# # map[map == 0] = 255 # background class is replaced by ignore_index
# # map = map - 1 # other classes are reduced by one
# # map[map == 254] = 255

# # classes_map = np.unique(map).tolist()
# # unique_classes = [model.config.id2label[idx] if idx!=255 else None for idx in classes_map]
# # print("Classes in this image:", unique_classes)

# # create coloured map
# color_seg = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8) # height, width, 3
# palette = np.array(ade_palette())
# for label, color in enumerate(palette):
#     mask = (map == label)[:,:,0]

#     # print(mask[:,:,0].shape)
#     color_seg[mask] = color
# # Convert to BGR
# color_seg = color_seg[..., ::-1]

# # Show image + mask
# img = np.array(image) * 0.5 + color_seg * 0.5
# img = img.astype(np.uint8)

# plt.figure(figsize=(15, 10))
# plt.imshow(img)
# plt.show()

# %%
def get_mask_from_SegFormer(image):
    # prepare the image for the model
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    # print(pixel_values.shape)


    # forward pass
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)

    # logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()
    # print(logits.shape)

    predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
    # print(predicted_segmentation_map.shape)

    return predicted_segmentation_map




def compute_iou( predicted_mask_binary, ground_truth_mask):

        ground_truth_mask = np.array(ground_truth_mask)/255
        ground_truth_mask_binary = (ground_truth_mask > 0.5).astype(int)

        if isinstance(predicted_mask_binary, torch.Tensor):
            predicted_mask_binary = predicted_mask_binary.cpu().detach().numpy()
            # print("HEY")

        # Calculate the intersection and union
        intersection = np.logical_and(predicted_mask_binary, ground_truth_mask_binary)
        union = np.logical_or(predicted_mask_binary, ground_truth_mask_binary)

        # if isinstance(union, torch.Tensor):
        #     union_sum = torch.sum(union)
        # else:
        #     union_sum = np.sum(union)



        union_sum = np.sum(union)
        if union_sum == 0:
            iou = 1.0
        else:
            iou = float(np.sum(intersection) / union_sum)

        # Compute IoU
        # iou = float(torch.sum(intersection) / torch.sum(union))

        return iou

# %%
from seruso_datasets import SerusoTestDataset

from torchvision import transforms
from torch.utils.data import DataLoader

from general_utils.augment_utils import *

config = {
    'seed': 42,
    'architecture': 'resnet50',
    'mode': 'normal',
    'image_size': 512,
    'scales' : '0.2, 0.5, 1.0, 2.0, 4.0, 6.0',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'with_flows': False,
    'with_mask': True
}


root_with_temporal_labels = '/home/andream99/Thesis/Datasets/SERUSO_DATASETS/test_set'

batch_size = 64
image_size = 512

augment = 'colorjitter' #'colorjitter'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

test_transform = transforms.Compose([
    
    transforms.Resize(input_size),
    # Normalize(imagenet_mean, imagenet_std),
])

normalize_for_cams = True


test_dataset = SerusoTestDataset(img_root = root_with_temporal_labels, classes_subfolders = ['before'], transform= test_transform, with_flow = config['with_flows'], with_mask = config['with_mask'])


# %%
ious = []



for idx in range (len(test_dataset)):

    image, gt, path = test_dataset[idx]

    predicted = get_mask_from_SegFormer(image)

    ious.append(compute_iou(predicted, gt))

print (masks_dir)
print(f"Mean Iou before: {np.mean(ious)}")


# %%
from seruso_datasets import SerusoTestDataset

from torchvision import transforms
from torch.utils.data import DataLoader

from general_utils.augment_utils import *

config = {
    'seed': 42,
    'architecture': 'resnet50',
    'mode': 'normal',
    'image_size': 512,
    'scales' : '0.2, 0.5, 1.0, 2.0, 4.0, 6.0',
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225],
    'with_flows': False,
    'with_mask': True
}


root_with_temporal_labels = '/home/andream99/Thesis/Datasets/SERUSO_DATASETS/test_set'

batch_size = 64
image_size = 512

augment = 'colorjitter' #'colorjitter'

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

input_size = (image_size, image_size)

normalize_fn = Normalize(imagenet_mean, imagenet_std)

test_transform = transforms.Compose([
    
    transforms.Resize(input_size),
    # Normalize(imagenet_mean, imagenet_std),
])

normalize_for_cams = True


test_dataset = SerusoTestDataset(img_root = root_with_temporal_labels, classes_subfolders = ['after'], transform= test_transform, with_flow = config['with_flows'], with_mask = config['with_mask'])


# %%
ious = []



for idx in range (len(test_dataset)):

    image, gt, path = test_dataset[idx]

    predicted = get_mask_from_SegFormer(image)

    ious.append(compute_iou(predicted, gt))

print (masks_dir)
print(f"Mean Iou after: {np.mean(ious)}")



