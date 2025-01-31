# %% [markdown]
# ## Fine-tune SegFormer on a custom dataset
# 
# In this notebook, we are going to fine-tune [SegFormerForSemanticSegmentation](https://huggingface.co/docs/transformers/main/model_doc/segformer#transformers.SegformerForSemanticSegmentation) on a custom **semantic segmentation** dataset. In semantic segmentation, the goal for the model is to label each pixel of an image with one of a list of predefined classes.
# 
# We load the encoder of the model with weights pre-trained on ImageNet-1k, and fine-tune it together with the decoder head, which starts with randomly initialized weights.

# %%
# !pip install -q transformers datasets evaluate

# %% [markdown]
# ## Download toy dataset
# 
# Here we download a small subset of the ADE20k dataset, which is an important benchmark for semantic segmentation. It contains 150 labels.
# 
# I've made a small subset just for demonstration purposes (namely the 10 first training and 10 first validation images + segmentation maps). The goal for the model is to overfit this tiny dataset (because that makes sure that it'll work on a larger scale).

# %%
import requests, zipfile, io

# def download_data():
#     url = "https://www.dropbox.com/s/l1e45oht447053f/ADE20k_toy_dataset.zip?dl=1"
#     r = requests.get(url)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall()

# download_data()

# %% [markdown]
# Note that this dataset is now also available on the hub :) you can directly check out the images [in your browser](scene_parse_150)! It can be easily loaded as follows (note that loading will take some time as the dataset is several GB's large):

# %%
from datasets import load_dataset

# load_entire_dataset = False

# if load_entire_dataset:
#   dataset = load_dataset("scene_parse_150")

# %% [markdown]
# ## Define PyTorch dataset and dataloaders
# 
# Here we define a [custom PyTorch dataset](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html). Each item of the dataset consists of an image and a corresponding segmentation map.

# %%
import os

# root_dir = "/home/andream99/Thesis/Datasets/SERUSO_DATASETS/new_5000/three_classes_5000"
# masks_dir = "/home/andream99/Thesis/final_Git/Seruso_dataset/experiments/GradCAM/cams"

# root_dir = "/home/andream99/Thesis/final_Git/Seruso_dataset/experiments/GradCAM/cams"

# train = True

# classes_subfolders = ['before']

# sub_path = "training" if train else "validation"
# # img_dir = os.path.join(root_dir, sub_path)
# # ann_dir = os.path.join(masks_dir, "cams", f"{sub_path}_maps")

# # read images
# file_names = []

# for cls_subf in classes_subfolders:
#     img_class_dir = os.path.join(sub_path, cls_subf)
#     for video_name in sorted(os.listdir(os.path.join(root_dir, img_class_dir))):
#         video_folder = os.path.join(img_class_dir, video_name)
#         for file_name in sorted(os.listdir(os.path.join(root_dir, video_folder))):
#             # image_file_names.append(os.path.join(video_folder, file_name))
#             file_path = os.path.join(video_folder, file_name)
#             file_path_no_ext = os.path.splitext(file_path)[0]  # Remove extension
#             file_names.append(file_path_no_ext)
# print(file_names)


# %%
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# # Load .npz file

# output_dir = 'output_images'
# os.makedirs(output_dir, exist_ok=True)

# folder = 'before-video-000 copy'



# files = os.listdir(folder)
# num_files = len(files)

# fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))

# for ax, filename in zip(axes, files):
#     npz_path = os.path.join(folder, filename)
#     array = np.load(npz_path)['array']

#     print(f"array.shape: {array.shape}")
#     print(f"array.type: {type(array)}")

#     array = (array ).astype(np.uint8)
#     image = Image.fromarray(array)
#     image.save(os.path.join(output_dir, f'{filename[:-4]}_gnu.png'))

#     ax.imshow(image, cmap='gray', interpolation='nearest')
#     ax.set_title("2D Binary Array")
#     ax.axis('off')  # Hide axis for better visualization
    

# plt.tight_layout()
# plt.show()

# print("EEEEELLLLAMADDONNNNAA")


# folder = 'before-video-000 copy'



# files = os.listdir(folder)
# num_files = len(files)

# fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))

# for ax, filename in zip(axes, files):
#     npz_path = os.path.join(folder, filename)
#     array = np.load(npz_path)['array']

#     ax.imshow(array, cmap='gray', interpolation='nearest')
#     ax.set_title("2D Binary Array")
#     ax.axis('off')  # Hide axis for better visualization

# plt.tight_layout()
# plt.show()


# folder = 'before-video-224'



# files = os.listdir(folder)
# num_files = len(files)

# fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))

# for ax, filename in zip(axes, files):
#     npz_path = os.path.join(folder, filename)
#     array = np.load(npz_path)['array']

#     ax.imshow(array, cmap='gray', interpolation='nearest')
#     ax.set_title("2D Binary Array")
#     ax.axis('off')  # Hide axis for better visualization

# plt.tight_layout()
# plt.show()

# folder = 'before-video-224 copy'



# files = os.listdir(folder)
# num_files = len(files)

# fig, axes = plt.subplots(1, num_files, figsize=(5 * num_files, 5))

# for ax, filename in zip(axes, files):
#     npz_path = os.path.join(folder, filename)
#     array = np.load(npz_path)['array']

#     ax.imshow(array, cmap='gray', interpolation='nearest')
#     ax.set_title("2D Binary Array")
#     ax.axis('off')  # Hide axis for better visualization

# plt.tight_layout()
# plt.show()



# print("Conversion complete.")

# %%
from torch.utils.data import Dataset
import os
from PIL import Image


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        # img_dir = os.path.join(root_dir, sub_path)
        # ann_dir = os.path.join(masks_dir, "cams", f"{sub_path}_maps")

        # read images
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

            # for root, dirs, files in os.walk(img_class_dir):
            #   image_file_names.extend(files)
            # self.images = sorted(image_file_names)

        # read annotations
        # annotation_file_names = []
        # for root, dirs, files in os.walk(self.ann_dir):
        #   annotation_file_names.extend(files)
        # self.annotations = sorted(annotation_file_names)

        # assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

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
        array = np.load(npz_path)['array']

        array = (array).astype(np.uint8)
        img = Image.fromarray(array)

        return img

# %% [markdown]
# Let's initialize the training + validation datasets. Important: we initialize the image processor with `reduce_labels=True`, as the classes in ADE20k go from 0 to 150, with 0 meaning "background". However, we want the labels to go from 0 to 149, and only train the model to recognize the 150 classes (which don't include "background"). Hence, we'll reduce all labels by 1 and replace 0 by 255, which is the `ignore_index` of SegFormer's loss function.

# %%
from transformers import SegformerImageProcessor

root_dir = '/home/andream99/Thesis/Datasets/SERUSO_DATASETS/new_5000/bef_aft_5000'
masks_dir = '/home/andream99/Thesis/final_Git/Seruso_dataset/experiments/GradCAM/cams'
image_processor = SegformerImageProcessor(reduce_labels=True)

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

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2)

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

# define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
# move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()
for epoch in range(20):  # loop over the dataset multiple times
   print("Epoch:", epoch)
   for idx, batch in enumerate(tqdm(train_dataloader)):
        # get the inputs;
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss, logits = outputs.loss, outputs.logits

        loss.backward()
        optimizer.step()

        # evaluate
        with torch.no_grad():
          upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
          predicted = upsampled_logits.argmax(dim=1)

          # note that the metric expects predictions + labels as numpy arrays
          metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())

        # let's print loss and metrics every 100 batches
        if idx % 100 == 0:
          # currently using _compute instead of compute
          # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
          metrics = metric._compute(
                  predictions=predicted.cpu(),
                  references=labels.cpu(),
                  num_labels=len(id2label),
                  ignore_index=255,
                  reduce_labels=False, # we've already reduced the labels ourselves
              )

          print("Loss:", loss.item())
          print("Mean_iou:", metrics["mean_iou"])
          print("Mean accuracy:", metrics["mean_accuracy"])


def save_model(model, model_path, parallel=False):
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

save_model(model, model_path = '/home/andream99/Thesis/final_Git/Seruso_dataset/SegFormer/seg_model.pth')

f

# %% [markdown]
# ## Inference
# 
# Finally, let's check whether the model has really learned something.
# 
# Let's test the trained model on an image (refer to my [inference notebook](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SegFormer/Segformer_inference_notebook.ipynb) for details):

# %%
image = Image.open('/content/ADE20k_toy_dataset/images/training/ADE_train_00000001.jpg')
image

# %%
# prepare the image for the model
pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
print(pixel_values.shape)

# %%
import torch

# forward pass
with torch.no_grad():
  outputs = model(pixel_values=pixel_values)

# %%
# logits are of shape (batch_size, num_labels, height/4, width/4)
logits = outputs.logits.cpu()
print(logits.shape)

# %%
def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

# %%
predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
print(predicted_segmentation_map)

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
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

# %% [markdown]
# Compare this to the ground truth segmentation map:

# %%
map = Image.open('/content/ADE20k_toy_dataset/annotations/training/ADE_train_00000001.png')
map

# %%
# convert map to NumPy array
map = np.array(map)
map[map == 0] = 255 # background class is replaced by ignore_index
map = map - 1 # other classes are reduced by one
map[map == 254] = 255

classes_map = np.unique(map).tolist()
unique_classes = [model.config.id2label[idx] if idx!=255 else None for idx in classes_map]
print("Classes in this image:", unique_classes)

# create coloured map
color_seg = np.zeros((map.shape[0], map.shape[1], 3), dtype=np.uint8) # height, width, 3
palette = np.array(ade_palette())
for label, color in enumerate(palette):
    color_seg[map == label, :] = color
# Convert to BGR
color_seg = color_seg[..., ::-1]

# Show image + mask
img = np.array(image) * 0.5 + color_seg * 0.5
img = img.astype(np.uint8)

plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.show()

# %% [markdown]
# Let's compute the metrics:

# %%
# metric expects a list of numpy arrays for both predictions and references
metrics = metric._compute(
                  predictions=[predicted_segmentation_map],
                  references=[map],
                  num_labels=len(id2label),
                  ignore_index=255,
                  reduce_labels=False, # we've already reduced the labels ourselves
              )

# %%
metrics.keys()

# %%
import pandas as pd

# print overall metrics
for key in list(metrics.keys())[:3]:
  print(key, metrics[key])

# pretty-print per category metrics as Pandas DataFrame
metric_table = dict()
for id, label in id2label.items():
    metric_table[label] = [
                           metrics["per_category_iou"][id],
                           metrics["per_category_accuracy"][id]
    ]

print("---------------------")
print("per-category metrics:")
pd.DataFrame.from_dict(metric_table, orient="index", columns=["IoU", "accuracy"])

# %%



