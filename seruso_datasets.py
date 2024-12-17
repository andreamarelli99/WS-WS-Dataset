import os
import cv2
import math
import random
import glob
import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from glob import glob
import os.path as osp

from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image

from POF_CAM.POFCAM_utils import frame_utils, torch_utils
from POF_CAM.POFCAM_utils.augment_utils import *
from imageio import imread

from torchvision import transforms



def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data



class CombinedDataset(data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __len__(self):
        length = 0 
        for i in range(len(self.datasets)):
            length += len(self.datsaets[i])
        return length

    def __getitem__(self, index):
        i = 0
        for j in range(len(self.datasets)):
            if i + len(self.datasets[j]) >= index:
                yield self.datasets[j][index-i]
                break
            i += len(self.datasets[j])

    def __add__(self, other):
        self.datasets.append(other)
        return self
    




class CamFlowDataset(data.Dataset):
    def __init__(self,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None,
                 augment = False,
                 return_path= False, 
                 return_img_path= False):

        self.return_path = return_path
        self.return_img_path = return_img_path

        self.augment = augment
        self.transform = transform
    
        self.loader = loader
        
        self.image_list = []
        self.flow_list = []
        self.label_list = []

    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):

        np.random.seed()

        if index!=(index % len(self.image_list)): assert NotImplementedError
        
        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])

        flow1 = frame_utils.read_gen(self.flow_list[index][0])
        flow2 = frame_utils.read_gen(self.flow_list[index][1])

        flow1 = -np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)
        
        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

        
        label = torch_utils.one_hot_embedding(self.class_dic[self.label_list[index]], self.classes)
        

        if self.augment:

            horizontal_flips = []
            vertical_flips = []
            rotation_degrees = []

            for i in range (3):

                horizontal_flips.append(random.getrandbits(1))
                vertical_flips.append(random.getrandbits(1))
                rotation_degrees.append(random.randint(-90,90))

            params = [horizontal_flips, vertical_flips, rotation_degrees]
            
            
            stored_transform = Three_images_trasform(params)
            img1, img2, img3 = stored_transform([img1, img2, img3])

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

            return [img1, img2, img3], [flow1, flow2], label, params
        
        
        else:

            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

            return [img1, img2, img3], [flow1, flow2], label


    def __len__(self):
        return len(self.image_list)

    def __add(self, other):
        return CombinedDataset([self, other])



class Seruso_three_classes_flow(CamFlowDataset):
    def __init__(self, 
                 img_root = '../../../../Datasets/SERUSO_DATASETS/main_dataset/Before_after_no_backgrounds/',
                 flow_root = '../../../../Datasets/SERUSO_DATASETS/main_dataset/optical_flows/',
                 dstype = 'training', 
                 transform: Optional[Callable] = None,
                 augment = False,
                 loader: Callable[[str], Any] = pil_loader,):
        super(Seruso_three_classes_flow, self).__init__(transform = transform, augment = augment, loader = loader)

        self.img_root = img_root
        self.dstype = dstype

        img_dir = dstype 
        assert(os.path.isdir(os.path.join(self.img_root,img_dir)))

        images = []
        self.class_names = []
        self.class_dic = {}

        for class_name in sorted(os.listdir(os.path.join(img_root, img_dir))):

            self.class_names.append(class_name)
            self.class_dic[class_name] = len(self.class_dic)

            for rel_frame_dir in sorted(glob(os.path.join(img_root, img_dir, class_name,'*','*.jpg'))):
                    
                rel_frame_dir = os.path.relpath(rel_frame_dir,os.path.join(img_root, img_dir, class_name))
                scene_dir, filename = os.path.split(rel_frame_dir)
                no_ext_filename = os.path.splitext(filename)[0]
                prefix, frame_nb = no_ext_filename.split('_')
                frame_nb = int(frame_nb)
                img1 = os.path.join(img_dir, class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb))
                img2 = os.path.join(img_dir, class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 1))
                img3 = os.path.join(img_dir, class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 2))
                flow1 = os.path.join(img_dir, class_name, scene_dir, '{}_{:04d}.flo'.format(prefix, frame_nb))
                flow2 = os.path.join(img_dir, class_name, scene_dir, '{}_{:04d}.flo'.format(prefix, frame_nb + 1))

                if (os.path.isfile(os.path.join(img_root,img1)) and os.path.isfile(os.path.join(img_root,img2)) and os.path.isfile(os.path.join(img_root,img3))):
                    images.append([[img1, img2, img3], [flow1, flow2], class_name])
        
        self.classes = len(self.class_dic)

        # Use split2list just to ensure the same data structure; actually we do not split here
        tbd_list, _ = split2list(images, split=1.1, default_split=1.1,order=True)

        self.image_list = []
        self.flow_list = []
        self.label_list = []

        for i in range(len(tbd_list)):

            im1 = os.path.join(img_root,tbd_list[i][0][0])
            im2 = os.path.join(img_root,tbd_list[i][0][1])
            im3 = os.path.join(img_root,tbd_list[i][0][2])

            flo1 = os.path.join(flow_root,tbd_list[i][1][0])
            flo2 = os.path.join(flow_root,tbd_list[i][1][1])

            label_name = tbd_list[i][2]

            self.image_list.append([im1, im2, im3])
            self.flow_list.append([flo1, flo2])
            self.label_list.append(label_name)





class CamFolderDataset(data.Dataset):
    def __init__(self,
                 loader: Callable[[str], Any],
                 transform: Optional[Callable] = None):
        
        self.transform = transform
        self.loader = loader
        
        self.image_list = []
        self.label_list = []

    def get_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

    def __getitem__(self, index):

        np.random.seed()

        if index!=(index % len(self.image_list)): assert NotImplementedError
        
        index = index % len(self.image_list)

        img = frame_utils.read_gen(self.image_list[index])
                                    

        if self.transform is not None:
            img = self.transform(img)

        label = torch_utils.one_hot_embedding(self.class_dic[self.label_list[index]], self.classes)
         
        return img, label

    def __len__(self):
        return len(self.image_list)

    def __add(self, other):
        return CombinedDataset([self, other])



class Seruso_three_classes(CamFolderDataset):
    def __init__(self, 
                 img_root = '../../../../Datasets/SERUSO_DATASETS/main_dataset/Before_after_no_backgrounds/',
                 transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = pil_loader,):
        super(Seruso_three_classes, self).__init__(transform = transform, loader = loader)

        self.img_root = img_root

        assert(os.path.isdir(os.path.join(self.img_root)))

        images = []
        self.class_names = []
        self.class_dic = {}

        for subfolder in sorted(os.listdir(os.path.join(img_root))):

            for class_name in sorted(os.listdir(os.path.join(img_root, subfolder))):

                if class_name not in self.class_names:
                
                    self.class_names.append(class_name)
                    self.class_dic[class_name] = len(self.class_dic)

                for rel_frame_dir in sorted(glob(os.path.join(img_root, subfolder, class_name,'*','*.jpg'))):
                        
                    rel_frame_dir = os.path.relpath(rel_frame_dir,os.path.join(img_root, '*', class_name))
                    scene_dir, filename = os.path.split(rel_frame_dir)
                    no_ext_filename = os.path.splitext(filename)[0]
                    prefix, frame_nb = no_ext_filename.split('_')
                    frame_nb = int(frame_nb)
                    img = os.path.join(subfolder, class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb))

                    if (os.path.isfile(os.path.join(img_root,img))):
                        images.append([img, class_name])
        
        self.classes = len(self.class_dic)

        # Use split2list just to ensure the same data structure; actually we do not split here
        tbd_list, _ = split2list(images, split=1.1, default_split=1.1,order=True)

        self.image_list = []
        self.label_list = []

        for i in range(len(tbd_list)):

            im1 = os.path.join(img_root,tbd_list[i][0])

            label_name = tbd_list[i][1]

            self.image_list.append(im1)
            self.label_list.append(label_name)





class SerusoTestDataset(CamFlowDataset):
    def __init__(self,
                 img_root = '../../../../Datasets/SERUSO_DATASETS/test_set',
                 classes_subfolders = ['before', 'after'],
                 transform: Optional[Callable] = None):
    
        self.transform = transform

        self.img_root = img_root

        assert(os.path.isdir(os.path.join(self.img_root)))

        images = []

        for class_name in sorted(os.listdir(os.path.join(img_root, "images"))):

            if class_name in classes_subfolders:

                for rel_frame_dir in sorted(glob(os.path.join(img_root, "images", class_name,'*','*.jpg'))):
                        
                    rel_frame_dir = os.path.relpath(rel_frame_dir,os.path.join(img_root, "images", class_name))
                    scene_dir, filename = os.path.split(rel_frame_dir)
                    no_ext_filename = os.path.splitext(filename)[0]
                    prefix, frame_nb = no_ext_filename.split('_')
                    frame_nb = int(frame_nb)
                    img1 = os.path.join("images", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb))
                    img2 = os.path.join("images", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 1))
                    img3 = os.path.join("images", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 2))
                    flow1 = os.path.join("optical_flows", class_name, scene_dir, '{}_{:04d}.flo'.format(prefix, frame_nb))
                    flow2 = os.path.join("optical_flows", class_name, scene_dir, '{}_{:04d}.flo'.format(prefix, frame_nb + 1))
                    mask1 = os.path.join("masks", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb))
                    mask2 = os.path.join("masks", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 1))
                    mask3 = os.path.join("masks", class_name, scene_dir, '{}_{:04d}.jpg'.format(prefix, frame_nb + 2))

                    if (os.path.isfile(os.path.join(img_root,img1)) and os.path.isfile(os.path.join(img_root,img2)) and os.path.isfile(os.path.join(img_root,img3))):
                        images.append([[img1, img2, img3], [flow1, flow2], [mask1, mask2, mask3]])


        # Use split2list just to ensure the same data structure; actually we do not split here
        tbd_list, _ = split2list(images, split=1.1, default_split=1.1,order=True)

        self.image_list = []
        self.mask_list = []
        self.flow_list = []

        for i in range(len(tbd_list)):

            im1 = os.path.join(img_root,tbd_list[i][0][0])
            im2 = os.path.join(img_root,tbd_list[i][0][1])
            im3 = os.path.join(img_root,tbd_list[i][0][2])

            flo1 = os.path.join(img_root,tbd_list[i][1][0])
            flo2 = os.path.join(img_root,tbd_list[i][1][1])

            msk1 = os.path.join(img_root,tbd_list[i][2][0])
            msk2 = os.path.join(img_root,tbd_list[i][2][1])
            msk3 = os.path.join(img_root,tbd_list[i][2][2])

            self.image_list.append([im1, im2, im3])
            self.flow_list.append([flo1, flo2])
            self.mask_list.append([msk1, msk2, msk3])


    def get_mask(self, mask_path):

        mask = Image.open(mask_path).convert("L")

        return mask

    def __getitem__(self, index):

        np.random.seed()

        if index!=(index % len(self.image_list)): assert NotImplementedError
        
        index = index % len(self.image_list)

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        img3 = frame_utils.read_gen(self.image_list[index][2])

        flow1 = frame_utils.read_gen(self.flow_list[index][0])
        flow2 = frame_utils.read_gen(self.flow_list[index][1])

        flow1 = np.array(flow1).astype(np.float32)
        flow2 = np.array(flow2).astype(np.float32)

        mask1 = self.get_mask(self.mask_list[index][0]) 
        mask2 = self.get_mask(self.mask_list[index][1])
        mask3 = self.get_mask(self.mask_list[index][2])

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

            mask1 = self.transform.transforms[0](mask1)
            mask2 = self.transform.transforms[0](mask2)
            mask3 = self.transform.transforms[0](mask3)         


        flow1 = torch.from_numpy(flow1).permute(2, 0, 1).float()
        flow2 = torch.from_numpy(flow2).permute(2, 0, 1).float()

        return [img1, img2, img3], [flow1, flow2], [mask1, mask2, mask3]




class LabeledDataset(data.Dataset):
    def __init__(self,
                 img_root = '../../../../Datasets/SERUSO_DATASETS/main_dataset/Labels/',
                 classes_subfolders = ['before', 'after'],
                 transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = pil_loader):
    
        self.transform = transform
        self.loader = loader
        
        self.image_list = []
        self.masks_list = []

        self.img_root = img_root

        assert(os.path.isdir(os.path.join(self.img_root)))

        images = []

        for class_subf in classes_subfolders:

            for rel_frame_dir in sorted(glob(os.path.join(img_root, 'images', class_subf, '*.jpg'))):

                rel_frame_dir = os.path.relpath(rel_frame_dir, os.path.join(img_root, '*'))

                class_dir, filename = os.path.split(rel_frame_dir)

                if class_subf in class_dir:
                    img = os.path.join('images', class_subf, filename)
                    mask = os.path.join('masks', class_subf, filename)

                    if (os.path.isfile(os.path.join(img_root,img))):
                        images.append([img, mask])



        # Use split2list just to ensure the same data structure; actually we do not split here
        tbd_list, _ = split2list(images, split=1.1, default_split=1.1,order=True)

        self.image_list = []
        self.label_list = []

        for i in range(len(tbd_list)):

            img = os.path.join(img_root, tbd_list[i][0])
            mask = os.path.join(img_root, tbd_list[i][1])

            self.image_list.append(img)
            self.masks_list.append(mask)


    def get_mask(self, mask_path):

        mask = Image.open(mask_path).convert("L")

        return mask
    

    def __getitem__(self, index):

        np.random.seed()

        if index!=(index % len(self.image_list)): assert NotImplementedError
        
        index = index % len(self.image_list)

        img = frame_utils.read_gen(self.image_list[index])
        mask = self.get_mask(self.masks_list[index])
                                    
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
         
        return img, mask

    def __len__(self):
        return len(self.image_list)

    def __add(self, other):
        return CombinedDataset([self, other])






def split2list(images, split, default_split=1.1,order = False):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        # assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) < split
    else:
        split_values = np.random.uniform(0,1,len(images)) < default_split
    
    if (not isinstance(split, str)) and (order==True):
        if isinstance(split, float):
            check_split = split
        else:
            check_split = default_split

        split_values = np.ones(len(images))==1
        split_values[int(len(images)*check_split):]=False
    
    if len(split_values)!=len(images):
        import pdb;pdb.set_trace()

    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    
    return train_samples, test_samples