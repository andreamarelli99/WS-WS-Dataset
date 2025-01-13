import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from ultralytics import SAM, FastSAM
import skimage.transform as st


class SAME:
    def __init__(self, model_path="sam2.1_t.pt", fast_SAM=False, shape=(512,512), number_classes=2):
        """
        Parameters:
        -----------
        """
        if fast_SAM:
            self.model = FastSAM("FastSAM-s.pt")
        else:
            self.model = SAM(model_path)
        self.shape = shape
        self.merger = MaxIoU_IMP2(number_classes)

    def compute_masks_direct(self, image):
                
        # Generate the mask using the SAM model
        masks = self.model(image)
        # resize the SAM prediction
        masks = np.array([st.resize(tmp, self.shape, order=0, preserve_range=True, anti_aliasing=False) for tmp in masks[0].masks.data.cpu().numpy()])
        return masks
    
    def merge_masks_direct(self, masks_sam, mask_original):
        if(mask_original.shape != self.shape):
            mask_original = st.resize(mask_original, self.shape, order=0, preserve_range=True, anti_aliasing=False)

        if(masks_sam.shape[1:] != self.shape):
            masks_sam = np.array([st.resize(tmp, self.shape, order=0, preserve_range=True, anti_aliasing=False) for tmp in masks_sam])
        
        mask_enhanced = self.merger.merge(mask_original, masks_sam)

        return mask_enhanced


    def compute_masks(self, origin_path, destination_path="dataset/SAM_masks"):
        """
        Process images from origin_path, create masks using the SAM model, and save them in destination_path.

        Args:
            origin_path (str): Path to the folder containing images.
            destination_path (str): Path to the folder where masks will be saved.
        """
        # check if the origin path exists  
        if(not os.path.exists(origin_path)):
            raise Warning("origin path not found " + origin_path)
        # if the destination folder doesn't exist create it 
        if(not os.path.exists(destination_path)):
            os.makedirs(destination_path)
            
        # Walk through the origin directory
        for root, _, files in tqdm(os.walk(origin_path)):
            # Determine the relative path
            relative_path = os.path.relpath(root, origin_path)

            # Create the corresponding destination path
            destination_dir = os.path.join(destination_path, relative_path)
            os.makedirs(destination_dir, exist_ok=True)

            for file in files:
                # Process only image files (you can extend this list as needed)
                if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                    # Full path to the input image
                    image_path = os.path.join(root, file)

                    # Load the image
                    image = Image.open(image_path).convert("RGB")
                    
                    # Generate the mask using the SAM model
                    masks = self.model(image)
                    # resize the SAM prediction
                    masks = np.array([st.resize(tmp, self.shape, order=0, preserve_range=True, anti_aliasing=False) for tmp in masks[0].masks.data.cpu().numpy()])

                    
                    # Save the mask as a .npz file in the destination folder
                    mask_save_path = os.path.join(destination_dir, f"{os.path.splitext(file)[0]}.npz")
                    np.savez_compressed(mask_save_path, data=masks)


    def merge_masks(self, origin_path, sam_path, destination_path, number_classes=2):
        
        # check if the origin path exists  
        if(not os.path.exists(origin_path)):
            raise Warning("origin path not found " + origin_path)
        if(not os.path.exists(sam_path)):
            raise Warning("origin path not found " + sam_path)
        
        # if the destination folder doesn't exist create it 
        if(not os.path.exists(destination_path)):
            os.makedirs(destination_path)
            
        # Walk through the origin directory
        for root, _, files in tqdm(os.walk(origin_path)):
            # Determine the relative path
            relative_path = os.path.relpath(root, origin_path)

            # Determin the corresponding SAM path
            relative_sam_path = os.path.join(sam_path, relative_path)

            # Create the corresponding destination path
            destination_dir = os.path.join(destination_path, relative_path)
            os.makedirs(destination_dir, exist_ok=True)

            for file in files:
                # Process only image files (you can extend this list as needed)
                if file.lower().endswith(".npz"):

                    print(file)

                    # Full path to the input image
                    mask_origin_path = os.path.join(root, file)
                    name_file_npz = os.path.splitext(file)[0] + ".npz"
                    mask_sam_path = os.path.join(relative_sam_path, name_file_npz)
                    
                    if(not os.path.exists(mask_sam_path)):
                        raise Warning("SAM mask not found " + mask_sam_path)
                    
                    print(mask_origin_path)
                    
                    # Load the orginal mask and the SAM masks
                    mask_original = np.load(mask_origin_path)["array"]
                    if(mask_original.shape != self.shape):
                        mask_original = st.resize(mask_original, self.shape, order=0, preserve_range=True, anti_aliasing=False)
                    masks_sam = np.load(mask_sam_path)["data"]
                    if(masks_sam.shape[1:] != self.shape):
                        masks_sam = np.array([st.resize(tmp, self.shape, order=0, preserve_range=True, anti_aliasing=False) for tmp in masks_sam])
                    
                    mask_enhanced = self.merger.merge(mask_original, masks_sam)
                    
                    # Save the mask as a .npz file in the destination folder
                    mask_save_path = os.path.join(destination_dir, name_file_npz)
                    np.savez_compressed(mask_save_path, data=mask_enhanced)

    def plot_file_direct(image, mask, mask_enhanced, mask_gt=None):
        """
        Plot the original image, the original mask, the SAM mask, and the enhanced mask.

        Args:
            image_path (str): Path to the image.
            mask_path (str): Path to the original mask.
            mask_enhanced_path (str): Path to the enhanced mask.
            groundtrugh_path (str): Path to the ground truth mask.
        """
        # Load the ground truth mask
        if mask_gt is not None:
            mask_gt = st.resize(np.array(mask_gt.convert("L")), mask_enhanced.shape, order=0, preserve_range=True, anti_aliasing=False)

        if mask_gt is None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        axs[0].imshow(image)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        if mask_gt is None:
            axs[1].set_title("Original Mask")
        else:
            axs[1].set_title("Original Mask (IoU: {:.2f})".format(SAME.compute_iou(mask, mask_gt)))
        axs[1].axis("off")

        axs[2].imshow(mask_enhanced, cmap="gray")
        if mask_gt is None:
            axs[2].set_title("Enhanced Mask")
        else:
            axs[2].set_title("Enhanced Mask (IoU: {:.2f})".format(SAME.compute_iou(mask_enhanced, mask_gt)))
        axs[2].axis("off")

        if mask_gt is not None:
            axs[3].imshow(mask_gt, cmap="gray")
            axs[3].set_title("Ground Truth Mask")
            axs[3].axis("off")

        plt.show()

    @staticmethod
    def plot_file(image_path, mask_path, mask_enhanced_path, groundtrugh_path=None):
        """
        Plot the original image, the original mask, the SAM mask, and the enhanced mask.

        Args:
            image_path (str): Path to the image.
            mask_path (str): Path to the original mask.
            mask_enhanced_path (str): Path to the enhanced mask.
            groundtrugh_path (str): Path to the ground truth mask.
        """
        # Load the image
        image = Image.open(image_path)
        # Load the original mask
        mask = np.load(mask_path)["array"]
        # Load the enhanced mask
        mask_enhanced = np.load(mask_enhanced_path)["data"]
        # Load the ground truth mask
        if groundtrugh_path is not None:
            mask_gt = st.resize(np.array(Image.open(groundtrugh_path).convert("L")), mask_enhanced.shape, order=0, preserve_range=True, anti_aliasing=False)

        if groundtrugh_path is None:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        else:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

        axs[0].imshow(image)
        axs[0].set_title("Image")
        axs[0].axis("off")

        axs[1].imshow(mask, cmap="gray")
        if groundtrugh_path is None:
            axs[1].set_title("Original Mask")
        else:
            axs[1].set_title("Original Mask (IoU: {:.2f})".format(SAME.compute_iou(mask, mask_gt)))
        axs[1].axis("off")

        axs[2].imshow(mask_enhanced, cmap="gray")
        if groundtrugh_path is None:
            axs[2].set_title("Enhanced Mask")
        else:
            axs[2].set_title("Enhanced Mask (IoU: {:.2f})".format(SAME.compute_iou(mask_enhanced, mask_gt)))
        axs[2].axis("off")

        if groundtrugh_path is not None:
            axs[3].imshow(mask_gt, cmap="gray")
            axs[3].set_title("Ground Truth Mask")
            axs[3].axis("off")

        plt.show()
    
    @staticmethod
    def compute_iou(mask1, mask2):
        """
        Compute the Intersection over Union (IoU) between two masks.

        Args:
            mask1 (np.ndarray): First binary mask.
            mask2 (np.ndarray): Second binary mask.

        Returns:
            float: IoU score.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0.0

class MaxIoU_IMP2():
    def __init__(self, num_cls=2):
        self.num_cls = num_cls

    def merge(self, input_cam, sam_masks, prediction_threshold=0.85):
        """
        input_cam: np.array
            numpy array of the cam
        name: str
            name of the image
        sam_folder: str
            path to the sam folder
        save_path: str
            path to the folder on where to save the image
        """
        
        processed_mask = np.zeros_like(input_cam)

        
        
        for i in range(1, self.num_cls):
            pre_cls = input_cam == i
            if np.sum(pre_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pre_cls)
            for cur_sam in sam_masks:
                # TODO: check if SAM is background=0 and object 1 or viceversa
                # (in case it SHOULD be enough to change the 255 to 0)
                cur_sam = cur_sam == 1
                sam_mask = np.logical_or(sam_mask, cur_sam)
                # intersection between pre_cls
                improve_thresh = 2 * np.sum((pre_cls == cur_sam) * pre_cls) - np.sum(cur_sam)
                #improve = np.sum((pre_cls == cur) * pre_cls) / np.sum(cur) （>0.5）
                # Note that the two way calculating (improve) are equivalent
                improve_pred_thresh = np.sum((pre_cls == cur_sam) * pre_cls) / np.sum(pre_cls)
                if improve_thresh > 0 or improve_pred_thresh >= prediction_threshold:

                    candidates.append(cur_sam)
                    iou += np.sum(pre_cls == cur_sam)
            cam_mask = np.logical_and(sam_mask==0, pre_cls==1)
            # Trust CAM if SAM has no prediction on that pixel
            candidates.append(cam_mask)
            processed_mask[np.sum(candidates, axis=0) > 0] = i
            """cur = np.array(Image.open(filename.path)) == 0
            intersection = np.logical_and(cur, input_cam)
            union = np.logical_or(cur, input_cam)
            iou_single = np.sum(intersection) / np.sum(union)
            # overlap_ratio = np.sum((pre_cls == cur) * pre_cls) / np.sum(pre_cls)
            # if improve > 0 or overlap_ratio >= self.threshold:

            if iou_single > 0.01:
                #plt.imshow(cur, cmap='gray')
                #plt.show()
                candidates.append(cur)
                seen.append(filename.path)"""

            processed_mask[np.sum(candidates, axis=0) > 0] = i

        
        return processed_mask
        #plt.show()




# merge = Merger(args.merge_type)
# segmentation_prediction = []

# for image_path in data_path_list:
#     # loads cam
#     computed_cam = etl.load_npy(os.path.join(args.path_prefix_output, args.data_folder_name, 'cam', args.cam_type), os.path.join(os.path.basename(image_path)+'.npy'))
#     # loads sam
#     # TODO: loads sam
#     segmentation_prediction.append(merge.merge(input_cam=computed_cam, input_sam=computed_sam))