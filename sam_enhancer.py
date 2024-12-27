import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from ultralytics import SAM, FastSAM

class SAME:
    def __init__(self, model_path="sam2.1_t.pt", fast_SAM=False):
        """
        Parameters:
        -----------
        """
        if fast_SAM:
            self.model = FastSAM("FastSAM-s.pt")
        else:
            self.model = SAM(model_path)

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

                    # Save the mask as a .npz file in the destination folder
                    mask_save_path = os.path.join(destination_dir, f"{os.path.splitext(file)[0]}.npz")
                    np.savez_compressed(mask_save_path, bool_array=masks[0].masks.data.cpu().numpy())

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
    
# import numpy as np
# from MERGE.merge import Merger
# from PIL import Image


class MaxIoU_IMP2(Merger):
    def __init__(self, num_cls=2):
        self.num_cls = num_cls

    def merge(self, input_cam, name, sam_folder, save_path, plot=False):
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
        seen = []
        processed_mask = np.zeros_like(input_cam)

        for i in range(1, self.num_cls):
            pre_cls = input_cam == i
            if np.sum(pre_cls) == 0:
                continue
            iou = 0
            candidates = []
            sam_mask = np.zeros_like(pre_cls)
            for filename in os.scandir(sam_folder):
                # TODO: check if SAM is background=0 and object 1 or viceversa
                # (in case it SHOULD be enough to change the 255 to 0)
                cur_sam = np.array(Image.open(filename.path)) == 0
                sam_mask = np.logical_or(sam_mask, cur_sam)
                # intersection between pre_cls
                improve_thresh = 2 * np.sum((pre_cls == cur_sam) * pre_cls) - np.sum(cur_sam)
                #improve = np.sum((pre_cls == cur) * pre_cls) / np.sum(cur) （>0.5）
                # Note that the two way calculating (improve) are equivalent
                improve_pred_thresh = np.sum((pre_cls == cur_sam) * pre_cls) / np.sum(pre_cls)
                if improve_thresh > 0 or improve_pred_thresh >= 0.85:

                    candidates.append(cur_sam)
                    seen.append(filename.path)
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

        plt.imsave(f'{save_path}/{name}.png', processed_mask, cmap='gray')
        if plot:
            plt.imshow(processed_mask, cmap='gray')
        #plt.show()





# merge = Merger(args.merge_type)
# segmentation_prediction = []

# for image_path in data_path_list:
#     # loads cam
#     computed_cam = etl.load_npy(os.path.join(args.path_prefix_output, args.data_folder_name, 'cam', args.cam_type), os.path.join(os.path.basename(image_path)+'.npy'))
#     # loads sam
#     # TODO: loads sam
#     segmentation_prediction.append(merge.merge(input_cam=computed_cam, input_sam=computed_sam))