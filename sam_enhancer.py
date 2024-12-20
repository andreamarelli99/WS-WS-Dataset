import yaml
from transformers import SamModel, SamProcessor, SamImageProcessor, SamConfig, SamVisionConfig, SamMaskDecoderConfig, SamPromptEncoderConfig
from typing import Optional, Union

class SAM:
    def __init__(self):
        """
        Parameters:
        -----------
        """
        self.processor = None
        self.model = None

    def construct_image_processor(self, config: Optional[Union[str, dict]] = None, from_pretrained: Optional[str] = None):
        """
        Wraps the SamProcessor class from the transformers library to process images.
        Parameters:
        -----------
        """
        if from_pretrained is not None:
            self.processor = SamProcessor.from_pretrained(from_pretrained)

        if config is None:
            self.processor = SamProcessor(SamImageProcessor(**config))
        else:
            if isinstance(config, str):
                with open(config, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                try:
                    config = dict(config)
                except:
                    raise TypeError("The config parameter must be a string or a dictionary-like.")
            self.processor = SamProcessor(SamImageProcessor(**config))

    def load_sam_model(self, config: Optional[Union[str, dict]] = None, from_pretrained: Optional[str] = None):
        """
        Wraps the SamModel class from the transformers library to load the model.
        Parameters:
        -----------
        """
        if from_pretrained is not None:
            self.model = SamModel.from_pretrained(from_pretrained)
        else:
            config = self.load_config(config)
            self.model = SamModel(config)

    def load_config(self, config: Optional[Union[str, dict]] = None):
        """
        Loads the configuration of the model from a yaml file.
        Parameters:
        -----------
        """
        
        if config is None:
            return SamConfig()
        else:
            vision_config = SamVisionConfig()
            mask_decoder_config = SamMaskDecoderConfig()
            prompt_encoder_config = SamPromptEncoderConfig()
            if isinstance(config, str):
                with open(config, 'r') as file:
                    config = yaml.safe_load(file)
            else:
                try:
                    config = dict(config)
                except:
                    raise TypeError("The config parameter must be a string or a dictionary-like.")
            if 'vision_config' in config:
                vision_config = SamVisionConfig(**config['vision_config'])
            if 'mask_decoder_config' in config:
                mask_decoder_config = SamMaskDecoderConfig(**config['mask_decoder_config'])
            if 'prompt_encoder_config' in config:
                prompt_encoder_config = SamPromptEncoderConfig(**config['prompt_encoder_config'])
            return SamConfig(vision_config=vision_config, mask_decoder_config=mask_decoder_config, prompt_encoder_config=prompt_encoder_config, **config)
        
    def compute_sam_masks(self, **kwargs):
        """
        Computes the mask of the image given a prompt.
        Parameters:
        -----------
        """
        if self.processor is not None:
            inputs = self.processor(**kwargs)
        else:
            inputs = kwargs

        return self.model(**inputs)
    
    def self_destruct(self):
        pass

import numpy as np
from MERGE.merge import Merger
from PIL import Image


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


sam = SAM()
sam.construct_image_processor(from_pretrained="facebook/sam-vit-base")
sam.load_sam_model(from_pretrained="facebook/sam-vit-base")
computed_sam = []

for image_path in data_path_list:
    # loads an image
    image = etl.get_image(image_path+'.png')
    # computes sam (returns a 3d array)
    # TODO: check what is returned
    sam_mask = sam.compute_sam_masks(image)
    # saves the sam
    # TODO: saving of the image

sam.self_destruct()




merge = Merger(args.merge_type)
segmentation_prediction = []

for image_path in data_path_list:
    # loads cam
    computed_cam = etl.load_npy(os.path.join(args.path_prefix_output, args.data_folder_name, 'cam', args.cam_type), os.path.join(os.path.basename(image_path)+'.npy'))
    # loads sam
    # TODO: loads sam
    segmentation_prediction.append(merge.merge(input_cam=computed_cam, input_sam=computed_sam))