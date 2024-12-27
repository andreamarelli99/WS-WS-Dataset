import cv2
import copy
import torch
import numpy as np
import torch.nn.functional as F




def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def get_strided_up_size(orig_size, stride):
    strided_size = get_strided_size(orig_size, stride)
    return strided_size[0]*stride, strided_size[1]*stride

def resize_for_tensors(tensors, size, mode='bilinear', align_corners=False):
    return F.interpolate(tensors, size, mode=mode, align_corners=align_corners)


def get_cam_of_scale(ori_image, scale, model_for_cam):
    # preprocessing
    image = copy.deepcopy(ori_image)

    ori_h, ori_w = image.shape[:2]
    new_size = (round(ori_w * scale), round(ori_h * scale))

    # Resize using OpenCV
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

    # image = normalize_fn(image)
    image = image.transpose((2, 0, 1))

    image = torch.from_numpy(image)
    flipped_image = image.flip(-1)

    images = torch.stack([image, flipped_image])
    images = images.cuda()

    # inferenece
    label, features = model_for_cam(images, with_cam=True)

    # print(f"label: {label}")

    # postprocessing
    cams = F.relu(features)
    cams = cams[0] + cams[1].flip(-1)

    return cams

def generate_cams(ori_image, cam_model, scales, normalize = True):
    
    ori_w, ori_h = ori_image.shape[0], ori_image.shape[1]

    strided_size = get_strided_size((ori_h, ori_w), 4)
    strided_up_size = get_strided_up_size((ori_h, ori_w), 16)

    cams_list = [get_cam_of_scale(ori_image, scale, cam_model) for scale in scales]
    hr = []    

    for i in range(3):

        hr_cams_list = [resize_for_tensors(cams.unsqueeze(0), strided_up_size)[0] for cams in cams_list]
        hr_cams = torch.sum(torch.stack(hr_cams_list), dim=0)[:, :ori_h, :ori_w]

        hr_cams = hr_cams[([i])]

        if normalize:
            hr_cams /= F.adaptive_max_pool2d(hr_cams, (1, 1)) + 1e-5

        # Visualize CAM on the original image
        
        hr.append(hr_cams[0])
        
    return hr







def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:

    img = cv2.resize(img, (int(mask.shape[1]), int(mask.shape[0])))


    heatmap = cv2.applyColorMap(np.uint8(255 * mask.cpu()), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(
            f"image_weight should be in the range [0, 1].\
                Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
