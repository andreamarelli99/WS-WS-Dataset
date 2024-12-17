import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from glob import glob

import numpy as np
import cv2
from torch._C import _ImperativeEngine as ImperativeEngine


def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid 
        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,1,H,W).repeat(B,1,1,1)
        yy = yy.view(1,1,H,W).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1] 
        vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
        vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

        vgrid = vgrid.permute(0,2,3,1)        
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask, mask


def resize_flows_batch(flows_batch, new_shape):

    factor = torch.tensor(flows_batch.shape[-2:])/torch.tensor(new_shape)

    flows_batch = F.interpolate(flows_batch, size = new_shape, mode='bilinear', align_corners=False).cuda()

    flows_batch[:, 0] /= factor[1]
    flows_batch[:, 1] /= factor[0]

    return flows_batch


def get_visual_OF(flow):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3))
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor((hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


    return bgr


class VariableMeta(type):
    def __instancecheck__(cls, other):
        return isinstance(other, torch.Tensor)


class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):  # type: ignore[misc]
    _execution_engine = ImperativeEngine()
