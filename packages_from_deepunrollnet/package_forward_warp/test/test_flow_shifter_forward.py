import cv2
import time
import torch
import meshzoo
import numpy as np
from skimage import io
from forward_warp_package import ForwardShift

def main():
    H = 480
    W = 640
    
    im1 = io.imread('data/0.png')
    im1 = im1.transpose(2, 0, 1)
    im1 = torch.from_numpy(im1).cuda().unsqueeze(0).float()
    
    im2 = io.imread('data/10.png')
    im2 = im2.transpose(2, 0, 1)
    im2 = torch.from_numpy(im2).cuda().unsqueeze(0).float()
    
    im = torch.cat([im1, im2], dim=0)[:,:3,:,:]

    flow = torch.ones([2, H, W]).float().cuda().unsqueeze(0) * (40.0)
    flow[:, 0, :, :] = flow[:, 0, :, :] * 1

    flow = flow.repeat(2, 1, 1, 1)

    im = im / 255.
    flow_shifter = ForwardShift.create_with_implicit_mesh(2, 3, H, W, 4, 0.05)
    warped_image, mask = flow_shifter(im, flow)

    cv2.imshow('original_image', im.detach().cpu().numpy()[1].transpose(1, 2, 0))
    cv2.imshow('shifted_image', warped_image.detach().cpu().numpy()[1].transpose(1, 2, 0))
    cv2.imshow('mask', mask.detach().cpu().numpy()[1].transpose(1, 2, 0))
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
