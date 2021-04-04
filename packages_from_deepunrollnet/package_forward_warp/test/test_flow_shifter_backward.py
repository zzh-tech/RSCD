import cv2
import time
import torch
import meshzoo
import numpy as np
from skimage import io
from forward_warp_package import ForwardShift

def main():
    crop_H=2
    crop_W=2
    im = io.imread('data/0.png')
    im = im.transpose(2, 0, 1)[1:2]
    im = torch.from_numpy(im).cuda().unsqueeze(0).float()/255.
    im = im[:,:,105:105+crop_H,100:100+crop_W].clone()
    
    B,C,H,W=im.size()
    flow = torch.zeros([1,2,H,W]).float().cuda()

    # create flow shifter
    flow_shifter = ForwardShift.create_with_implicit_mesh(B, C, H, W, 2, 1)

    # Forward
    im.requires_grad=True
    flow.requires_grad=True

    warped_image, _ = flow_shifter(im, flow)

    # backward analytical gradients
    loss = torch.sum(warped_image[:,:,0,0]**2)
    loss.backward()
    #print('loss', loss)

    grad_flow_analytic=flow.grad.clone()
    grad_im_analytic=im.grad.clone()

    print('\n\n======= Image check =======')
    print('im\n', im)
    print('warped_image\n', warped_image)
    
    # check image gradients
    print('\n\n======= Image gradients check =======')
    eps=1e-3
    grad_im_numerical=torch.zeros_like(grad_im_analytic)
    for r in range(H):
        for c in range(W):
            im2=im.clone()
            im2[:,:,r,c] += eps
            with torch.no_grad():
                warped_image2, _=flow_shifter(im2, flow)
            loss2=torch.sum(warped_image2[:,:,0,0]**2)
            grad_im_numerical[:,:,r,c]=(loss2-loss)/eps

    print('grad_im_analytic\n', grad_im_analytic)
    print('grad_im_numerical\n', grad_im_numerical)
    
    # check flow gradients
    print('\n\n======= Flow gradients check =======')
    eps=1e-2
    grad_flow_numerical=torch.zeros_like(grad_flow_analytic)
    for c in range(2):
        for y in range(H):
            for x in range(W):
                flow2=flow.clone()
                flow2[0,c,y,x] += eps
                with torch.no_grad():
                    warped_image2, _=flow_shifter(im, flow2)
                loss2=torch.sum(warped_image2[:,:,0,0]**2)
                grad_flow_numerical[:,c,y,x]=(loss2-loss)/eps

    print('grad_flow_analytic\n', grad_flow_analytic)
    print('grad_flow_numerical\n', grad_flow_numerical)
    

if __name__ == '__main__':
    main()
