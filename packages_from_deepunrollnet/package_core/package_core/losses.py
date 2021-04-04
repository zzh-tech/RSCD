import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

import math

from .image_proc import *
from .geometry import *

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    def forward(self, output, target, weight=None, mean=False):
        error = torch.abs(output - target)
        if weight is not None:
            error = error * weight.float()
            if mean!=False:
                return error.sum() / weight.float().sum()
        if mean!=False:
            return error.mean()
        return error.sum()

class VariationLoss(nn.Module):
    def __init__(self, nc, grad_fn=Grid_gradient_central_diff):
        super(VariationLoss, self).__init__()
        self.grad_fn = grad_fn(nc)

    def forward(self, image, weight=None, mean=False):
        dx, dy = self.grad_fn(image)
        variation = dx**2 + dy**2

        if weight is not None:
            variation = variation * weight.float()
            if mean!=False:
                return variation.sum() / weight.sum()
        if mean!=False:
            return variation.mean()
        return variation.sum()

class EdgeAwareVariationLoss(nn.Module):
    def __init__(self, in1_nc, in2_nc, grad_fn=Grid_gradient_central_diff):
        super(EdgeAwareVariationLoss, self).__init__()
        self.in1_grad_fn = grad_fn(in1_nc)
        self.in2_grad_fn = grad_fn(in2_nc)

    def forward(self, in1, in2, mean=False):
        in1_dx, in1_dy = self.in1_grad_fn(in1)
        in2_dx, in2_dy = self.in2_grad_fn(in2)

        abs_in1_dx, abs_in1_dy = in1_dx.abs().sum(dim=1,keepdim=True), in1_dy.abs().sum(dim=1,keepdim=True)
        abs_in2_dx, abs_in2_dy = in2_dx.abs().sum(dim=1,keepdim=True), in2_dy.abs().sum(dim=1,keepdim=True)

        weight_dx, weight_dy = torch.exp(-abs_in2_dx), torch.exp(-abs_in2_dy)

        variation = weight_dx*abs_in1_dx + weight_dy*abs_in1_dy

        if mean!=False:
            return variation.mean()
        return variation.sum()

class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer:
                break
        return model
        
    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()
            
    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

class SSIMLoss(nn.Module):
    def __init__(self, nc=3):
        super(SSIMLoss, self).__init__()
        self.window_size=5
        self.gaussian_img_kernel = self.create_gaussian_window(self.window_size, nc).float()

    def create_gaussian_window(self, window_size, channel):
        def _gaussian(window_size, sigma):
            gauss = torch.Tensor(
                [math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()

        _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window@(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2, mask=None):
        self.gaussian_img_kernel = self.gaussian_img_kernel.to(img1.device)

        params = {'weight': self.gaussian_img_kernel,
              'groups': 3, 'padding': self.window_size//2}
        mu1 = F.conv2d(img1, **params)
        mu2 = F.conv2d(img2, **params)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, **params) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, **params) - mu2_sq
        sigma12 = F.conv2d(img1*img2, **params) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            ssim_map = ssim_map * mask

        return (1.-ssim_map.mean())*0.5

def EPE3D_loss(input_flow, target_flow, mask=None):
    """
    :param the estimated optical / scene flow
    :param the ground truth / target optical / scene flow
    :param the mask, the mask has value 0 for all areas that are invalid
    """
    invalid = None
    if mask is not None:
        invalid = 1.-mask

    epe_map = torch.norm(target_flow-input_flow,p=2,dim=1)
    B = epe_map.shape[0]

    invalid_flow = (target_flow != target_flow) # check Nan same as torch.isnan

    mask = (invalid_flow[:,0,:,:] | invalid_flow[:,1,:,:] | invalid_flow[:,2,:,:]) 
    if invalid is not None:
        mask = mask | (invalid.view(mask.shape) > 0)

    epes = []
    for idx in range(B):
        epe_sample = epe_map[idx][~mask[idx].data]
        if len(epe_sample) == 0:
            epes.append(torch.zeros(()).type_as(input_flow))
        else:
            epes.append(epe_sample.mean()) 

    return torch.stack(epes)

def compute_RT_EPE_loss(T_est, T_gt, depth0, K, mask=None): 
    """ Compute the epe point error of rotation & translation
    :param estimated rotation matrix Bx3x3
    :param estimated translation vector Bx3
    :param ground truth rotation matrix Bx3x3
    :param ground truth translation vector Bx3
    :param reference depth image, 
    :param camera intrinsic 
    """
    R_est = T_est[:,:3,:3]
    t_est = T_est[:,:3,3]
    R_gt = T_gt[:,:3,:3]
    t_gt = T_gt[:,:3,3]

    loss = 0
    if R_est.dim() > 3: # training time [batch, num_poses, rot_row, rot_col]
        rH, rW = 60, 80 # we train the algorithm using a downsized input, (since the size of the input is not super important at training time)

        B,C,H,W = depth0.shape
        rdepth = func.interpolate(depth0, size=(rH, rW), mode='bilinear')
        rmask = func.interpolate(mask.float(), size=(rH,rW), mode='bilinear')
        rK = K.clone()
        rK[:,0] *= float(rW) / W
        rK[:,1] *= float(rH) / H
        rK[:,2] *= float(rW) / W
        rK[:,3] *= float(rH) / H
        xyz = batch_inverse_project(rdepth, rK)
        flow_gt = batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        for idx in range(R_est.shape[1]):
            flow_est= batch_transform_xyz(xyz, R_est[:,idx], t_est[:,idx], get_Jacobian=False)
            loss += EPE3D_loss(flow_est, flow_gt.detach(), rmask) #* (1<<idx) scaling does not help that much
    else:
        xyz = batch_inverse_project(depth0, K)
        flow_gt = batch_transform_xyz(xyz, R_gt, t_gt, get_Jacobian=False)

        flow_est= batch_transform_xyz(xyz, R_est, t_est, get_Jacobian=False)
        loss = EPE3D_loss(flow_est, flow_gt, mask)

    loss = loss.sum()/float(len(loss))
    return loss
