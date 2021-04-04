import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from forward_warp_package_lib import Flow_forward_shift
from .utils import generate_2D_mesh

class ForwardShiftFunction(Function):
    '''
    Definition of differentiable rasterize operation
    Some parts of the code are implemented in CUDA
    Currently implemented only for cuda Tensors
    '''
    @staticmethod
    def forward(ctx, shifter, src_image, flow_src_to_tar):
        '''
        Forward pass
        '''
        ctx.shifter=shifter

        B,C,H,W=src_image.size()
        tar_image_w_I = torch.zeros_like(src_image)
        tar_image_w = torch.zeros([B,1,H,W], dtype=torch.float32, device=src_image.device)
        mask= torch.zeros([B,1,H,W], dtype=torch.float32, device=src_image.device)
        shifter.forward(src_image, flow_src_to_tar, tar_image_w_I, tar_image_w, mask)
        output_img = tar_image_w_I / (tar_image_w + 1e-8)

        return output_img, mask

    @staticmethod
    def backward(ctx, grad_tar_image, grad_mask):
        B,C,H,W=grad_tar_image.size()
        shifter=ctx.shifter
 
        grad_src_image = torch.zeros_like(grad_tar_image)
        grad_flow = torch.zeros([B,2,H,W], dtype=torch.float32, device=grad_tar_image.device);

        shifter.backward(grad_tar_image, grad_src_image, grad_flow)

        return None, grad_src_image, grad_flow


class ForwardShift(nn.Module):
    '''
    Wrapper around the autograd function
    Currently implemented only for cuda Tensors
    '''

    def __init__(self, mesh_grid, nchannels, kernel_radius, kernel_sigma2):
        super(ForwardShift, self).__init__()
        self.forward_warper = Flow_forward_shift(mesh_grid, nchannels, kernel_radius, kernel_sigma2)
        B,_,H,W=mesh_grid.size()
        self.B=B
        self.C=nchannels
        self.H=H
        self.W=W

    @classmethod
    def create_with_implicit_mesh(cls, B, C, H, W, kernel_radius=4, kernel_sigma2=0.5):
        grid, _ = generate_2D_mesh(H,W)
        grid = grid.int().cuda().unsqueeze(0)
        grid = grid.repeat(B, 1, 1, 1)
        return cls(grid, C, kernel_radius, kernel_sigma2)

    def forward(self, src_image, flow_src_to_tar):
        B,C,H,W=src_image.size()
        assert(B==self.B)
        assert(C==self.C)
        assert(H==self.H)
        assert(W==self.W)

        image, mask = ForwardShiftFunction.apply(
            self.forward_warper, 
            src_image, 
            flow_src_to_tar)

        return image, mask

